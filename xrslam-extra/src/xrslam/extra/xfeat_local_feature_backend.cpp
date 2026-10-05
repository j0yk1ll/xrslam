#include <xrslam/extra/xfeat_local_feature_backend.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <xrslam/extra/opencv_image.h>
#include <xrslam/extra/poisson_disk_filter.h>

namespace xrslam::extra {

namespace {

class XFeatLocalFeatureBackend final : public LocalFeatureBackend {
    struct FeatureSet {
        std::vector<vector<2>> keypoints;
        std::vector<size_t> source_indices;
        std::vector<float> descriptors;
        size_t descriptor_dim = 0;
        size_t width = 0;
        size_t height = 0;
        double timestamp = 0.0;
    };

    struct DenseDescriptorMap {
        std::vector<float> descriptors;
        size_t channels = 0;
        size_t feature_width = 0;
        size_t feature_height = 0;
        size_t image_width = 0;
        size_t image_height = 0;
        double timestamp = 0.0;
    };

  public:
    explicit XFeatLocalFeatureBackend(const std::string &model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "xrslam-xfeat") {
        if (const char *ratio = std::getenv("XRSLAM_XFEAT_TRIGGER_RATIO");
            ratio && ratio[0] != '\0') {
            char *end = nullptr;
            const double parsed = std::strtod(ratio, &end);
            if (end != ratio && *end == '\0' &&
                std::isfinite(parsed) && parsed > 0.0 && parsed <= 1.0) {
                xfeat_trigger_ratio_ = parsed;
            } else {
                throw std::runtime_error(
                    "XRSLAM_XFEAT_TRIGGER_RATIO must be in (0, 1]");
            }
        }

        if (const char *stride =
                std::getenv("XRSLAM_LIGHTERGLUE_ANCHOR_STRIDE");
            stride && stride[0] != '\0') {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(stride, &end, 10);
            if (end != stride && *end == '\0' && parsed > 0) {
                recovery_anchor_stride_ = static_cast<size_t>(parsed);
            } else {
                throw std::runtime_error(
                    "XRSLAM_LIGHTERGLUE_ANCHOR_STRIDE must be a positive "
                    "integer");
            }
        }
        if (const char *top_k = std::getenv("XRSLAM_LIGHTERGLUE_TOP_K");
            top_k && top_k[0] != '\0') {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(top_k, &end, 10);
            if (end != top_k && *end == '\0' && parsed >= 64) {
                matcher_top_k_ = static_cast<size_t>(parsed);
            } else {
                throw std::runtime_error(
                    "XRSLAM_LIGHTERGLUE_TOP_K must be >= 64");
            }
        }
        if (const char *threshold =
                std::getenv("XRSLAM_LIGHTERGLUE_SCORE_THRESHOLD");
            threshold && threshold[0] != '\0') {
            char *end = nullptr;
            const double parsed = std::strtod(threshold, &end);
            if (end != threshold && *end == '\0' &&
                std::isfinite(parsed) && parsed >= 0.0 && parsed <= 1.0) {
                lighterglue_score_threshold_ = parsed;
            } else {
                throw std::runtime_error(
                    "XRSLAM_LIGHTERGLUE_SCORE_THRESHOLD must be in [0, 1]");
            }
        }

        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetInterOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(
            env_, model_path.c_str(), session_options_);

        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 3) {
            throw std::runtime_error(
                "unexpected XFeat ONNX interface; expected 1 input and at "
                "least 3 outputs");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ =
            session_->GetInputNameAllocated(0, allocator).get();
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            output_names_.emplace_back(
                session_->GetOutputNameAllocated(i, allocator).get());
        }

        std::fprintf(
            stderr,
            "[XFeat] ONNX model loaded (%zu outputs); adaptive detection "
            "enabled, trigger ratio=%.2f, tracking remains KLT.\n",
            output_names_.size(), xfeat_trigger_ratio_);

        if (const char *matcher_model =
                std::getenv("XRSLAM_LIGHTERGLUE_MODEL");
            matcher_model && matcher_model[0] != '\0') {
            matcher_session_ = std::make_unique<Ort::Session>(
                env_, matcher_model, session_options_);
            if (matcher_session_->GetInputCount() != 4 ||
                matcher_session_->GetOutputCount() != 2) {
                throw std::runtime_error(
                    "unexpected LighterGlue ONNX interface; expected "
                    "4 inputs and 2 outputs");
            }

            Ort::AllocatorWithDefaultOptions matcher_allocator;
            for (size_t i = 0; i < matcher_session_->GetInputCount(); ++i) {
                matcher_input_names_.emplace_back(
                    matcher_session_
                        ->GetInputNameAllocated(i, matcher_allocator)
                        .get());
            }
            for (size_t i = 0; i < matcher_session_->GetOutputCount(); ++i) {
                matcher_output_names_.emplace_back(
                    matcher_session_
                        ->GetOutputNameAllocated(i, matcher_allocator)
                        .get());
            }

            std::fprintf(
                stderr,
                "[XFeat] LighterGlue recovery enabled: %s "
                "(top_k=%zu score>=%.2f anchor_stride=%zu)\n",
                matcher_model, matcher_top_k_,
                lighterglue_score_threshold_, recovery_anchor_stride_);

            if (const char *dense_model =
                    std::getenv("XRSLAM_XFEAT_DENSE_MODEL");
                dense_model && dense_model[0] != '\0') {
                dense_session_ = std::make_unique<Ort::Session>(
                    env_, dense_model, session_options_);
                if (dense_session_->GetInputCount() != 1 ||
                    dense_session_->GetOutputCount() != 1) {
                    throw std::runtime_error(
                        "unexpected dense XFeat ONNX interface; expected "
                        "1 input and 1 output");
                }

                Ort::AllocatorWithDefaultOptions dense_allocator;
                dense_input_name_ =
                    dense_session_
                        ->GetInputNameAllocated(0, dense_allocator)
                        .get();
                dense_output_name_ =
                    dense_session_
                        ->GetOutputNameAllocated(0, dense_allocator)
                        .get();

                std::fprintf(
                    stderr,
                    "[XFeat] track-conditioned dense descriptors enabled: "
                    "%s\n",
                    dense_model);
            } else {
                std::fprintf(
                    stderr,
                    "[XFeat] LighterGlue track recovery disabled: set "
                    "XRSLAM_XFEAT_DENSE_MODEL to a dense-descriptor ONNX "
                    "model.\n");
            }
        }
    }

    void detect_keypoints(const Image *image,
                          std::vector<vector<2>> &keypoints,
                          size_t max_points,
                          double keypoint_distance) override {
        const auto *opencv_image =
            dynamic_cast<const OpenCvImage *>(image);
        if (!opencv_image || opencv_image->image.empty()) {
            image->detect_keypoints(keypoints, max_points,
                                    keypoint_distance);
            return;
        }

        // XFeat is a recovery/initialization detector in v2. Routine feature
        // replenishment stays on GFTT while KLT has retained a healthy track
        // population.
        const size_t survivor_count = keypoints.size();
        const size_t trigger_count =
            max_points == 0
                ? 0
                : std::max<size_t>(
                      1, static_cast<size_t>(
                             std::ceil(xfeat_trigger_ratio_ * max_points)));
        const bool use_xfeat =
            survivor_count == 0 ||
            (max_points > 0 && survivor_count < trigger_count);

        if (!use_xfeat) {
            ++opencv_detection_count_;
            image->detect_keypoints(keypoints, max_points,
                                    keypoint_distance);
            maybe_print_policy(survivor_count, trigger_count);
            return;
        }

        ++xfeat_detection_count_;
        try {
            detect_xfeat(opencv_image, keypoints, max_points,
                         keypoint_distance);
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                         "[XFeat] inference failed, falling back to OpenCV "
                         "detector for this frame: %s\n",
                         e.what());
            image->detect_keypoints(keypoints, max_points,
                                    keypoint_distance);
        }
        maybe_print_policy(survivor_count, trigger_count);
    }

    void detect_recovery_anchor_keypoints(
        const Image *image,
        std::vector<vector<2>> &keypoints,
        size_t max_points,
        double keypoint_distance) override {
        // 0052 showed that forcing native XFeat replenishment on every
        // retained anchor adds substantial CPU cost while almost none of the
        // later LighterGlue reference inliers retain exact track identity.
        // Keep that experiment available only as an explicit opt-in. With the
        // variable unset, anchor frames use the same frozen 0035-v2 adaptive
        // detector policy as every other replenishment frame.
        const char *seed_native =
            std::getenv("XRSLAM_XFEAT_SEED_RECOVERY_ANCHORS");
        if (!seed_native ||
            seed_native[0] != '1' ||
            seed_native[1] != '\0') {
            detect_keypoints(
                image, keypoints, max_points, keypoint_distance);
            return;
        }

        const auto *opencv_image =
            dynamic_cast<const OpenCvImage *>(image);
        if (!opencv_image || opencv_image->image.empty()) {
            image->detect_keypoints(
                keypoints, max_points, keypoint_distance);
            return;
        }

        try {
            ++xfeat_detection_count_;
            detect_xfeat(
                opencv_image, keypoints, max_points,
                keypoint_distance);
            ++recovery_anchor_detection_count_;
            if (recovery_anchor_detection_count_ <= 5 ||
                recovery_anchor_detection_count_ % 25 == 0) {
                std::fprintf(
                    stderr,
                    "[XFeat] recovery-anchor native seeding: "
                    "count=%zu frame_t=%.6f total_keypoints=%zu\n",
                    recovery_anchor_detection_count_,
                    image->t, keypoints.size());
            }
        } catch (const std::exception &e) {
            std::fprintf(
                stderr,
                "[XFeat] recovery-anchor inference failed; "
                "falling back to OpenCV for this frame: %s\n",
                e.what());
            image->detect_keypoints(
                keypoints, max_points, keypoint_distance);
        }
    }

    void track_keypoints(
        const Image *image, const Image *next_image,
        const std::vector<vector<2>> &curr_keypoints,
        std::vector<vector<2>> &next_keypoints,
        std::vector<char> &result_status) override {
        // Deliberately preserve XRSLAM's pyramidal LK + forward/backward
        // consistency path. Learned matching is a separate recovery path.
        image->track_keypoints(next_image, curr_keypoints, next_keypoints,
                               result_status);
    }

    size_t recovery_anchor_stride() const override {
        return matcher_session_ && dense_session_
                   ? recovery_anchor_stride_
                   : 0;
    }

    void cache_keyframe_features(const Image *image) override {
        if (!matcher_session_)
            return;
        if (cached_features(image))
            return;

        const auto *opencv_image =
            dynamic_cast<const OpenCvImage *>(image);
        if (!opencv_image || opencv_image->image.empty())
            return;

        try {
            FeatureSet features =
                extract_features(opencv_image, matcher_top_k_);
            cache_feature_set(image, std::move(features));
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                         "[XFeat] feature-cache extraction failed: %s\n",
                         e.what());
        }
    }

    bool has_cached_features(const Image *image) override {
        return cached_features(image) != nullptr;
    }

    bool match_keyframes(
        const Image *reference_image, const Image *current_image,
        std::vector<LocalFeatureMatch> &matches) override {
        matches.clear();
        if (!matcher_session_)
            return false;

        try {
            using clock = std::chrono::steady_clock;
            double reference_extract_ms = 0.0;
            double current_extract_ms = 0.0;
            double matcher_ms = 0.0;

            const FeatureSet *reference =
                cached_features(reference_image);
            const bool reference_was_cached = reference != nullptr;
            FeatureSet reference_storage;
            if (!reference) {
                const auto *opencv_image =
                    dynamic_cast<const OpenCvImage *>(reference_image);
                if (!opencv_image || opencv_image->image.empty()) {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] backend reject=reference_image_empty\n");
                    return false;
                }

                const auto begin = clock::now();
                reference_storage =
                    extract_features(opencv_image, matcher_top_k_);
                reference_extract_ms =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - begin)
                        .count();
                cache_feature_set(reference_image,
                                  std::move(reference_storage));
                reference = cached_features(reference_image);
            }

            const FeatureSet *current =
                cached_features(current_image);
            const bool current_was_cached = current != nullptr;
            FeatureSet current_storage;
            if (!current) {
                const auto *opencv_image =
                    dynamic_cast<const OpenCvImage *>(current_image);
                if (!opencv_image || opencv_image->image.empty()) {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] backend reject=current_image_empty\n");
                    return false;
                }

                const auto begin = clock::now();
                current_storage =
                    extract_features(opencv_image, matcher_top_k_);
                current_extract_ms =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - begin)
                        .count();
                cache_feature_set(current_image,
                                  std::move(current_storage));
                current = cached_features(current_image);
            }

            if (!current || !reference ||
                reference->keypoints.size() < 5 ||
                current->keypoints.size() < 5) {
                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] backend reject=insufficient_features "
                    "reference=%zu current=%zu\n",
                    reference ? reference->keypoints.size() : 0,
                    current ? current->keypoints.size() : 0);
                return false;
            }

            const auto matcher_begin = clock::now();
            run_lighterglue(*reference, *current, matches);
            matcher_ms =
                std::chrono::duration<double, std::milli>(
                    clock::now() - matcher_begin)
                    .count();

            std::fprintf(
                stderr,
                "[LighterGlueDiag] backend reference_features=%zu "
                "current_features=%zu reference_cached=%d current_cached=%d "
                "extract_ref_ms=%.3f extract_cur_ms=%.3f "
                "matcher_ms=%.3f filtered_matches=%zu\n",
                reference->keypoints.size(), current->keypoints.size(),
                reference_was_cached ? 1 : 0,
                current_was_cached ? 1 : 0,
                reference_extract_ms, current_extract_ms, matcher_ms,
                matches.size());

            return !matches.empty();
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                         "[XFeat] LighterGlue recovery failed: %s\n",
                         e.what());
            matches.clear();
            return false;
        }
    }

    bool match_track_keypoints(
        const Image *reference_image,
        const std::vector<vector<2>> &reference_points,
        const Image *current_image,
        std::vector<LocalFeatureMatch> &matches) override {
        matches.clear();
        if (!matcher_session_ || !dense_session_ ||
            reference_points.size() < 5) {
            return false;
        }

        try {
            using clock = std::chrono::steady_clock;
            double dense_extract_ms = 0.0;
            double reference_sample_ms = 0.0;
            double current_extract_ms = 0.0;
            double matcher_ms = 0.0;

            const auto *reference_cv =
                dynamic_cast<const OpenCvImage *>(reference_image);
            const auto *current_cv =
                dynamic_cast<const OpenCvImage *>(current_image);
            if (!reference_cv || reference_cv->image.empty() ||
                !current_cv || current_cv->image.empty()) {
                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] track_backend reject=image_empty\n");
                return false;
            }

            DenseDescriptorMap *dense =
                cached_dense_map(reference_image);
            const bool dense_was_cached = dense != nullptr;
            if (!dense) {
                const auto begin = clock::now();
                DenseDescriptorMap storage =
                    extract_dense_descriptor_map(reference_cv);
                dense_extract_ms =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - begin)
                        .count();
                cache_dense_map(reference_image, std::move(storage));
                dense = cached_dense_map(reference_image);
            }
            if (!dense)
                return false;

            const auto sample_begin = clock::now();
            FeatureSet reference =
                sample_track_features(*dense, reference_points);
            reference_sample_ms =
                std::chrono::duration<double, std::milli>(
                    clock::now() - sample_begin)
                    .count();

            // Use the sparse ONNX only as a detector. Its descriptor
            // outputs are not numerically compatible with the official-weight
            // dense XFeat map used for reference tracks, so resample current
            // descriptors from the same dense model as the reference side.
            const FeatureSet *current_sparse =
                cached_features(current_image);
            const bool current_sparse_was_cached =
                current_sparse != nullptr;
            FeatureSet current_sparse_storage;
            if (!current_sparse) {
                const auto begin = clock::now();
                current_sparse_storage =
                    extract_features(current_cv, matcher_top_k_);
                current_extract_ms =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - begin)
                        .count();
                cache_feature_set(current_image,
                                  std::move(current_sparse_storage));
                current_sparse = cached_features(current_image);
            }

            if (!current_sparse || reference.keypoints.size() < 5 ||
                current_sparse->keypoints.size() < 5) {
                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] track_backend "
                    "reject=insufficient_features reference=%zu current=%zu\n",
                    reference.keypoints.size(),
                    current_sparse ? current_sparse->keypoints.size() : 0);
                return false;
            }

            double current_dense_ms = 0.0;
            double current_sample_ms = 0.0;
            DenseDescriptorMap *current_dense =
                cached_dense_map(current_image);
            const bool current_dense_was_cached =
                current_dense != nullptr;
            if (!current_dense) {
                const auto begin = clock::now();
                DenseDescriptorMap storage =
                    extract_dense_descriptor_map(current_cv);
                current_dense_ms =
                    std::chrono::duration<double, std::milli>(
                        clock::now() - begin)
                        .count();
                cache_dense_map(current_image, std::move(storage));
                current_dense = cached_dense_map(current_image);
            }
            if (!current_dense) {
                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] track_backend "
                    "reject=current_dense_unavailable\n");
                return false;
            }

            const auto current_sample_begin = clock::now();
            FeatureSet current = sample_track_features(
                *current_dense, current_sparse->keypoints);
            current_sample_ms =
                std::chrono::duration<double, std::milli>(
                    clock::now() - current_sample_begin)
                    .count();

            if (current.keypoints.size() < 5) {
                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] track_backend "
                    "reject=current_dense_features current=%zu\n",
                    current.keypoints.size());
                return false;
            }

            const char *matcher_mode_env =
                std::getenv("XRSLAM_XFEAT_RECOVERY_MATCHER");
            const bool use_mnn =
                matcher_mode_env &&
                std::string(matcher_mode_env) == "mnn";
            const char *matcher_name =
                use_mnn ? "mnn" : "lighterglue";

            const auto matcher_begin = clock::now();
            if (use_mnn) {
                matches.clear();

                const size_t reference_count =
                    reference.keypoints.size();
                const size_t current_count =
                    current.keypoints.size();
                const size_t descriptor_dim =
                    reference.descriptor_dim;

                if (descriptor_dim == 0 ||
                    descriptor_dim != current.descriptor_dim ||
                    reference.descriptors.size() !=
                        reference_count * descriptor_dim ||
                    current.descriptors.size() !=
                        current_count * descriptor_dim) {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] track_backend "
                        "reject=mnn_descriptor_shape "
                        "reference_dim=%zu current_dim=%zu\n",
                        reference.descriptor_dim,
                        current.descriptor_dim);
                    return false;
                }

                std::vector<size_t> best_current(
                    reference_count, static_cast<size_t>(-1));
                std::vector<float> best_current_score(
                    reference_count, -2.0f);
                std::vector<size_t> best_reference(
                    current_count, static_cast<size_t>(-1));
                std::vector<float> best_reference_score(
                    current_count, -2.0f);

                for (size_t i = 0; i < reference_count; ++i) {
                    const float *reference_descriptor =
                        reference.descriptors.data() +
                        i * descriptor_dim;

                    for (size_t j = 0; j < current_count; ++j) {
                        const float *current_descriptor =
                            current.descriptors.data() +
                            j * descriptor_dim;

                        float cosine = 0.0f;
                        for (size_t d = 0; d < descriptor_dim; ++d) {
                            cosine += reference_descriptor[d] *
                                      current_descriptor[d];
                        }

                        if (cosine > best_current_score[i]) {
                            best_current_score[i] = cosine;
                            best_current[i] = j;
                        }
                        if (cosine > best_reference_score[j]) {
                            best_reference_score[j] = cosine;
                            best_reference[j] = i;
                        }
                    }
                }

                for (size_t i = 0; i < reference_count; ++i) {
                    const size_t j = best_current[i];
                    if (j == static_cast<size_t>(-1) ||
                        j >= current_count ||
                        best_reference[j] != i) {
                        continue;
                    }

                    LocalFeatureMatch match;
                    match.reference_index =
                        reference.source_indices.empty()
                            ? i
                            : reference.source_indices[i];
                    match.reference_point =
                        reference.keypoints[i];
                    match.current_point =
                        current.keypoints[j];
                    match.confidence =
                        0.5 *
                        (static_cast<double>(
                             best_current_score[i]) +
                         1.0);
                    matches.emplace_back(std::move(match));
                }
            } else {
                run_lighterglue(reference, current, matches);
            }

            matcher_ms =
                std::chrono::duration<double, std::milli>(
                    clock::now() - matcher_begin)
                    .count();

            std::fprintf(
                stderr,
                "[LighterGlueDiag] track_backend matcher=%s "
                "reference_tracks=%zu reference_descriptors=%zu "
                "current_features=%zu "
                "reference_dense_cached=%d current_sparse_cached=%d "
                "current_dense_cached=%d dense_ref_ms=%.3f "
                "sample_ref_ms=%.3f sparse_current_ms=%.3f "
                "dense_current_ms=%.3f sample_current_ms=%.3f "
                "matcher_ms=%.3f matches=%zu\n",
                matcher_name,
                reference_points.size(), reference.keypoints.size(),
                current.keypoints.size(), dense_was_cached ? 1 : 0,
                current_sparse_was_cached ? 1 : 0,
                current_dense_was_cached ? 1 : 0,
                dense_extract_ms, reference_sample_ms,
                current_extract_ms, current_dense_ms,
                current_sample_ms, matcher_ms, matches.size());

            return !matches.empty();
        } catch (const std::exception &e) {
            std::fprintf(
                stderr,
                "[XFeat] track-conditioned LighterGlue recovery failed: "
                "%s\n",
                e.what());
            matches.clear();
            return false;
        }
    }

  private:
    void maybe_print_policy(size_t survivor_count, size_t trigger_count) const {
        const size_t total =
            xfeat_detection_count_ + opencv_detection_count_;
        if (total > 0 && total % 50 == 0) {
            std::fprintf(
                stderr,
                "[XFeat] detector policy: XFeat=%zu OpenCV=%zu "
                "survivors=%zu trigger=%zu\n",
                xfeat_detection_count_, opencv_detection_count_,
                survivor_count, trigger_count);
        }
    }

    static float keypoint_coordinate(const Ort::Value &tensor,
                                     size_t index) {
        const auto info = tensor.GetTensorTypeAndShapeInfo();
        switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return tensor.GetTensorData<float>()[index];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return static_cast<float>(
                tensor.GetTensorData<double>()[index]);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return static_cast<float>(
                tensor.GetTensorData<int64_t>()[index]);
        default:
            throw std::runtime_error(
                "unsupported XFeat keypoint tensor type");
        }
    }

    static size_t parse_match_index(const Ort::Value &tensor,
                                    size_t index) {
        const auto info = tensor.GetTensorTypeAndShapeInfo();
        switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return static_cast<size_t>(
                tensor.GetTensorData<int64_t>()[index]);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return static_cast<size_t>(
                tensor.GetTensorData<int32_t>()[index]);
        default:
            throw std::runtime_error(
                "unsupported LighterGlue match tensor type");
        }
    }

    static double cubic_weight(double x) {
        // PyTorch grid_sample bicubic uses cubic convolution with alpha=-0.75.
        constexpr double alpha = -0.75;
        x = std::abs(x);
        if (x <= 1.0) {
            return (alpha + 2.0) * x * x * x -
                   (alpha + 3.0) * x * x + 1.0;
        }
        if (x < 2.0) {
            return alpha * x * x * x -
                   5.0 * alpha * x * x +
                   8.0 * alpha * x - 4.0 * alpha;
        }
        return 0.0;
    }

    DenseDescriptorMap extract_dense_descriptor_map(
        const OpenCvImage *opencv_image) {
        cv::Mat rgb;
        const cv::Mat &src = opencv_image->image;
        if (src.channels() == 1) {
            cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
        } else if (src.channels() == 3) {
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        } else {
            throw std::runtime_error("unsupported image channel count");
        }

        cv::Mat blob = cv::dnn::blobFromImage(
            rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false,
            CV_32F);
        const std::array<int64_t, 4> input_shape{
            1, 3, rgb.rows, rgb.cols};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, blob.ptr<float>(), blob.total(),
            input_shape.data(), input_shape.size());

        const char *input_names[] = {dense_input_name_.c_str()};
        const char *output_names[] = {dense_output_name_.c_str()};
        auto outputs = dense_session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1);
        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error(
                "invalid dense XFeat descriptor output");
        }

        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || shape[0] != 1 ||
            shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0) {
            throw std::runtime_error(
                "dense XFeat output must be float32 [1,C,H,W]");
        }

        DenseDescriptorMap result;
        result.channels = static_cast<size_t>(shape[1]);
        result.feature_height = static_cast<size_t>(shape[2]);
        result.feature_width = static_cast<size_t>(shape[3]);
        result.image_width = static_cast<size_t>(src.cols);
        result.image_height = static_cast<size_t>(src.rows);
        result.timestamp = opencv_image->t;

        const size_t count = info.GetElementCount();
        const float *data = outputs[0].GetTensorData<float>();
        result.descriptors.assign(data, data + count);
        return result;
    }

    FeatureSet sample_track_features(
        const DenseDescriptorMap &dense,
        const std::vector<vector<2>> &points) const {
        FeatureSet result;
        result.width = dense.image_width;
        result.height = dense.image_height;
        result.timestamp = dense.timestamp;
        result.descriptor_dim = dense.channels;
        result.keypoints.reserve(points.size());
        result.source_indices.reserve(points.size());
        result.descriptors.reserve(points.size() * dense.channels);

        if (dense.channels == 0 || dense.image_width < 2 ||
            dense.image_height < 2) {
            return result;
        }

        for (size_t point_index = 0; point_index < points.size();
             ++point_index) {
            const vector<2> &p = points[point_index];
            if (!std::isfinite(p.x()) || !std::isfinite(p.y()) ||
                p.x() < 0.0 || p.y() < 0.0 ||
                p.x() > static_cast<double>(dense.image_width - 1) ||
                p.y() > static_cast<double>(dense.image_height - 1)) {
                continue;
            }

            // Official sparse XFeat first resizes the input to dimensions
            // divisible by 32, samples descriptors in that resized coordinate
            // system, then scales keypoints back to the original image. Undo
            // that final keypoint scaling here before reproducing
            // InterpolateSparse2d/grid_sample(align_corners=false).
            const size_t resized_width =
                (dense.image_width / 32) * 32;
            const size_t resized_height =
                (dense.image_height / 32) * 32;
            if (resized_width < 2 || resized_height < 2)
                continue;

            const double resized_x =
                p.x() * resized_width /
                static_cast<double>(dense.image_width);
            const double resized_y =
                p.y() * resized_height /
                static_cast<double>(dense.image_height);

            const double fx =
                resized_x * dense.feature_width /
                    static_cast<double>(resized_width - 1) -
                0.5;
            const double fy =
                resized_y * dense.feature_height /
                    static_cast<double>(resized_height - 1) -
                0.5;
            const int x0 = static_cast<int>(std::floor(fx));
            const int y0 = static_cast<int>(std::floor(fy));

            std::vector<float> descriptor(dense.channels, 0.0f);
            for (int oy = -1; oy <= 2; ++oy) {
                const int sy = y0 + oy;
                const double wy = cubic_weight(fy - sy);
                if (wy == 0.0 || sy < 0 ||
                    sy >= static_cast<int>(dense.feature_height)) {
                    continue;
                }

                for (int ox = -1; ox <= 2; ++ox) {
                    const int sx = x0 + ox;
                    const double wx = cubic_weight(fx - sx);
                    if (wx == 0.0 || sx < 0 ||
                        sx >= static_cast<int>(dense.feature_width)) {
                        continue;
                    }

                    const float weight =
                        static_cast<float>(wx * wy);
                    for (size_t c = 0; c < dense.channels; ++c) {
                        const size_t offset =
                            (c * dense.feature_height +
                             static_cast<size_t>(sy)) *
                                dense.feature_width +
                            static_cast<size_t>(sx);
                        descriptor[c] +=
                            weight * dense.descriptors[offset];
                    }
                }
            }

            double norm_sq = 0.0;
            for (float value : descriptor)
                norm_sq += static_cast<double>(value) * value;
            if (norm_sq <= 1.0e-12)
                continue;

            const float inv_norm =
                static_cast<float>(1.0 / std::sqrt(norm_sq));
            for (float &value : descriptor)
                value *= inv_norm;

            result.keypoints.emplace_back(p);
            result.source_indices.emplace_back(point_index);
            result.descriptors.insert(
                result.descriptors.end(),
                descriptor.begin(), descriptor.end());
        }
        return result;
    }

    void cache_dense_map(
        const Image *image, DenseDescriptorMap dense) {
        auto existing = dense_cache_.find(image);
        if (existing == dense_cache_.end())
            dense_cache_order_.push_back(image);
        dense_cache_[image] = std::move(dense);

        while (dense_cache_order_.size() > dense_cache_limit_) {
            const Image *oldest = dense_cache_order_.front();
            dense_cache_order_.pop_front();
            if (oldest != image)
                dense_cache_.erase(oldest);
        }
    }

    DenseDescriptorMap *cached_dense_map(const Image *image) {
        auto it = dense_cache_.find(image);
        if (it == dense_cache_.end())
            return nullptr;
        if (std::abs(it->second.timestamp - image->t) > 1.0e-9) {
            dense_cache_.erase(it);
            return nullptr;
        }
        return &it->second;
    }

    FeatureSet extract_features(const OpenCvImage *opencv_image,
                                size_t top_k) {
        cv::Mat rgb;
        const cv::Mat &src = opencv_image->image;
        if (src.channels() == 1) {
            cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
        } else if (src.channels() == 3) {
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        } else {
            throw std::runtime_error("unsupported image channel count");
        }

        cv::Mat blob = cv::dnn::blobFromImage(
            rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false,
            CV_32F);
        const std::array<int64_t, 4> input_shape{
            1, 3, rgb.rows, rgb.cols};

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, blob.ptr<float>(), blob.total(),
            input_shape.data(), input_shape.size());

        const char *input_names[] = {input_name_.c_str()};
        std::vector<const char *> output_name_ptrs;
        output_name_ptrs.reserve(output_names_.size());
        for (const auto &name : output_names_)
            output_name_ptrs.emplace_back(name.c_str());

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_name_ptrs.data(), output_name_ptrs.size());

        if (outputs.size() < 3 || !outputs[0].IsTensor() ||
            !outputs[1].IsTensor() || !outputs[2].IsTensor()) {
            throw std::runtime_error("invalid XFeat ONNX outputs");
        }

        const auto kpt_info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto kpt_shape = kpt_info.GetShape();
        if (kpt_shape.empty() || kpt_shape.back() != 2) {
            throw std::runtime_error(
                "XFeat keypoint output must end in dimension 2");
        }

        const size_t point_count = kpt_info.GetElementCount() / 2;
        FeatureSet result;
        result.width = static_cast<size_t>(src.cols);
        result.height = static_cast<size_t>(src.rows);
        result.timestamp = opencv_image->t;
        if (point_count == 0)
            return result;

        const auto descriptor_info =
            outputs[1].GetTensorTypeAndShapeInfo();
        const auto descriptor_shape = descriptor_info.GetShape();
        if (descriptor_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            descriptor_shape.empty()) {
            throw std::runtime_error(
                "XFeat descriptor output must be float32");
        }
        const size_t descriptor_dim =
            static_cast<size_t>(descriptor_shape.back());
        if (descriptor_dim == 0 ||
            descriptor_info.GetElementCount() <
                point_count * descriptor_dim) {
            throw std::runtime_error(
                "invalid XFeat descriptor tensor shape");
        }
        const float *descriptors =
            outputs[1].GetTensorData<float>();

        const auto score_info = outputs[2].GetTensorTypeAndShapeInfo();
        if (score_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "XFeat score output must be float32");
        }
        const size_t score_count = score_info.GetElementCount();
        const float *scores = outputs[2].GetTensorData<float>();

        std::vector<size_t> order(point_count);
        std::iota(order.begin(), order.end(), size_t(0));
        if (score_count >= point_count) {
            std::stable_sort(
                order.begin(), order.end(),
                [scores](size_t a, size_t b) {
                    return scores[a] > scores[b];
                });
        }

        const size_t keep =
            top_k == 0 ? point_count : std::min(top_k, point_count);
        result.descriptor_dim = descriptor_dim;
        result.keypoints.reserve(keep);
        result.descriptors.reserve(keep * descriptor_dim);

        for (size_t rank = 0; rank < keep; ++rank) {
            const size_t i = order[rank];
            vector<2> p;
            p.x() = keypoint_coordinate(outputs[0], 2 * i);
            p.y() = keypoint_coordinate(outputs[0], 2 * i + 1);
            if (!std::isfinite(p.x()) || !std::isfinite(p.y()) ||
                p.x() < 0.0 || p.y() < 0.0 ||
                p.x() >= src.cols || p.y() >= src.rows) {
                continue;
            }

            result.keypoints.emplace_back(p);
            const float *descriptor =
                descriptors + i * descriptor_dim;
            result.descriptors.insert(
                result.descriptors.end(), descriptor,
                descriptor + descriptor_dim);
        }
        return result;
    }

    void cache_feature_set(const Image *image, FeatureSet features) {
        if (!matcher_session_ || features.keypoints.empty())
            return;

        auto existing = feature_cache_.find(image);
        if (existing == feature_cache_.end())
            cache_order_.push_back(image);
        feature_cache_[image] = std::move(features);

        while (cache_order_.size() > cache_limit_) {
            const Image *oldest = cache_order_.front();
            cache_order_.pop_front();
            if (oldest != image)
                feature_cache_.erase(oldest);
        }
    }

    FeatureSet *cached_features(const Image *image) {
        auto it = feature_cache_.find(image);
        if (it == feature_cache_.end())
            return nullptr;
        if (std::abs(it->second.timestamp - image->t) > 1.0e-9) {
            feature_cache_.erase(it);
            return nullptr;
        }
        return &it->second;
    }

    void run_lighterglue(
        const FeatureSet &reference, const FeatureSet &current,
        std::vector<LocalFeatureMatch> &matches) {
        if (reference.descriptor_dim != current.descriptor_dim ||
            reference.descriptor_dim == 0) {
            throw std::runtime_error(
                "incompatible XFeat descriptor dimensions");
        }

        const size_t n0 = reference.keypoints.size();
        const size_t n1 = current.keypoints.size();
        const size_t dim = reference.descriptor_dim;

        std::vector<float> kpts0(2 * n0);
        std::vector<float> kpts1(2 * n1);
        const double scale0 =
            std::max(reference.width, reference.height) / 2.0;
        const double scale1 =
            std::max(current.width, current.height) / 2.0;
        const double cx0 = reference.width / 2.0;
        const double cy0 = reference.height / 2.0;
        const double cx1 = current.width / 2.0;
        const double cy1 = current.height / 2.0;

        for (size_t i = 0; i < n0; ++i) {
            kpts0[2 * i] =
                static_cast<float>((reference.keypoints[i].x() - cx0) /
                                   scale0);
            kpts0[2 * i + 1] =
                static_cast<float>((reference.keypoints[i].y() - cy0) /
                                   scale0);
        }
        for (size_t i = 0; i < n1; ++i) {
            kpts1[2 * i] =
                static_cast<float>((current.keypoints[i].x() - cx1) /
                                   scale1);
            kpts1[2 * i + 1] =
                static_cast<float>((current.keypoints[i].y() - cy1) /
                                   scale1);
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> kpts0_shape{
            1, static_cast<int64_t>(n0), 2};
        const std::array<int64_t, 3> kpts1_shape{
            1, static_cast<int64_t>(n1), 2};
        const std::array<int64_t, 3> desc0_shape{
            1, static_cast<int64_t>(n0), static_cast<int64_t>(dim)};
        const std::array<int64_t, 3> desc1_shape{
            1, static_cast<int64_t>(n1), static_cast<int64_t>(dim)};

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, kpts0.data(), kpts0.size(),
            kpts0_shape.data(), kpts0_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, kpts1.data(), kpts1.size(),
            kpts1_shape.data(), kpts1_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float *>(reference.descriptors.data()),
            reference.descriptors.size(),
            desc0_shape.data(), desc0_shape.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float *>(current.descriptors.data()),
            current.descriptors.size(),
            desc1_shape.data(), desc1_shape.size()));

        std::vector<const char *> input_names;
        for (const auto &name : matcher_input_names_)
            input_names.emplace_back(name.c_str());
        std::vector<const char *> output_names;
        for (const auto &name : matcher_output_names_)
            output_names.emplace_back(name.c_str());

        auto outputs = matcher_session_->Run(
            Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
            inputs.size(), output_names.data(), output_names.size());

        if (outputs.size() != 2 || !outputs[0].IsTensor() ||
            !outputs[1].IsTensor()) {
            throw std::runtime_error(
                "invalid LighterGlue ONNX outputs");
        }

        const auto match_info =
            outputs[0].GetTensorTypeAndShapeInfo();
        const auto match_shape = match_info.GetShape();
        if (match_shape.empty() || match_shape.back() != 2) {
            throw std::runtime_error(
                "LighterGlue matches must end in dimension 2");
        }
        const size_t match_count =
            match_info.GetElementCount() / 2;

        const auto score_info =
            outputs[1].GetTensorTypeAndShapeInfo();
        if (score_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "LighterGlue scores must be float32");
        }
        const size_t score_count = score_info.GetElementCount();
        const float *scores = outputs[1].GetTensorData<float>();

        matches.reserve(match_count);
        for (size_t i = 0; i < match_count; ++i) {
            if (i >= score_count ||
                scores[i] < lighterglue_score_threshold_)
                continue;

            const size_t i0 = parse_match_index(outputs[0], 2 * i);
            const size_t i1 = parse_match_index(outputs[0], 2 * i + 1);
            if (i0 >= n0 || i1 >= n1)
                continue;

            LocalFeatureMatch match;
            match.reference_index =
                reference.source_indices.empty()
                    ? i0
                    : reference.source_indices[i0];
            match.reference_point = reference.keypoints[i0];
            match.current_point = current.keypoints[i1];
            match.confidence = scores[i];
            matches.emplace_back(match);
        }
    }

    void detect_xfeat(const OpenCvImage *opencv_image,
                      std::vector<vector<2>> &keypoints,
                      size_t max_points,
                      double keypoint_distance) {
        const size_t extraction_limit =
            matcher_session_
                ? std::max(max_points, matcher_top_k_)
                : max_points;
        FeatureSet features =
            extract_features(opencv_image, extraction_limit);

        PoissonDiskFilter<2> filter(keypoint_distance);
        filter.preset_points(keypoints);

        constexpr double border = 20.0;
        const size_t candidate_limit =
            max_points == 0
                ? features.keypoints.size()
                : std::min(max_points, features.keypoints.size());
        for (size_t i = 0; i < candidate_limit; ++i) {
            const auto &p = features.keypoints[i];
            if (p.x() < border || p.y() < border ||
                p.x() >= opencv_image->image.cols - border ||
                p.y() >= opencv_image->image.rows - border) {
                continue;
            }
            if (filter.insert_point(p))
                keypoints.emplace_back(p);
        }

        if (matcher_session_)
            cache_feature_set(opencv_image, std::move(features));
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;

    std::unique_ptr<Ort::Session> matcher_session_;
    std::vector<std::string> matcher_input_names_;
    std::vector<std::string> matcher_output_names_;

    std::unique_ptr<Ort::Session> dense_session_;
    std::string dense_input_name_;
    std::string dense_output_name_;

    std::unordered_map<const Image *, FeatureSet> feature_cache_;
    std::deque<const Image *> cache_order_;

    std::unordered_map<const Image *, DenseDescriptorMap> dense_cache_;
    std::deque<const Image *> dense_cache_order_;
    size_t dense_cache_limit_ = 4;

    size_t recovery_anchor_stride_ = 10;
    size_t cache_limit_ = 8;
    size_t matcher_top_k_ = 512;
    double lighterglue_score_threshold_ = 0.1;

    double xfeat_trigger_ratio_ = 0.5;
    size_t xfeat_detection_count_ = 0;
    size_t opencv_detection_count_ = 0;
    size_t recovery_anchor_detection_count_ = 0;
};

} // namespace

std::shared_ptr<LocalFeatureBackend>
make_xfeat_local_feature_backend(const std::string &model_path) {
    return std::make_shared<XFeatLocalFeatureBackend>(model_path);
}

} // namespace xrslam::extra
