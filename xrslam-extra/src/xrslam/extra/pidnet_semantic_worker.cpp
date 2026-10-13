#include <xrslam/extra/pidnet_semantic_worker.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace xrslam::extra {

PidNetSemanticWorker::PidNetSemanticWorker(
    const std::string &model_path, const std::string &device,
    std::uint32_t classes, int canvas_width, int canvas_height,
    std::string taxonomy, std::string dump_dir, std::size_t max_pending,
    bool enable_dense, double dense_voxel_size_m, int dense_pixel_stride,
    DenseSemanticQualityOptions quality, std::string model_id,
    std::vector<std::string> class_names)
    : segmenter_(model_path, device, classes, canvas_width, canvas_height),
      taxonomy_(std::move(taxonomy)), model_id_(std::move(model_id)),
      class_names_(std::move(class_names)), dump_dir_(std::move(dump_dir)),
      max_pending_(max_pending), dense_pixel_stride_(dense_pixel_stride),
      dense_quality_(quality), landmark_map_(classes) {
    if (max_pending_ == 0 || max_pending_ > 64)
        throw std::invalid_argument("semantic worker queue must have 1..64 slots");
    if (taxonomy_.empty() || model_id_.empty())
        throw std::invalid_argument("semantic taxonomy/model identity must not be empty");
    if (class_names_.size() != classes)
        throw std::invalid_argument("semantic class names must match model class count");
    if (!dump_dir_.empty())
        std::filesystem::create_directories(dump_dir_);
    if (dense_pixel_stride_ < 1 || dense_pixel_stride_ > 16)
        throw std::invalid_argument("dense pixel stride must be 1..16");
    dense_quality_.validate();
    if (dense_quality_.enabled && !enable_dense)
        throw std::invalid_argument("dense quality requires dense semantics");
    if (enable_dense)
        dense_map_ = std::make_unique<DenseSemanticVoxelMap>(
            classes, dense_voxel_size_m, 250000);
    thread_ = std::thread([this] { run(); });
}

PidNetSemanticWorker::~PidNetSemanticWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wakeup_.notify_all();
    // Drain the bounded queue so accepted jobs are not silently discarded.
    if (thread_.joinable()) thread_.join();
}

void PidNetSemanticWorker::wait_until_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this] { return pending_.empty() && !processing_; });
}

bool PidNetSemanticWorker::submit(SemanticKeyframeSnapshot &&snapshot) noexcept {
    try {
        if (!snapshot.image.valid() || snapshot.session_id == 0 ||
            !std::isfinite(snapshot.timestamp))
            return false;
        const auto id = std::make_pair(snapshot.session_id, snapshot.frame_id);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || records_.count(id) ||
            pending_.size() >= max_pending_) {
            if (pending_.size() >= max_pending_)
                ++dropped_;
            return false;
        }
        pending_.emplace_back(std::move(snapshot));
        wakeup_.notify_one();
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<SemanticFusedLandmark> PidNetSemanticWorker::find_landmark(
    std::uint64_t session_id, std::size_t track_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.find(session_id, track_id);
}

std::vector<SemanticFusedLandmark> PidNetSemanticWorker::landmark_snapshot(
    std::optional<std::uint64_t> session_id, bool stable_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.snapshot(session_id, stable_only);
}

std::size_t PidNetSemanticWorker::landmark_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.size();
}

std::size_t PidNetSemanticWorker::stable_landmark_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.stable_count();
}

std::vector<DenseSemanticVoxelPoint> PidNetSemanticWorker::dense_snapshot(
    bool stable_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dense_map_) return {};
    if (!dense_quality_.enabled) return dense_map_->snapshot(stable_only);
    return dense_map_->snapshot_filtered(
        stable_only, dense_quality_.min_views,
        dense_quality_.min_neighbors);
}

double PidNetSemanticWorker::dense_voxel_size_m() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dense_map_ ? dense_map_->voxel_size_m() : 0.0;
}

std::size_t PidNetSemanticWorker::record_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
}

std::vector<SemanticKeyframeHit> PidNetSemanticWorker::query_class(
    const std::string &name, double min_mean_confidence,
    double min_pixel_coverage, std::size_t limit) const {
    std::vector<SemanticKeyframeHit> hits;
    if (!std::isfinite(min_mean_confidence) ||
        !std::isfinite(min_pixel_coverage) ||
        min_mean_confidence < 0 || min_mean_confidence > 1 ||
        min_pixel_coverage < 0 || min_pixel_coverage > 1 || limit == 0)
        return hits;
    const auto it=std::find(class_names_.begin(),class_names_.end(),name);
    if(it==class_names_.end())return hits;
    const auto class_id=static_cast<std::uint32_t>(
        std::distance(class_names_.begin(),it));
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[key, rec] : records_) {
        if (class_id >= rec.classes.size()) continue;
        const auto &summary=rec.classes[class_id];
        if (!summary.pixel_count ||
            summary.mean_confidence < min_mean_confidence ||
            summary.coverage < min_pixel_coverage)continue;
        hits.push_back(SemanticKeyframeHit{
            rec.session_id,rec.frame_id,rec.timestamp,class_id,
            summary.coverage,summary.mean_confidence,
            rec.body_position,rec.body_quaternion_xyzw,
            rec.taxonomy,rec.model_id,false});
    }
    std::sort(hits.begin(),hits.end(),[](const auto &a,const auto &b){
        if(a.timestamp!=b.timestamp)return a.timestamp>b.timestamp;
        if(a.session_id!=b.session_id)return a.session_id<b.session_id;
        return a.frame_id<b.frame_id;
    });
    if(hits.size()>limit)hits.resize(limit);
    return hits;
}

std::optional<SemanticKeyframeRecord> PidNetSemanticWorker::find(
    std::uint64_t session_id, std::uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find({session_id, frame_id});
    if (it == records_.end()) return std::nullopt;
    return it->second;
}

void PidNetSemanticWorker::run() noexcept {
    for (;;) {
        SemanticKeyframeSnapshot job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wakeup_.wait(lock, [this] {return stopping_ || !pending_.empty();});
            if (pending_.empty()) return;
            job = std::move(pending_.front());
            pending_.pop_front();
            processing_ = true;
        }
        try {
            process(std::move(job));
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                "[PIDNetSemantic] event=error session=%llu frame=%llu error=%s\n",
                static_cast<unsigned long long>(job.session_id),
                static_cast<unsigned long long>(job.frame_id), e.what());
        } catch (...) {
            std::fprintf(stderr, "[PIDNetSemantic] event=error unknown=1\n");
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            processing_ = false;
            if (pending_.empty()) idle_cv_.notify_all();
        }
    }
}

void PidNetSemanticWorker::process(SemanticKeyframeSnapshot job) {
    const int type = CV_MAKETYPE(CV_8U, static_cast<int>(job.image.channels));
    // Job owns its packed pixels until inference has completed.
    const cv::Mat input(static_cast<int>(job.image.height),
                        static_cast<int>(job.image.width), type,
                        job.image.bytes.data());
    const SemanticSegmentationResult output = segmenter_.segment(input);
    if (output.label.size() != input.size() ||
        output.confidence.size() != input.size() ||
        output.label.type() != CV_8UC1 ||
        output.confidence.type() != CV_32FC1)
        throw std::runtime_error("PIDNet worker: unexpected mask dimensions/type");

    SemanticKeyframeRecord rec;
    rec.session_id = job.session_id;
    rec.frame_id = job.frame_id;
    rec.timestamp = job.timestamp;
    rec.width = job.image.width;
    rec.height = job.image.height;
    rec.num_classes = output.num_classes;
    rec.taxonomy = taxonomy_;
    rec.model_id = model_id_;
    rec.body_position = job.body_position;
    rec.body_quaternion_xyzw = job.body_quaternion_xyzw;
    rec.camera_position = job.camera_position;
    rec.camera_quaternion_xyzw = job.camera_quaternion_xyzw;
    rec.K_row_major = job.K_row_major;

    rec.classes.resize(output.num_classes);
    const double area = static_cast<double>(output.label.total());
    std::vector<double> confidence_sum(output.num_classes, 0.0);
    for (int y=0;y<output.label.rows;++y) {
        for (int x=0;x<output.label.cols;++x) {
            const auto id=output.label.at<std::uint8_t>(y,x);
            if(id>=output.num_classes)
                throw std::runtime_error("PIDNet emitted class ID out of taxonomy bounds");
            const double confidence=output.confidence.at<float>(y,x);
            if(!std::isfinite(confidence)||confidence<0||confidence>1)
                throw std::runtime_error("PIDNet emitted invalid confidence");
            rec.classes[id].class_id=id;
            ++rec.classes[id].pixel_count;
            confidence_sum[id]+=confidence;
        }
    }
    for (std::uint32_t i=0;i<output.num_classes;++i) {
        auto &v=rec.classes[i];
        if(v.pixel_count)v.mean_confidence=confidence_sum[i]/v.pixel_count;
        v.coverage=static_cast<double>(v.pixel_count)/area;
    }

    if (!cv::imencode(".png", output.label, rec.labels_png))
        throw std::runtime_error("PIDNet worker: failed to encode label PNG");
    cv::Mat quantized_confidence;
    output.confidence.convertTo(quantized_confidence, CV_8UC1, 255.0);
    if (!cv::imencode(".png", quantized_confidence, rec.confidence_png))
        throw std::runtime_error("PIDNet worker: failed to encode confidence PNG");

    rec.landmarks.reserve(job.observations.size());
    for (const auto &obs : job.observations) {
        if (!std::isfinite(obs.u) || !std::isfinite(obs.v)) continue;
        const int x = static_cast<int>(std::lround(obs.u));
        const int y = static_cast<int>(std::lround(obs.v));
        if (x < 0 || y < 0 || x >= output.label.cols ||
            y >= output.label.rows) continue;
        rec.landmarks.push_back({obs.track_id, obs.keypoint_index,
            output.label.at<std::uint8_t>(y, x),
            output.confidence.at<float>(y, x),
            obs.position_session_world});
    }

    if (!dump_dir_.empty()) {
        const auto base = std::to_string(job.session_id) + "_" +
                          std::to_string(job.frame_id);
        const std::filesystem::path dir(dump_dir_);
        if (!cv::imwrite((dir / (base + "_labels.png")).string(), output.label))
            throw std::runtime_error("PIDNet worker: failed to write label PNG");
    }

    // A track may appear multiple times in a keyframe; choose the
    // highest-confidence pixel per track. One keyframe is *one* vote per
    // landmark, not one vote per feature/keypoint alias.
    std::map<std::size_t, SemanticLandmarkVote> votes;
    for (const auto &sample : rec.landmarks) {
        SemanticLandmarkVote vote;
        vote.session_id = rec.session_id;
        vote.frame_id = rec.frame_id;
        vote.track_id = sample.track_id;
        vote.label = sample.label;
        vote.confidence = sample.confidence;
        vote.position_session_world = sample.position_session_world;
        const auto it = votes.find(sample.track_id);
        if (it == votes.end() || vote.confidence > it->second.confidence)
            votes[sample.track_id] = vote;
    }

    // Dense surface samples: back-project *registered sensor-metric*
    // depth through the archived intrinsic matrix and camera-to-world pose.
    // Work is outside the worker's map mutex and off the tracker thread.
    std::vector<DenseSemanticSample> dense_samples;
    std::size_t depth_edge_rejections = 0;
    if (dense_map_ && job.depth.valid() &&
        job.depth.width == job.image.width &&
        job.depth.height == job.image.height) {
        const double fx = job.K_row_major[0], fy = job.K_row_major[4];
        const double cx = job.K_row_major[2], cy = job.K_row_major[5];
        if (std::isfinite(fx) && std::isfinite(fy) && fx > 0 && fy > 0 &&
            std::isfinite(cx) && std::isfinite(cy)) {
            const int w = static_cast<int>(job.image.width);
            const int h = static_cast<int>(job.image.height);
            for (int y = 0; y < h; y += dense_pixel_stride_)
                for (int x = 0; x < w; x += dense_pixel_stride_) {
                    const auto at = static_cast<std::size_t>(y) * w + x;
                    const float z = job.depth.meters[at];
                    if (!std::isfinite(z) || z < 0.25f || z > 5.0f)
                        continue;
                    if (dense_quality_.enabled &&
                        !dense_semantic_depth_is_consistent(
                            job.depth, static_cast<std::uint32_t>(x),
                            static_cast<std::uint32_t>(y), dense_quality_)) {
                        ++depth_edge_rejections;
                        continue;
                    }
                    const float conf = output.confidence.at<float>(y, x);
                    if (!std::isfinite(conf) || conf < 0.50f)
                        continue;
                    const std::array<double, 3> camera_xyz{
                        (x - cx) * z / fx, (y - cy) * z / fy, z};
                    DenseSemanticSample sample;
                    sample.session_id = job.session_id;
                    sample.frame_id = job.frame_id;
                    sample.xyz = DenseSemanticVoxelMap::camera_to_session_world(
                        job.camera_quaternion_xyzw, job.camera_position,
                        camera_xyz);
                    sample.label = output.label.at<std::uint8_t>(y, x);
                    sample.confidence = conf;
                    const std::size_t p = at * job.image.channels;
                    if (job.image.channels == 1) {
                        sample.rgb = {job.image.bytes[p], job.image.bytes[p],
                                      job.image.bytes[p]};
                    } else {
                        // Canonical camera images use OpenCV BGR(A) order.
                        sample.rgb = {job.image.bytes[p+2], job.image.bytes[p+1],
                                      job.image.bytes[p]};
                    }
                    dense_samples.emplace_back(sample);
                }
        }
    }

    const auto samples = rec.landmarks.size();
    std::size_t count = 0, fused = 0, landmark_count = 0, stable_count = 0;
    std::size_t dense_fused = 0, dense_count = 0, dense_at_capacity = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = std::make_pair(rec.session_id, rec.frame_id);
        if (records_.count(id)) return; // never double-count a keyframe
        for (const auto &[track, vote] : votes) {
            if (landmark_map_.observe(vote)) ++fused;
        }
        if (dense_map_) {
            for (const auto &sample : dense_samples)
                if (dense_map_->observe(sample)) ++dense_fused;
            dense_count = dense_map_->size();
            dense_at_capacity = dense_map_->capacity_rejections();
        }
        records_.emplace(id, std::move(rec));
        count = records_.size();
        landmark_count = landmark_map_.size();
        stable_count = landmark_map_.stable_count();
    }
    if (dense_map_)
        std::fprintf(stderr,
            "[DenseSemantic] event=frame session=%llu frame=%llu "
            "metric_depth=%d candidates=%zu fused=%zu voxels=%zu "
            "capacity_rejections=%zu depth_edge_rejections=%zu "
            "state_mutation=0\n",
            static_cast<unsigned long long>(job.session_id),
            static_cast<unsigned long long>(job.frame_id),
            job.depth.valid() ? 1 : 0, dense_samples.size(),
            dense_fused, dense_count, dense_at_capacity,
            depth_edge_rejections);
    std::fprintf(stderr,
        "[PIDNetSemantic] event=processed session=%llu frame=%llu "
        "t=%.9f size=%ux%u sampled=%zu stored=%zu "
        "fused_views=%zu landmark_tracks=%zu stable_tracks=%zu "
        "state_mutation=0\n",
        static_cast<unsigned long long>(job.session_id),
        static_cast<unsigned long long>(job.frame_id), job.timestamp,
        job.image.width, job.image.height, samples, count,
        fused, landmark_count, stable_count);
}

} // namespace xrslam::extra
