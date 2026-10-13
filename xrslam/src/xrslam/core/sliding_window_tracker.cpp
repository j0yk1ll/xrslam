#include <xrslam/common.h>
#include <xrslam/core/detail.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/core/keyframe_archive.h>
#include <xrslam/core/place_graph_4dof_shadow.h>
#include <xrslam/core/frontend_worker.h>
#include <xrslam/core/relocalization_runtime_profile.h>
#include <xrslam/core/sliding_window_tracker.h>
#include <xrslam/estimation/solver.h>
#include <xrslam/geometry/lie_algebra.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/geometry/pnp.h>
#include <xrslam/inspection.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/map/track.h>
#include <xrslam/utility/unique_timer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>

namespace xrslam {

namespace {


size_t place_retrieval_top_k() {
    constexpr size_t default_top_k = 20;

    const char *value =
        std::getenv("XRSLAM_PLACE_RETRIEVAL_TOP_K");
    if (!value || value[0] == '\0')
        return default_top_k;

    char *end = nullptr;
    const unsigned long long parsed =
        std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 ||
        parsed >
            static_cast<unsigned long long>(
                std::numeric_limits<size_t>::max())) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalShadow] invalid "
                "XRSLAM_PLACE_RETRIEVAL_TOP_K=%s; using %zu\n",
                value, default_top_k);
            warned = true;
        }
        return default_top_k;
    }

    return static_cast<size_t>(parsed);
}




size_t orb_association_lag() {
    constexpr size_t default_lag = 20;

    const char *value =
        std::getenv("XRSLAM_ORB_ASSOCIATION_LAG");
    if (!value || value[0] == '\0')
        return default_lag;

    char *end = nullptr;
    const unsigned long long parsed =
        std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 ||
        parsed >
            static_cast<unsigned long long>(
                std::numeric_limits<size_t>::max())) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(
                stderr,
                "[OrbAssociationShadow] invalid "
                "XRSLAM_ORB_ASSOCIATION_LAG=%s; using %zu\n",
                value, default_lag);
            warned = true;
        }
        return default_lag;
    }

    return static_cast<size_t>(parsed);
}

size_t orb_pnp_max_hamming() {
    constexpr size_t default_max_hamming = 60;
    constexpr size_t maximum_max_hamming = 256;

    const char *value =
        std::getenv("XRSLAM_ORB_PNP_MAX_HAMMING");
    if (!value || value[0] == '\0')
        return default_max_hamming;

    char *end = nullptr;
    const unsigned long long parsed =
        std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' ||
        parsed > maximum_max_hamming) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(
                stderr,
                "[OrbPnPShadow] invalid "
                "XRSLAM_ORB_PNP_MAX_HAMMING=%s; using %zu\n",
                value, default_max_hamming);
            warned = true;
        }
        return default_max_hamming;
    }

    return static_cast<size_t>(parsed);
}

struct HammingDistanceStats {
    size_t min = 0;
    size_t p25 = 0;
    size_t median = 0;
    size_t p75 = 0;
    size_t max = 0;
    double mean = 0.0;
};

HammingDistanceStats hamming_distance_stats(
    const std::vector<LocalDescriptorMatch> &matches) {
    HammingDistanceStats stats;
    if (matches.empty())
        return stats;

    std::vector<size_t> distances;
    distances.reserve(matches.size());
    double total = 0.0;
    for (const auto &match : matches) {
        distances.emplace_back(match.distance);
        total += static_cast<double>(match.distance);
    }
    std::sort(distances.begin(), distances.end());

    stats.min = distances.front();
    stats.p25 = distances[
        (distances.size() - 1) / 4];
    stats.median = distances[
        (distances.size() - 1) / 2];
    stats.p75 = distances[
        3 * (distances.size() - 1) / 4];
    stats.max = distances.back();
    stats.mean =
        total / static_cast<double>(distances.size());
    return stats;
}


const char *local_descriptor_type_name(LocalDescriptorType type) {
    switch (type) {
    case LocalDescriptorType::BINARY_U8:
        return "binary_u8";
    case LocalDescriptorType::FLOAT32:
        return "float32";
    }
    return "unknown";
}

double camera_rotation_delta_deg(
    const PoseState &a, const PoseState &b) {
    const matrix<3> R_delta =
        a.q.conjugate().matrix() * b.q.matrix();
    const double cosine = std::max(
        -1.0,
        std::min(1.0, (R_delta.trace() - 1.0) * 0.5));
    return std::acos(cosine) * 180.0 / M_PI;
}

struct PnpDirectionalConditioningDiagnostic {
    size_t observation_count = 0;
    bool finite = false;
    double reprojection_rmse_px =
        std::numeric_limits<double>::quiet_NaN();
    double information_condition =
        std::numeric_limits<double>::quiet_NaN();
    vector<3> sigma_translation_ref_m =
        vector<3>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    vector<3> sigma_rotation_world_deg =
        vector<3>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    double estimated_pixel_sigma_px =
        std::numeric_limits<double>::quiet_NaN();
    double graph_covariance_condition =
        std::numeric_limits<double>::quiet_NaN();
    matrix<4> graph_covariance_unit =
        matrix<4>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    vector<4> graph_covariance_eigenvalues =
        vector<4>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    vector<4> graph_weak_mode =
        vector<4>::Constant(
            std::numeric_limits<double>::quiet_NaN());
    double graph_sigma_rms =
        std::numeric_limits<double>::quiet_NaN();
    double graph_residual_gain = 1.0;
};

PnpDirectionalConditioningDiagnostic
pnp_directional_conditioning(
    const PoseState &body_pose,
    const ExtrinsicParams &camera,
    const matrix<3> &K,
    const PoseState &reference_body_pose,
    const std::vector<vector<3>> &landmarks_world,
    const std::vector<vector<2>> &observations_pixel) {
    PnpDirectionalConditioningDiagnostic diagnostic;
    if (landmarks_world.size() < 4 ||
        landmarks_world.size() != observations_pixel.size()) {
        return diagnostic;
    }

    diagnostic.observation_count = landmarks_world.size();

    const auto project =
        [&camera, &K](
            const PoseState &pose,
            const vector<3> &landmark_world,
            vector<2> &pixel) {
            const vector<3> point_center =
                pose.q.conjugate() *
                (landmark_world - pose.p);
            const vector<3> point_camera =
                camera.q_cs.conjugate() *
                (point_center - camera.p_cs);
            if (!point_camera.allFinite() ||
                point_camera.z() <= 1.0e-6) {
                return false;
            }
            pixel = apply_k(point_camera, K);
            return pixel.allFinite();
        };

    Eigen::MatrixXd J(
        2 * static_cast<Eigen::Index>(landmarks_world.size()),
        6);
    double squared_error = 0.0;

    for (size_t i = 0; i < landmarks_world.size(); ++i) {
        vector<2> projected;
        if (!project(body_pose, landmarks_world[i], projected)) {
            return diagnostic;
        }
        squared_error +=
            (projected - observations_pixel[i]).squaredNorm();
    }

    diagnostic.reprojection_rmse_px =
        std::sqrt(
            squared_error /
            static_cast<double>(landmarks_world.size()));

    constexpr double translation_step_m = 1.0e-4;
    constexpr double rotation_step_rad = 1.0e-5;
    const matrix<3> reference_R =
        reference_body_pose.q.matrix();

    for (size_t column = 0; column < 6; ++column) {
        PoseState plus = body_pose;
        PoseState minus = body_pose;
        const double step =
            column < 3
                ? translation_step_m
                : rotation_step_rad;

        if (column < 3) {
            const vector<3> direction =
                reference_R.col(
                    static_cast<Eigen::Index>(column));
            plus.p += step * direction;
            minus.p -= step * direction;
        } else {
            vector<3> axis = vector<3>::Zero();
            axis(static_cast<Eigen::Index>(column - 3)) = 1.0;
            const quaternion plus_delta(
                Eigen::AngleAxisd(step, axis));
            const quaternion minus_delta(
                Eigen::AngleAxisd(-step, axis));
            plus.q = plus_delta * body_pose.q;
            minus.q = minus_delta * body_pose.q;
            plus.q.normalize();
            minus.q.normalize();
        }

        for (size_t i = 0; i < landmarks_world.size(); ++i) {
            vector<2> projected_plus;
            vector<2> projected_minus;
            if (!project(
                    plus, landmarks_world[i], projected_plus) ||
                !project(
                    minus, landmarks_world[i], projected_minus)) {
                return diagnostic;
            }
            J.block<2, 1>(
                2 * static_cast<Eigen::Index>(i),
                static_cast<Eigen::Index>(column)) =
                (projected_plus - projected_minus) /
                (2.0 * step);
        }
    }

    const Eigen::Matrix<double, 6, 6> information =
        J.transpose() * J;
    const Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, 6, 6>> eigen_solver(information);
    if (eigen_solver.info() != Eigen::Success) {
        return diagnostic;
    }

    const auto eigenvalues = eigen_solver.eigenvalues();
    const double min_information = eigenvalues.minCoeff();
    const double max_information = eigenvalues.maxCoeff();
    if (!std::isfinite(min_information) ||
        !std::isfinite(max_information) ||
        min_information <= 0.0 ||
        max_information <= 0.0 ||
        min_information <= max_information * 1.0e-12) {
        return diagnostic;
    }

    diagnostic.information_condition =
        max_information / min_information;
    const Eigen::Matrix<double, 6, 6> covariance =
        information.ldlt().solve(
            Eigen::Matrix<double, 6, 6>::Identity());
    if (!covariance.allFinite()) {
        return diagnostic;
    }

    for (size_t i = 0; i < 3; ++i) {
        diagnostic.sigma_translation_ref_m(
            static_cast<Eigen::Index>(i)) =
            std::sqrt(std::max(
                0.0,
                covariance(
                    static_cast<Eigen::Index>(i),
                    static_cast<Eigen::Index>(i))));
        diagnostic.sigma_rotation_world_deg(
            static_cast<Eigen::Index>(i)) =
            std::sqrt(std::max(
                0.0,
                covariance(
                    static_cast<Eigen::Index>(i + 3),
                    static_cast<Eigen::Index>(i + 3)))) *
            180.0 / M_PI;
    }

    const double residual_dof = std::max(
        1.0,
        2.0 * static_cast<double>(landmarks_world.size()) - 6.0);
    diagnostic.estimated_pixel_sigma_px =
        std::sqrt(squared_error / residual_dof);

    constexpr std::array<Eigen::Index, 4> graph_indices = {
        0, 1, 2, 5};
    matrix<4> graph_covariance_rad;
    for (Eigen::Index r = 0; r < 4; ++r) {
        for (Eigen::Index c = 0; c < 4; ++c) {
            graph_covariance_rad(r, c) =
                covariance(graph_indices[r], graph_indices[c]);
        }
    }

    matrix<4> graph_unit_scale = matrix<4>::Identity();
    graph_unit_scale(3, 3) = 0.1 * 180.0 / M_PI;
    diagnostic.graph_covariance_unit =
        graph_unit_scale * graph_covariance_rad * graph_unit_scale;

    const Eigen::SelfAdjointEigenSolver<matrix<4>>
        graph_covariance_eigen_solver(
            diagnostic.graph_covariance_unit);
    if (graph_covariance_eigen_solver.info() != Eigen::Success) {
        return diagnostic;
    }

    diagnostic.graph_covariance_eigenvalues =
        graph_covariance_eigen_solver.eigenvalues();
    const double min_graph_covariance =
        diagnostic.graph_covariance_eigenvalues.minCoeff();
    const double max_graph_covariance =
        diagnostic.graph_covariance_eigenvalues.maxCoeff();
    if (!std::isfinite(min_graph_covariance) ||
        !std::isfinite(max_graph_covariance) ||
        min_graph_covariance <= 0.0 ||
        max_graph_covariance <= 0.0) {
        return diagnostic;
    }

    diagnostic.graph_covariance_condition =
        max_graph_covariance / min_graph_covariance;
    diagnostic.graph_sigma_rms =
        diagnostic.estimated_pixel_sigma_px *
        std::sqrt(std::max(
            0.0, diagnostic.graph_covariance_unit.trace() / 4.0));
    // Frozen development calibration: median graph-space RMS
    // uncertainty of accepted V2_03 place edges, fixed before
    // held-out V2_02 evaluation.
    constexpr double graph_reference_sigma_rms =
        0.03049885884247795;
    if (std::isfinite(diagnostic.graph_sigma_rms) &&
        diagnostic.graph_sigma_rms > 0.0) {
        diagnostic.graph_residual_gain = std::min(
            1.0,
            graph_reference_sigma_rms / diagnostic.graph_sigma_rms);
    }
    diagnostic.graph_weak_mode =
        graph_covariance_eigen_solver.eigenvectors().col(3);
    Eigen::Index weak_mode_largest_index = 0;
    diagnostic.graph_weak_mode.cwiseAbs().maxCoeff(
        &weak_mode_largest_index);
    if (diagnostic.graph_weak_mode(weak_mode_largest_index) < 0.0) {
        diagnostic.graph_weak_mode *= -1.0;
    }

    diagnostic.finite =
        diagnostic.sigma_translation_ref_m.allFinite() &&
        diagnostic.sigma_rotation_world_deg.allFinite() &&
        diagnostic.graph_covariance_unit.allFinite() &&
        diagnostic.graph_covariance_eigenvalues.allFinite() &&
        diagnostic.graph_weak_mode.allFinite() &&
        std::isfinite(diagnostic.reprojection_rmse_px) &&
        std::isfinite(diagnostic.estimated_pixel_sigma_px) &&
        std::isfinite(diagnostic.graph_sigma_rms) &&
        std::isfinite(diagnostic.graph_residual_gain) &&
        std::isfinite(diagnostic.information_condition) &&
        std::isfinite(diagnostic.graph_covariance_condition);
    return diagnostic;
}

} // namespace

SlidingWindowTracker::SlidingWindowTracker(std::unique_ptr<Map> keyframe_map,
                                           std::shared_ptr<Config> config)
    : map(std::move(keyframe_map)), config(config) {
    for (size_t j = 1; j < map->frame_num(); ++j) {
        Frame *frame_i = map->get_frame(j - 1);
        Frame *frame_j = map->get_frame(j);
        if (config->has_imu()) {
            frame_j->preintegration.integrate(frame_j->image->t,
                                              frame_i->motion.bg,
                                              frame_i->motion.ba, true, true);
        } else {
            frame_j->tag(FT_FIX_MOTION) = true;
        }
    }
}

SlidingWindowTracker::~SlidingWindowTracker() = default;

void SlidingWindowTracker::mirror_frame(Map *feature_tracking_map,
                                        size_t frame_id) {
    Frame *keyframe = map->get_frame(map->frame_num() - 1);
    Frame *new_frame_i = keyframe;
    if (!keyframe->subframes.empty()) {
        new_frame_i = keyframe->subframes.back().get();
    }

    size_t frame_index_i =
        feature_tracking_map->frame_index_by_id(new_frame_i->id());
    size_t frame_index_j = feature_tracking_map->frame_index_by_id(frame_id);

    if (frame_index_i == nil() || frame_index_j == nil())
        return;

    Frame *old_frame_i = feature_tracking_map->get_frame(frame_index_i);
    Frame *old_frame_j = feature_tracking_map->get_frame(frame_index_j);

    std::unique_ptr<Frame> curr_frame = std::move(old_frame_j->clone());
    std::vector<ImuData> &new_data = curr_frame->preintegration.data;
    for (size_t index = frame_index_j - 1; index > frame_index_i; --index) {
        std::vector<ImuData> old_data =
            feature_tracking_map->get_frame(index)->preintegration.data;
        new_data.insert(new_data.begin(), old_data.begin(), old_data.end());
    }

    map->attach_frame(curr_frame->clone());
    Frame *new_frame_j = map->get_frame(map->frame_num() - 1);

    for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
        if (Track *track = old_frame_i->get_track(ki)) {
            if (size_t kj = track->get_keypoint_index(old_frame_j);
                kj != nil()) {
                Track *new_track = new_frame_i->get_track(ki, map.get());
                new_track->add_keypoint(new_frame_j, kj);
                track->tag(TT_TRASH) =
                    new_track->tag(TT_TRASH) && !new_track->tag(TT_STATIC);
            }
        }
    }

    map->prune_tracks([](const Track *track) {
        return track->tag(TT_TRASH) && !track->tag(TT_STATIC);
    });

    if (config->has_imu()) {
        new_frame_j->preintegration.integrate(new_frame_j->image->t,
                                              new_frame_i->motion.bg,
                                              new_frame_i->motion.ba, true, true);
        new_frame_j->preintegration.predict(new_frame_i, new_frame_j);
    } else {
        new_frame_j->pose = new_frame_i->pose;
        new_frame_j->motion = new_frame_i->motion;
        new_frame_j->tag(FT_FIX_MOTION) = true;
    }
}

void SlidingWindowTracker::archive_optimized_keyframes() {
    const bool semantic_enabled = semantic_session_id_ != 0 && detail &&
        detail->semantic_keyframe_sink();
    if (!relocalization_v1_profile_enabled() && !semantic_enabled)
        return;

    LocalDescriptorExtractor *local_extractor = nullptr;
    if (relocalization_v1_profile_enabled()) {
        if (!detail) {
            std::fprintf(stderr,
                         "[LocalDescriptorShadow] reject=no_detail\n");
        } else {
            local_extractor =
                detail->local_descriptor_extractor();
            if (!local_extractor) {
                std::fprintf(
                    stderr,
                    "[LocalDescriptorShadow] reject=no_extractor\n");
            }
        }
    }

    std::string active_before_slide;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        if (!frame || !frame->tag(FT_KEYFRAME))
            continue;

        if (!active_before_slide.empty())
            active_before_slide += ';';
        active_before_slide += std::to_string(frame->id());
    }

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        if (!frame || !frame->tag(FT_KEYFRAME) || !frame->image)
            continue;

        ArchivedKeyframe archived;
        archived.frame_id = frame->id();
        archived.timestamp = frame->image->t;
        archived.body_pose = frame->pose;
        archived.camera_pose = frame->get_pose(frame->camera);
        archived.K = frame->K;
        archived.observations.reserve(frame->keypoint_num());

        for (size_t keypoint_index = 0;
             keypoint_index < frame->keypoint_num();
             ++keypoint_index) {
            Track *track = frame->get_track(keypoint_index);
            if (!track || track->tag(TT_TRASH) ||
                !track->all_tagged(
                    TT_VALID, TT_TRIANGULATED, TT_STATIC)) {
                continue;
            }

            const vector<3> bearing =
                frame->get_keypoint(keypoint_index);
            const vector<3> landmark_world =
                track->get_landmark_point();
            const vector<2> pixel =
                apply_k(bearing, frame->K);

            if (!bearing.allFinite() ||
                !landmark_world.allFinite() ||
                !pixel.allFinite()) {
                continue;
            }

            archived.observations.push_back(
                ArchivedLandmarkObservation{
                    track->id(),
                    keypoint_index,
                    bearing,
                    pixel,
                    landmark_world});
        }

        const double timestamp = archived.timestamp;
        const size_t observation_count =
            archived.observations.size();

        const ArchivedKeyframe *existing =
            keyframe_archive_.get(frame->id());
        const bool local_descriptors_already_complete =
            existing && existing->local_descriptors_complete;

        if (local_extractor &&
            !local_descriptors_already_complete) {
            const bool source_available_before =
                frame->image->has_place_recognition_source();

            if (!source_available_before) {
                std::fprintf(
                    stderr,
                    "[LocalDescriptorShadow] frame_id=%zu t=%.9f "
                    "reject=source_unavailable "
                    "source_available_before=0\n",
                    frame->id(), frame->image->t);
            } else {
                std::vector<vector<2>> points;
                points.reserve(archived.observations.size());
                for (const auto &observation :
                     archived.observations) {
                    points.emplace_back(observation.pixel);
                }

                const auto begin =
                    std::chrono::steady_clock::now();
                try {
                    LocalDescriptorSet descriptors =
                        local_extractor->extract(
                            *frame->image, points);
                    const size_t descriptor_count =
                        descriptors.size();

                    bool mapping_valid =
                        descriptors.valid() &&
                        descriptors.type ==
                            local_extractor->type() &&
                        descriptors.dimension ==
                            local_extractor->dimension();

                    size_t previous_source_index = 0;
                    for (size_t row = 0;
                         mapping_valid &&
                         row < descriptor_count; ++row) {
                        const size_t source_index =
                            descriptors.source_indices[row];
                        if (source_index >=
                                archived.observations.size() ||
                            (row > 0 &&
                             source_index <=
                                 previous_source_index)) {
                            mapping_valid = false;
                            break;
                        }
                        previous_source_index = source_index;
                    }

                    if (!points.empty() &&
                        descriptor_count == 0) {
                        mapping_valid = false;
                    }

                    if (!mapping_valid) {
                        std::fprintf(
                            stderr,
                            "[LocalDescriptorShadow] "
                            "frame_id=%zu t=%.9f "
                            "reject=invalid_descriptors "
                            "observations=%zu descriptors=%zu "
                            "type=%s dimension=%zu "
                            "expected_type=%s "
                            "expected_dimension=%zu\n",
                            frame->id(), frame->image->t,
                            archived.observations.size(),
                            descriptor_count,
                            local_descriptor_type_name(
                                descriptors.type),
                            descriptors.dimension,
                            local_descriptor_type_name(
                                local_extractor->type()),
                            local_extractor->dimension());
                    } else {
                        archived.local_descriptors =
                            std::move(descriptors);
                        archived.local_descriptors_complete =
                            true;
                        ++local_descriptor_keyframe_count_;

                        const bool place_ready =
                            !relocalization_v1_profile_enabled() ||
                            place_keyframes_.find(
                                static_cast<PlaceKey>(
                                    frame->id()));
                        if (place_ready &&
                            !relocalization_v1_profile_enabled()) {
                            frame->image
                                ->retain_place_recognition_source(
                                    false);
                        }

                        const bool source_available_after =
                            frame->image
                                ->has_place_recognition_source();
                        const double extract_ms =
                            std::chrono::duration<
                                double, std::milli>(
                                std::chrono::steady_clock::now() -
                                begin)
                                .count();
                        const double coverage =
                            archived.observations.empty()
                                ? 1.0
                                : static_cast<double>(
                                      archived.local_descriptors
                                          .size()) /
                                      static_cast<double>(
                                          archived.observations
                                              .size());

                        std::fprintf(
                            stderr,
                            "[LocalDescriptorShadow] "
                            "frame_id=%zu t=%.9f "
                            "observations=%zu descriptors=%zu "
                            "coverage=%.6f type=%s "
                            "dimension=%zu "
                            "source_available_before=1 "
                            "source_available_after=%d "
                            "extract_ms=%.3f "
                            "keyframe_count=%zu\n",
                            frame->id(), frame->image->t,
                            archived.observations.size(),
                            archived.local_descriptors.size(),
                            coverage,
                            local_descriptor_type_name(
                                archived.local_descriptors.type),
                            archived.local_descriptors.dimension,
                            source_available_after ? 1 : 0,
                            extract_ms,
                            local_descriptor_keyframe_count_);
                    }
                } catch (const std::exception &e) {
                    std::fprintf(
                        stderr,
                        "[LocalDescriptorShadow] "
                        "frame_id=%zu t=%.9f "
                        "reject=extract_exception error=%s\n",
                        frame->id(), frame->image->t,
                        e.what());
                }
            }
        }

        const bool inserted =
            keyframe_archive_.upsert(std::move(archived));

        // Only first archive insertion schedules semantic inference. Later
        // upserts update optimized geometry without destroying the separate
        // session/keyframe semantic record. This never alters VIO states.
        if (inserted && semantic_enabled) {
            auto staged = staged_semantic_images_.find(frame->id());
            const ArchivedKeyframe *source =
                keyframe_archive_.get(frame->id());
            if (staged != staged_semantic_images_.end() && source) {
                SemanticKeyframeSnapshot snapshot;
                snapshot.session_id = semantic_session_id_;
                snapshot.frame_id = static_cast<std::uint64_t>(frame->id());
                snapshot.timestamp = source->timestamp;
                snapshot.image = std::move(staged->second);
                auto depth = staged_semantic_depths_.find(frame->id());
                if (depth != staged_semantic_depths_.end())
                    snapshot.depth = std::move(depth->second);
                for (int d = 0; d < 3; ++d) {
                    snapshot.body_position[d] = source->body_pose.p[d];
                    snapshot.camera_position[d] = source->camera_pose.p[d];
                }
                snapshot.body_quaternion_xyzw = {
                    source->body_pose.q.x(), source->body_pose.q.y(),
                    source->body_pose.q.z(), source->body_pose.q.w()};
                snapshot.camera_quaternion_xyzw = {
                    source->camera_pose.q.x(), source->camera_pose.q.y(),
                    source->camera_pose.q.z(), source->camera_pose.q.w()};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        snapshot.K_row_major[3 * r + c] = source->K(r, c);
                snapshot.observations.reserve(source->observations.size());
                for (const auto &obs : source->observations)
                    snapshot.observations.push_back({
                        obs.track_id, obs.keypoint_index,
                        obs.pixel.x(), obs.pixel.y(),
                        {obs.landmark_world.x(),
                         obs.landmark_world.y(),
                         obs.landmark_world.z()}});
                const bool queued = detail->semantic_keyframe_sink()->submit(
                    std::move(snapshot));
                std::fprintf(stderr,
                    "[PIDNetSemantic] event=keyframe session=%llu "
                    "frame=%zu t=%.9f queued=%d state_mutation=0\n",
                    static_cast<unsigned long long>(semantic_session_id_),
                    frame->id(), source->timestamp, queued ? 1 : 0);
            } else {
                std::fprintf(stderr,
                    "[PIDNetSemantic] event=missing_source session=%llu "
                    "frame=%zu state_mutation=0\n",
                    static_cast<unsigned long long>(semantic_session_id_),
                    frame->id());
            }
            // Without place retrieval, the retained source is no longer
            // needed once a private image snapshot has been submitted.
            if (!relocalization_v1_profile_enabled())
                frame->image->retain_place_recognition_source(false);
        }

        // 0112b exact graph-node export. Emit every archive upsert so
        // offline graph reconstruction can retain the latest optimized
        // snapshot for each keyframe instead of approximating it from
        // the output trajectory. This is diagnostic-only.
        const ArchivedKeyframe *graph_node =
            keyframe_archive_.get(frame->id());
        if (graph_node) {
            std::fprintf(
                stderr,
                "[PlaceGraphNodeShadow] frame_id=%zu t=%.9f "
                "body_p=%.9f,%.9f,%.9f "
                "body_q=%.9f,%.9f,%.9f,%.9f "
                "inserted=%d refreshed=%d state_mutation=0\n",
                graph_node->frame_id, graph_node->timestamp,
                graph_node->body_pose.p.x(),
                graph_node->body_pose.p.y(),
                graph_node->body_pose.p.z(),
                graph_node->body_pose.q.x(),
                graph_node->body_pose.q.y(),
                graph_node->body_pose.q.z(),
                graph_node->body_pose.q.w(),
                inserted ? 1 : 0, inserted ? 0 : 1);
        }

        if (relocalization_v1_profile_enabled()) {
            const ArchivedKeyframe *stored =
                keyframe_archive_.get(frame->id());
            if (stored &&
                stored->local_descriptors_complete &&
                frame->image
                    ->has_place_recognition_source() &&
                orb_association_processed_
                    .emplace(frame->id())
                    .second) {
                diagnose_orb_association(frame);

                const bool place_ready =
                    !relocalization_v1_profile_enabled() ||
                    place_keyframes_.find(
                        static_cast<PlaceKey>(
                            frame->id()));
                const bool retrieval_verification_pending =
                    std::any_of(
                        pending_place_retrieval_candidates_.begin(),
                        pending_place_retrieval_candidates_.end(),
                        [frame](const PendingPlaceRetrieval &pending) {
                            return pending.frame_id == frame->id();
                        });
                if (place_ready &&
                    !retrieval_verification_pending) {
                    frame->image
                        ->retain_place_recognition_source(
                            false);
                }
            }
        }

        if (inserted) {
            std::fprintf(
                stderr,
                "[PlaceKeyframeArchive] frame_id=%zu t=%.9f "
                "observations=%zu archive_size=%zu "
                "active_before_slide=%s\n",
                frame->id(),
                timestamp,
                observation_count,
                keyframe_archive_.size(),
                active_before_slide.c_str());
        }
    }
}

void SlidingWindowTracker::diagnose_orb_association(
    Frame *frame) {
    if (!relocalization_v1_profile_enabled() ||
        !frame || !frame->image || !detail) {
        return;
    }

    LocalDescriptorExtractor *extractor =
        detail->local_descriptor_extractor();
    if (!extractor ||
        extractor->type() !=
            LocalDescriptorType::BINARY_U8) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu "
            "reject=binary_extractor_unavailable\n",
            frame->id());
        return;
    }

    const auto &order =
        keyframe_archive_.insertion_order();
    const auto current_it =
        std::find(
            order.begin(), order.end(), frame->id());
    if (current_it == order.end()) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu "
            "reject=current_not_archived\n",
            frame->id());
        return;
    }

    const size_t current_archive_index =
        static_cast<size_t>(
            current_it - order.begin());
    const size_t lag = orb_association_lag();
    if (current_archive_index < lag) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu t=%.9f "
            "lag=%zu archive_index=%zu "
            "skip=insufficient_history\n",
            frame->id(), frame->image->t,
            lag, current_archive_index);
        return;
    }

    const size_t reference_frame_id =
        order[current_archive_index - lag];
    const ArchivedKeyframe *reference =
        keyframe_archive_.get(reference_frame_id);
    if (!reference ||
        !reference->local_descriptors_complete ||
        !reference->local_descriptors.valid() ||
        reference->local_descriptors.type !=
            LocalDescriptorType::BINARY_U8 ||
        reference->local_descriptors.dimension !=
            extractor->dimension()) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu "
            "reference=%zu lag=%zu "
            "reject=reference_descriptors_unavailable\n",
            frame->id(), reference_frame_id, lag);
        return;
    }
    if (reference->local_descriptors.size() == 0) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu "
            "reference=%zu lag=%zu "
            "skip=reference_empty\n",
            frame->id(), reference_frame_id, lag);
        return;
    }

    std::vector<vector<2>> current_points;
    current_points.reserve(frame->keypoint_num());
    for (size_t keypoint_index = 0;
         keypoint_index < frame->keypoint_num();
         ++keypoint_index) {
        const vector<2> pixel =
            apply_k(
                frame->get_keypoint(keypoint_index),
                frame->K);
        if (pixel.allFinite())
            current_points.emplace_back(pixel);
    }

    const auto extract_begin =
        std::chrono::steady_clock::now();
    try {
        LocalDescriptorSet current_descriptors =
            extractor->extract(
                *frame->image, current_points);
        const double extract_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                extract_begin)
                .count();

        if (!current_descriptors.valid() ||
            current_descriptors.type !=
                LocalDescriptorType::BINARY_U8 ||
            current_descriptors.dimension !=
                reference->local_descriptors.dimension ||
            current_descriptors.size() == 0) {
            std::fprintf(
                stderr,
                "[OrbAssociationShadow] current=%zu "
                "reference=%zu lag=%zu "
                "reject=current_descriptors_invalid "
                "current_keypoints=%zu current_points=%zu "
                "current_descriptors=%zu "
                "dimension=%zu extract_ms=%.3f\n",
                frame->id(), reference_frame_id, lag,
                frame->keypoint_num(),
                current_points.size(),
                current_descriptors.size(),
                current_descriptors.dimension,
                extract_ms);
            return;
        }

        const auto match_begin =
            std::chrono::steady_clock::now();
        const LocalDescriptorMatchResult matches =
            match_binary_descriptors_mutual_nn(
                reference->local_descriptors,
                current_descriptors);
        const double match_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                match_begin)
                .count();

        std::vector<vector<3>> pnp_points_world;
        std::vector<vector<2>> pnp_points_normalized;
        std::vector<vector<2>> pnp_points_pixel;
        size_t mapped_2d3d = 0;
        size_t pnp_hamming_filtered = 0;
        const size_t pnp_max_hamming =
            orb_pnp_max_hamming();
        pnp_points_world.reserve(
            matches.mutual_matches.size());
        pnp_points_normalized.reserve(
            matches.mutual_matches.size());
        pnp_points_pixel.reserve(
            matches.mutual_matches.size());

        const matrix<3> K_inv = frame->K.inverse();
        for (const auto &match :
             matches.mutual_matches) {
            if (match.reference_descriptor_index >=
                    reference->local_descriptors
                        .source_indices.size() ||
                match.current_descriptor_index >=
                    current_descriptors
                        .source_indices.size()) {
                continue;
            }

            const size_t observation_index =
                reference->local_descriptors
                    .source_indices[
                        match.reference_descriptor_index];
            const size_t current_point_index =
                current_descriptors
                    .source_indices[
                        match.current_descriptor_index];
            if (observation_index >=
                    reference->observations.size() ||
                current_point_index >=
                    current_points.size()) {
                continue;
            }

            const vector<3> &landmark_world =
                reference->observations[
                    observation_index].landmark_world;
            const vector<2> &pixel =
                current_points[current_point_index];
            const vector<3> normalized_h =
                K_inv *
                vector<3>{pixel.x(), pixel.y(), 1.0};
            if (!landmark_world.allFinite() ||
                !normalized_h.allFinite() ||
                std::abs(normalized_h.z()) <= 1.0e-12) {
                continue;
            }

            ++mapped_2d3d;
            if (match.distance > pnp_max_hamming) {
                ++pnp_hamming_filtered;
                continue;
            }

            pnp_points_world.emplace_back(landmark_world);
            pnp_points_normalized.emplace_back(
                normalized_h.hnormalized());
            pnp_points_pixel.emplace_back(pixel);
        }

        const HammingDistanceStats raw_stats =
            hamming_distance_stats(
                matches.nearest_neighbors);
        const HammingDistanceStats mutual_stats =
            hamming_distance_stats(
                matches.mutual_matches);

        std::fprintf(
            stderr,
            "[OrbAssociationShadow] "
            "current=%zu t=%.9f "
            "reference=%zu reference_t=%.9f lag=%zu "
            "reference_observations=%zu "
            "reference_descriptors=%zu "
            "current_keypoints=%zu current_points=%zu "
            "current_descriptors=%zu "
            "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
            "raw_hamming_min=%zu raw_hamming_p25=%zu "
            "raw_hamming_median=%zu raw_hamming_p75=%zu "
            "raw_hamming_max=%zu raw_hamming_mean=%.3f "
            "mutual_hamming_min=%zu mutual_hamming_p25=%zu "
            "mutual_hamming_median=%zu mutual_hamming_p75=%zu "
            "mutual_hamming_max=%zu mutual_hamming_mean=%.3f "
            "extract_ms=%.3f match_ms=%.3f "
            "threshold=none state_mutation=0\n",
            frame->id(), frame->image->t,
            reference_frame_id, reference->timestamp, lag,
            reference->observations.size(),
            reference->local_descriptors.size(),
            frame->keypoint_num(),
            current_points.size(),
            current_descriptors.size(),
            matches.nearest_neighbors.size(),
            matches.mutual_matches.size(),
            mapped_2d3d,
            raw_stats.min, raw_stats.p25,
            raw_stats.median, raw_stats.p75,
            raw_stats.max, raw_stats.mean,
            mutual_stats.min, mutual_stats.p25,
            mutual_stats.median, mutual_stats.p75,
            mutual_stats.max, mutual_stats.mean,
            extract_ms, match_ms);

        if (relocalization_v1_profile_enabled()) {
            constexpr size_t min_correspondences = 6;
            constexpr size_t min_verified_inliers = 8;
            constexpr double min_verified_inlier_ratio = 0.50;
            const size_t correspondence_count =
                pnp_points_world.size();

            std::fprintf(
                stderr,
                "[OrbPnPInputShadow] current=%zu reference=%zu "
                "mapped_2d3d=%zu hamming_filtered=%zu "
                "correspondences=%zu max_hamming=%zu\n",
                frame->id(), reference_frame_id,
                mapped_2d3d, pnp_hamming_filtered,
                correspondence_count, pnp_max_hamming);

            if (correspondence_count <
                min_correspondences) {
                std::fprintf(
                    stderr,
                    "[OrbPnPShadow] current=%zu t=%.9f "
                    "reference=%zu reference_t=%.9f lag=%zu "
                    "correspondences=%zu min_required=%zu "
                    "skip=insufficient_correspondences "
                    "max_hamming=%zu min_inliers=%zu "
                    "min_inlier_ratio=%.6f "
                    "geometrically_verified=0 "
                    "state_mutation=0\n",
                    frame->id(), frame->image->t,
                    reference_frame_id,
                    reference->timestamp, lag,
                    correspondence_count,
                    min_correspondences,
                    pnp_max_hamming,
                    min_verified_inliers,
                    min_verified_inlier_ratio);
            } else {
                const auto pnp_begin =
                    std::chrono::steady_clock::now();
                try {
                    std::vector<char> inlier_mask;
                    const double pnp_noise_sigma_norm =
                        1.0 / frame->K(0, 0);
                    const matrix<4> T_cw =
                        find_pnp_matrix(
                            pnp_points_world,
                            pnp_points_normalized,
                            inlier_mask,
                            pnp_noise_sigma_norm);
                    const double pnp_ms =
                        std::chrono::duration<
                            double, std::milli>(
                            std::chrono::steady_clock::now() -
                            pnp_begin)
                            .count();

                    if (inlier_mask.size() !=
                        correspondence_count) {
                        std::fprintf(
                            stderr,
                            "[OrbPnPShadow] current=%zu "
                            "reference=%zu lag=%zu "
                            "correspondences=%zu "
                            "reject=invalid_inlier_mask "
                            "mask_size=%zu pnp_ms=%.3f "
                            "max_hamming=%zu min_inliers=%zu "
                            "min_inlier_ratio=%.6f "
                            "geometrically_verified=0 "
                            "state_mutation=0\n",
                            frame->id(),
                            reference_frame_id, lag,
                            correspondence_count,
                            inlier_mask.size(), pnp_ms,
                            pnp_max_hamming,
                            min_verified_inliers,
                            min_verified_inlier_ratio);
                    } else {
                        size_t inlier_count = 0;
                        for (char inlier : inlier_mask) {
                            if (inlier)
                                ++inlier_count;
                        }

                        const double inlier_ratio =
                            static_cast<double>(
                                inlier_count) /
                            static_cast<double>(
                                correspondence_count);

                        if (inlier_count <
                                min_correspondences ||
                            !T_cw.allFinite()) {
                            std::fprintf(
                                stderr,
                                "[OrbPnPShadow] "
                                "current=%zu t=%.9f "
                                "reference=%zu "
                                "reference_t=%.9f lag=%zu "
                                "correspondences=%zu "
                                "inliers=%zu "
                                "inlier_ratio=%.6f "
                                "pose_finite=0 "
                                "pnp_threshold_parameter_px=1.000 "
                                "pnp_ms=%.3f "
                                "max_hamming=%zu min_inliers=%zu "
                                "min_inlier_ratio=%.6f "
                                "geometrically_verified=0 "
                                "state_mutation=0\n",
                                frame->id(),
                                frame->image->t,
                                reference_frame_id,
                                reference->timestamp,
                                lag,
                                correspondence_count,
                                inlier_count,
                                inlier_ratio,
                                pnp_ms,
                                pnp_max_hamming,
                                min_verified_inliers,
                                min_verified_inlier_ratio);
                        } else {
                            const matrix<3> R_cw =
                                T_cw.block<3, 3>(0, 0);
                            const vector<3> t_cw =
                                T_cw.block<3, 1>(0, 3);
                            const matrix<3> R_wc =
                                R_cw.transpose();

                            PoseState pnp_camera_pose;
                            pnp_camera_pose.q =
                                quaternion(R_wc);
                            pnp_camera_pose.q.normalize();
                            pnp_camera_pose.p =
                                -R_wc * t_cw;

                            double inlier_squared_error =
                                0.0;
                            size_t positive_depth_inliers =
                                0;
                            for (size_t i = 0;
                                 i < correspondence_count;
                                 ++i) {
                                if (!inlier_mask[i])
                                    continue;

                                const vector<3> point_camera =
                                    R_cw *
                                        pnp_points_world[i] +
                                    t_cw;
                                if (!point_camera.allFinite() ||
                                    point_camera.z() <= 1.0e-6) {
                                    continue;
                                }

                                const vector<2> projected =
                                    apply_k(
                                        point_camera,
                                        frame->K);
                                const double error =
                                    (projected -
                                     pnp_points_pixel[i])
                                        .norm();
                                inlier_squared_error +=
                                    error * error;
                                ++positive_depth_inliers;
                            }

                            const double inlier_rmse_px =
                                positive_depth_inliers == 0
                                    ? std::numeric_limits<
                                          double>::quiet_NaN()
                                    : std::sqrt(
                                          inlier_squared_error /
                                          static_cast<double>(
                                              positive_depth_inliers));

                            const PoseState vio_camera_pose =
                                frame->get_pose(
                                    frame->camera);
                            const double
                                translation_delta_m =
                                    (pnp_camera_pose.p -
                                     vio_camera_pose.p)
                                        .norm();
                            const double
                                rotation_delta_deg =
                                    camera_rotation_delta_deg(
                                        pnp_camera_pose,
                                        vio_camera_pose);

                            const bool pose_finite =
                                pnp_camera_pose.p
                                    .allFinite() &&
                                pnp_camera_pose.q.coeffs()
                                    .allFinite() &&
                                std::isfinite(
                                    translation_delta_m) &&
                                std::isfinite(
                                    rotation_delta_deg);
                            const bool geometrically_verified =
                                pose_finite &&
                                inlier_count >=
                                    min_verified_inliers &&
                                inlier_ratio >=
                                    min_verified_inlier_ratio;

                            std::fprintf(
                                stderr,
                                "[OrbPnPShadow] "
                                "current=%zu t=%.9f "
                                "reference=%zu "
                                "reference_t=%.9f lag=%zu "
                                "correspondences=%zu "
                                "inliers=%zu "
                                "inlier_ratio=%.6f "
                                "positive_depth_inliers=%zu "
                                "inlier_rmse_px=%.6f "
                                "pnp_p=%.9f,%.9f,%.9f "
                                "pnp_q=%.9f,%.9f,%.9f,%.9f "
                                "translation_delta_m=%.9f "
                                "rotation_delta_deg=%.9f "
                                "pose_finite=%d "
                                "pnp_threshold_parameter_px=1.000 "
                                "pnp_ms=%.3f "
                                "max_hamming=%zu min_inliers=%zu "
                                "min_inlier_ratio=%.6f "
                                "geometrically_verified=%d "
                                "state_mutation=0\n",
                                frame->id(),
                                frame->image->t,
                                reference_frame_id,
                                reference->timestamp,
                                lag,
                                correspondence_count,
                                inlier_count,
                                inlier_ratio,
                                positive_depth_inliers,
                                inlier_rmse_px,
                                pnp_camera_pose.p.x(),
                                pnp_camera_pose.p.y(),
                                pnp_camera_pose.p.z(),
                                pnp_camera_pose.q.x(),
                                pnp_camera_pose.q.y(),
                                pnp_camera_pose.q.z(),
                                pnp_camera_pose.q.w(),
                                translation_delta_m,
                                rotation_delta_deg,
                                pose_finite ? 1 : 0,
                                pnp_ms,
                                pnp_max_hamming,
                                min_verified_inliers,
                                min_verified_inlier_ratio,
                                geometrically_verified ? 1 : 0);
                        }
                    }
                } catch (const std::exception &e) {
                    std::fprintf(
                        stderr,
                        "[OrbPnPShadow] current=%zu "
                        "reference=%zu lag=%zu "
                        "correspondences=%zu "
                        "reject=pnp_exception error=%s "
                        "max_hamming=%zu min_inliers=%zu "
                        "min_inlier_ratio=%.6f "
                        "geometrically_verified=0 "
                        "state_mutation=0\n",
                        frame->id(),
                        reference_frame_id, lag,
                        correspondence_count,
                        e.what(),
                        pnp_max_hamming,
                        min_verified_inliers,
                        min_verified_inlier_ratio);
                }
            }
        }
    } catch (const std::exception &e) {
        std::fprintf(
            stderr,
            "[OrbAssociationShadow] current=%zu "
            "reference=%zu lag=%zu "
            "reject=extract_exception error=%s\n",
            frame->id(), reference_frame_id,
            lag, e.what());
    }
}

void SlidingWindowTracker::diagnose_retrieved_place_candidates(
    Frame *frame,
    const std::vector<PlaceCandidate> &candidates) {
    if (!relocalization_v1_profile_enabled() ||
        !relocalization_v1_profile_enabled() ||
        !frame || !frame->image || !detail ||
        candidates.empty()) {
        return;
    }

    LocalDescriptorExtractor *extractor =
        detail->local_descriptor_extractor();
    if (!extractor ||
        extractor->type() !=
            LocalDescriptorType::BINARY_U8) {
        std::fprintf(
            stderr,
            "[PlaceRetrievalVerificationShadow] current=%zu "
            "reject=binary_extractor_unavailable "
            "state_mutation=0\n",
            frame->id());
        return;
    }

    std::vector<vector<2>> current_points;
    current_points.reserve(frame->keypoint_num());
    for (size_t keypoint_index = 0;
         keypoint_index < frame->keypoint_num();
         ++keypoint_index) {
        const vector<2> pixel =
            apply_k(
                frame->get_keypoint(keypoint_index),
                frame->K);
        if (pixel.allFinite())
            current_points.emplace_back(pixel);
    }

    const auto extract_begin =
        std::chrono::steady_clock::now();
    LocalDescriptorSet current_descriptors;
    double extract_ms = 0.0;
    try {
        current_descriptors =
            extractor->extract(
                *frame->image, current_points);
        extract_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                extract_begin)
                .count();
    } catch (const std::exception &e) {
        std::fprintf(
            stderr,
            "[PlaceRetrievalVerificationShadow] current=%zu "
            "t=%.9f candidate_count=%zu "
            "reject=extract_exception error=%s "
            "state_mutation=0\n",
            frame->id(), frame->image->t,
            candidates.size(), e.what());
        return;
    }

    if (!current_descriptors.valid() ||
        current_descriptors.type !=
            LocalDescriptorType::BINARY_U8 ||
        current_descriptors.dimension !=
            extractor->dimension() ||
        current_descriptors.size() == 0) {
        std::fprintf(
            stderr,
            "[PlaceRetrievalVerificationShadow] current=%zu "
            "t=%.9f candidate_count=%zu "
            "reject=current_descriptors_invalid "
            "current_keypoints=%zu current_points=%zu "
            "current_descriptors=%zu dimension=%zu "
            "extract_ms=%.3f state_mutation=0\n",
            frame->id(), frame->image->t,
            candidates.size(),
            frame->keypoint_num(),
            current_points.size(),
            current_descriptors.size(),
            current_descriptors.dimension,
            extract_ms);
        return;
    }

    constexpr size_t min_correspondences = 6;
    constexpr size_t min_verified_inliers = 8;
    constexpr double min_verified_inlier_ratio = 0.50;
    const size_t pnp_max_hamming =
        orb_pnp_max_hamming();
    const matrix<3> K_inv = frame->K.inverse();

    std::vector<PlaceKey> verified_keys;
    verified_keys.reserve(candidates.size());
    size_t verified_rank_min =
        std::numeric_limits<size_t>::max();
    size_t verified_rank_max = 0;
    size_t verified_reference_frame_min =
        std::numeric_limits<size_t>::max();
    size_t verified_reference_frame_max = 0;
    size_t verified_max_frame_separation = 0;
    double verified_max_abs_timestamp_separation = 0.0;
    struct VerifiedCandidateTemporalDiagnostic {
        PlaceKey key = 0;
        size_t rank = 0;
        size_t reference_frame_id = 0;
        double reference_timestamp = 0.0;
        double distance = 0.0;
        size_t correspondence_count = 0;
        size_t inlier_count = 0;
        double inlier_ratio = 0.0;
        std::vector<vector<3>> inlier_landmarks_world;
        std::vector<vector<2>> inlier_observations_pixel;
        PoseState pnp_camera_pose;
        double translation_delta_m = 0.0;
        double rotation_delta_deg = 0.0;
        double abs_timestamp_separation = 0.0;
    };
    std::vector<VerifiedCandidateTemporalDiagnostic>
        verified_temporal_diagnostics;
    verified_temporal_diagnostics.reserve(candidates.size());

    for (size_t rank = 0;
         rank < candidates.size();
         ++rank) {
        const PlaceCandidate &candidate =
            candidates[rank];
        const PlaceKeyframe *historical =
            place_keyframes_.find(candidate.key);
        if (!historical) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] current=%zu "
                "t=%.9f rank=%zu key=%llu distance=%.9f "
                "reject=metadata_unavailable temporal_excluded=0 "
                "state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                candidate.distance);
            continue;
        }

        const size_t reference_frame_id =
            historical->frame_id;
        const size_t frame_separation =
            frame->id() >= reference_frame_id
                ? frame->id() - reference_frame_id
                : reference_frame_id - frame->id();
        const double timestamp_separation =
            frame->image->t -
            historical->timestamp;

        const ArchivedKeyframe *reference =
            keyframe_archive_.get(reference_frame_id);
        if (!reference) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] current=%zu "
                "t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "reject=reference_not_archived "
                "temporal_excluded=0 state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation);
            continue;
        }

        if (!reference->local_descriptors_complete ||
            !reference->local_descriptors.valid() ||
            reference->local_descriptors.type !=
                LocalDescriptorType::BINARY_U8 ||
            reference->local_descriptors.dimension !=
                current_descriptors.dimension) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] current=%zu "
                "t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "reject=reference_descriptors_unavailable "
                "temporal_excluded=0 state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation);
            continue;
        }

        if (reference->local_descriptors.size() == 0) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] current=%zu "
                "t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "skip=reference_empty temporal_excluded=0 "
                "geometrically_verified=0 state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation);
            continue;
        }

        const auto match_begin =
            std::chrono::steady_clock::now();
        const LocalDescriptorMatchResult matches =
            match_binary_descriptors_mutual_nn(
                reference->local_descriptors,
                current_descriptors);
        const double match_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                match_begin)
                .count();

        std::vector<vector<3>> pnp_points_world;
        std::vector<vector<2>> pnp_points_normalized;
        std::vector<vector<2>> pnp_points_pixel;
        size_t mapped_2d3d = 0;
        size_t pnp_hamming_filtered = 0;
        pnp_points_world.reserve(
            matches.mutual_matches.size());
        pnp_points_normalized.reserve(
            matches.mutual_matches.size());
        pnp_points_pixel.reserve(
            matches.mutual_matches.size());

        for (const auto &match :
             matches.mutual_matches) {
            if (match.reference_descriptor_index >=
                    reference->local_descriptors
                        .source_indices.size() ||
                match.current_descriptor_index >=
                    current_descriptors
                        .source_indices.size()) {
                continue;
            }

            const size_t observation_index =
                reference->local_descriptors
                    .source_indices[
                        match.reference_descriptor_index];
            const size_t current_point_index =
                current_descriptors
                    .source_indices[
                        match.current_descriptor_index];
            if (observation_index >=
                    reference->observations.size() ||
                current_point_index >=
                    current_points.size()) {
                continue;
            }

            const vector<3> &landmark_world =
                reference->observations[
                    observation_index].landmark_world;
            const vector<2> &pixel =
                current_points[current_point_index];
            const vector<3> normalized_h =
                K_inv *
                vector<3>{pixel.x(), pixel.y(), 1.0};
            if (!landmark_world.allFinite() ||
                !normalized_h.allFinite() ||
                std::abs(normalized_h.z()) <= 1.0e-12) {
                continue;
            }

            ++mapped_2d3d;
            if (match.distance > pnp_max_hamming) {
                ++pnp_hamming_filtered;
                continue;
            }

            pnp_points_world.emplace_back(landmark_world);
            pnp_points_normalized.emplace_back(
                normalized_h.hnormalized());
            pnp_points_pixel.emplace_back(pixel);
        }

        const HammingDistanceStats raw_stats =
            hamming_distance_stats(
                matches.nearest_neighbors);
        const HammingDistanceStats mutual_stats =
            hamming_distance_stats(
                matches.mutual_matches);
        const size_t correspondence_count =
            pnp_points_world.size();

        if (correspondence_count <
            min_correspondences) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] "
                "current=%zu t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "reference_observations=%zu "
                "reference_descriptors=%zu "
                "current_keypoints=%zu current_points=%zu "
                "current_descriptors=%zu "
                "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
                "hamming_filtered=%zu correspondences=%zu "
                "raw_hamming_min=%zu raw_hamming_median=%zu "
                "raw_hamming_max=%zu "
                "mutual_hamming_min=%zu "
                "mutual_hamming_median=%zu "
                "mutual_hamming_max=%zu "
                "extract_ms=%.3f match_ms=%.3f pnp_ms=0.000 "
                "max_hamming=%zu min_inliers=%zu "
                "min_inlier_ratio=%.6f "
                "skip=insufficient_correspondences "
                "temporal_excluded=0 geometrically_verified=0 "
                "state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation,
                reference->observations.size(),
                reference->local_descriptors.size(),
                frame->keypoint_num(),
                current_points.size(),
                current_descriptors.size(),
                matches.nearest_neighbors.size(),
                matches.mutual_matches.size(),
                mapped_2d3d,
                pnp_hamming_filtered,
                correspondence_count,
                raw_stats.min, raw_stats.median,
                raw_stats.max,
                mutual_stats.min, mutual_stats.median,
                mutual_stats.max,
                extract_ms, match_ms,
                pnp_max_hamming,
                min_verified_inliers,
                min_verified_inlier_ratio);
            continue;
        }

        const auto pnp_begin =
            std::chrono::steady_clock::now();
        try {
            std::vector<char> inlier_mask;
            const double pnp_noise_sigma_norm =
                1.0 / frame->K(0, 0);
            const matrix<4> T_cw =
                find_pnp_matrix(
                    pnp_points_world,
                    pnp_points_normalized,
                    inlier_mask,
                    pnp_noise_sigma_norm);
            const double pnp_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    pnp_begin)
                    .count();

            if (inlier_mask.size() !=
                correspondence_count) {
                std::fprintf(
                    stderr,
                    "[PlaceRetrievalVerificationShadow] "
                    "current=%zu t=%.9f rank=%zu key=%llu "
                    "candidate_frame_id=%zu candidate_t=%.9f "
                    "distance=%.9f frame_separation=%zu "
                    "timestamp_separation=%.9f "
                    "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
                    "hamming_filtered=%zu correspondences=%zu "
                    "mask_size=%zu extract_ms=%.3f "
                    "match_ms=%.3f pnp_ms=%.3f "
                    "max_hamming=%zu min_inliers=%zu "
                    "min_inlier_ratio=%.6f "
                    "reject=invalid_inlier_mask "
                    "temporal_excluded=0 geometrically_verified=0 "
                    "state_mutation=0\n",
                    frame->id(), frame->image->t,
                    rank,
                    static_cast<unsigned long long>(
                        candidate.key),
                    reference_frame_id,
                    historical->timestamp,
                    candidate.distance,
                    frame_separation,
                    timestamp_separation,
                    matches.nearest_neighbors.size(),
                    matches.mutual_matches.size(),
                    mapped_2d3d,
                    pnp_hamming_filtered,
                    correspondence_count,
                    inlier_mask.size(),
                    extract_ms, match_ms, pnp_ms,
                    pnp_max_hamming,
                    min_verified_inliers,
                    min_verified_inlier_ratio);
                continue;
            }

            size_t inlier_count = 0;
            for (char inlier : inlier_mask) {
                if (inlier)
                    ++inlier_count;
            }
            const double inlier_ratio =
                static_cast<double>(inlier_count) /
                static_cast<double>(
                    correspondence_count);

            if (inlier_count <
                    min_correspondences ||
                !T_cw.allFinite()) {
                std::fprintf(
                    stderr,
                    "[PlaceRetrievalVerificationShadow] "
                    "current=%zu t=%.9f rank=%zu key=%llu "
                    "candidate_frame_id=%zu candidate_t=%.9f "
                    "distance=%.9f frame_separation=%zu "
                    "timestamp_separation=%.9f "
                    "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
                    "hamming_filtered=%zu correspondences=%zu "
                    "inliers=%zu inlier_ratio=%.6f "
                    "pose_finite=0 extract_ms=%.3f "
                    "match_ms=%.3f pnp_ms=%.3f "
                    "pnp_threshold_parameter_px=1.000 "
                    "max_hamming=%zu min_inliers=%zu "
                    "min_inlier_ratio=%.6f "
                    "reject=invalid_pose "
                    "temporal_excluded=0 geometrically_verified=0 "
                    "state_mutation=0\n",
                    frame->id(), frame->image->t,
                    rank,
                    static_cast<unsigned long long>(
                        candidate.key),
                    reference_frame_id,
                    historical->timestamp,
                    candidate.distance,
                    frame_separation,
                    timestamp_separation,
                    matches.nearest_neighbors.size(),
                    matches.mutual_matches.size(),
                    mapped_2d3d,
                    pnp_hamming_filtered,
                    correspondence_count,
                    inlier_count,
                    inlier_ratio,
                    extract_ms, match_ms, pnp_ms,
                    pnp_max_hamming,
                    min_verified_inliers,
                    min_verified_inlier_ratio);
                continue;
            }

            const matrix<3> R_cw =
                T_cw.block<3, 3>(0, 0);
            const vector<3> t_cw =
                T_cw.block<3, 1>(0, 3);
            const matrix<3> R_wc =
                R_cw.transpose();

            PoseState pnp_camera_pose;
            pnp_camera_pose.q =
                quaternion(R_wc);
            pnp_camera_pose.q.normalize();
            pnp_camera_pose.p =
                -R_wc * t_cw;

            double inlier_squared_error = 0.0;
            size_t positive_depth_inliers = 0;
            std::vector<vector<3>> inlier_landmarks_world;
            std::vector<vector<2>> inlier_observations_pixel;
            inlier_landmarks_world.reserve(inlier_count);
            inlier_observations_pixel.reserve(inlier_count);
            for (size_t i = 0;
                 i < correspondence_count;
                 ++i) {
                if (!inlier_mask[i])
                    continue;

                inlier_landmarks_world.emplace_back(
                    pnp_points_world[i]);
                inlier_observations_pixel.emplace_back(
                    pnp_points_pixel[i]);

                const vector<3> point_camera =
                    R_cw * pnp_points_world[i] +
                    t_cw;
                if (!point_camera.allFinite() ||
                    point_camera.z() <= 1.0e-6) {
                    continue;
                }

                const vector<2> projected =
                    apply_k(
                        point_camera,
                        frame->K);
                const double error =
                    (projected -
                     pnp_points_pixel[i])
                        .norm();
                inlier_squared_error +=
                    error * error;
                ++positive_depth_inliers;
            }

            const double inlier_rmse_px =
                positive_depth_inliers == 0
                    ? std::numeric_limits<
                          double>::quiet_NaN()
                    : std::sqrt(
                          inlier_squared_error /
                          static_cast<double>(
                              positive_depth_inliers));

            const PoseState vio_camera_pose =
                frame->get_pose(
                    frame->camera);
            const double translation_delta_m =
                (pnp_camera_pose.p -
                 vio_camera_pose.p)
                    .norm();
            const double rotation_delta_deg =
                camera_rotation_delta_deg(
                    pnp_camera_pose,
                    vio_camera_pose);

            const bool pose_finite =
                pnp_camera_pose.p.allFinite() &&
                pnp_camera_pose.q.coeffs()
                    .allFinite() &&
                std::isfinite(
                    translation_delta_m) &&
                std::isfinite(
                    rotation_delta_deg);
            const bool geometrically_verified =
                pose_finite &&
                inlier_count >=
                    min_verified_inliers &&
                inlier_ratio >=
                    min_verified_inlier_ratio;

            if (geometrically_verified) {
                verified_keys.emplace_back(candidate.key);
                verified_rank_min =
                    std::min(verified_rank_min, rank);
                verified_rank_max =
                    std::max(verified_rank_max, rank);
                verified_reference_frame_min =
                    std::min(
                        verified_reference_frame_min,
                        reference_frame_id);
                verified_reference_frame_max =
                    std::max(
                        verified_reference_frame_max,
                        reference_frame_id);
                verified_max_frame_separation =
                    std::max(
                        verified_max_frame_separation,
                        frame_separation);
                verified_max_abs_timestamp_separation =
                    std::max(
                        verified_max_abs_timestamp_separation,
                        std::abs(timestamp_separation));
                verified_temporal_diagnostics.push_back(
                    VerifiedCandidateTemporalDiagnostic{
                        candidate.key,
                        rank,
                        reference_frame_id,
                        historical->timestamp,
                        candidate.distance,
                        correspondence_count,
                        inlier_count,
                        inlier_ratio,
                        std::move(inlier_landmarks_world),
                        std::move(inlier_observations_pixel),
                        pnp_camera_pose,
                        translation_delta_m,
                        rotation_delta_deg,
                        std::abs(timestamp_separation)});
            }

            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] "
                "current=%zu t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "reference_observations=%zu "
                "reference_descriptors=%zu "
                "current_keypoints=%zu current_points=%zu "
                "current_descriptors=%zu "
                "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
                "hamming_filtered=%zu correspondences=%zu "
                "inliers=%zu inlier_ratio=%.6f "
                "positive_depth_inliers=%zu "
                "inlier_rmse_px=%.6f "
                "pnp_p=%.9f,%.9f,%.9f "
                "pnp_q=%.9f,%.9f,%.9f,%.9f "
                "translation_delta_m=%.9f "
                "rotation_delta_deg=%.9f "
                "pose_finite=%d "
                "extract_ms=%.3f match_ms=%.3f pnp_ms=%.3f "
                "pnp_threshold_parameter_px=1.000 "
                "max_hamming=%zu min_inliers=%zu "
                "min_inlier_ratio=%.6f "
                "temporal_excluded=0 geometrically_verified=%d "
                "state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation,
                reference->observations.size(),
                reference->local_descriptors.size(),
                frame->keypoint_num(),
                current_points.size(),
                current_descriptors.size(),
                matches.nearest_neighbors.size(),
                matches.mutual_matches.size(),
                mapped_2d3d,
                pnp_hamming_filtered,
                correspondence_count,
                inlier_count,
                inlier_ratio,
                positive_depth_inliers,
                inlier_rmse_px,
                pnp_camera_pose.p.x(),
                pnp_camera_pose.p.y(),
                pnp_camera_pose.p.z(),
                pnp_camera_pose.q.x(),
                pnp_camera_pose.q.y(),
                pnp_camera_pose.q.z(),
                pnp_camera_pose.q.w(),
                translation_delta_m,
                rotation_delta_deg,
                pose_finite ? 1 : 0,
                extract_ms, match_ms, pnp_ms,
                pnp_max_hamming,
                min_verified_inliers,
                min_verified_inlier_ratio,
                geometrically_verified ? 1 : 0);
        } catch (const std::exception &e) {
            const double pnp_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    pnp_begin)
                    .count();
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] "
                "current=%zu t=%.9f rank=%zu key=%llu "
                "candidate_frame_id=%zu candidate_t=%.9f "
                "distance=%.9f frame_separation=%zu "
                "timestamp_separation=%.9f "
                "raw_nn=%zu mutual=%zu mapped_2d3d=%zu "
                "hamming_filtered=%zu correspondences=%zu "
                "extract_ms=%.3f match_ms=%.3f pnp_ms=%.3f "
                "max_hamming=%zu min_inliers=%zu "
                "min_inlier_ratio=%.6f "
                "reject=pnp_exception error=%s "
                "temporal_excluded=0 geometrically_verified=0 "
                "state_mutation=0\n",
                frame->id(), frame->image->t,
                rank,
                static_cast<unsigned long long>(
                    candidate.key),
                reference_frame_id,
                historical->timestamp,
                candidate.distance,
                frame_separation,
                timestamp_separation,
                matches.nearest_neighbors.size(),
                matches.mutual_matches.size(),
                mapped_2d3d,
                pnp_hamming_filtered,
                correspondence_count,
                extract_ms, match_ms, pnp_ms,
                pnp_max_hamming,
                min_verified_inliers,
                min_verified_inlier_ratio,
                e.what());
        }
    }

    std::vector<PlaceKey> temporal_dt2_keys;
    std::vector<PlaceKey> temporal_dt5_keys;
    std::vector<PlaceKey> temporal_dt10_keys;
    std::vector<double> temporal_dt2_reference_times;
    std::vector<double> temporal_dt5_reference_times;
    std::vector<double> temporal_dt10_reference_times;
    temporal_dt2_keys.reserve(
        verified_temporal_diagnostics.size());
    temporal_dt5_keys.reserve(
        verified_temporal_diagnostics.size());
    temporal_dt10_keys.reserve(
        verified_temporal_diagnostics.size());
    temporal_dt2_reference_times.reserve(
        verified_temporal_diagnostics.size());
    temporal_dt5_reference_times.reserve(
        verified_temporal_diagnostics.size());
    temporal_dt10_reference_times.reserve(
        verified_temporal_diagnostics.size());

    for (const auto &verified :
         verified_temporal_diagnostics) {
        if (verified.abs_timestamp_separation >= 2.0) {
            temporal_dt2_keys.emplace_back(verified.key);
            temporal_dt2_reference_times.emplace_back(
                verified.reference_timestamp);
        }
        if (verified.abs_timestamp_separation >= 5.0) {
            temporal_dt5_keys.emplace_back(verified.key);
            temporal_dt5_reference_times.emplace_back(
                verified.reference_timestamp);
        }
        if (verified.abs_timestamp_separation >= 10.0) {
            temporal_dt10_keys.emplace_back(verified.key);
            temporal_dt10_reference_times.emplace_back(
                verified.reference_timestamp);
        }
    }

    const auto count_overlap =
        [](const std::vector<PlaceKey> &current,
           const std::vector<PlaceKey> &previous) {
            size_t overlap = 0;
            for (PlaceKey key : current) {
                if (std::find(
                        previous.begin(),
                        previous.end(),
                        key) != previous.end()) {
                    ++overlap;
                }
            }
            return overlap;
        };

    const auto count_time_groups =
        [](std::vector<double> timestamps,
           double max_neighbor_gap_s) {
            if (timestamps.empty())
                return size_t{0};
            std::sort(timestamps.begin(), timestamps.end());
            size_t groups = 1;
            for (size_t i = 1; i < timestamps.size(); ++i) {
                if (timestamps[i] - timestamps[i - 1] >
                    max_neighbor_gap_s) {
                    ++groups;
                }
            }
            return groups;
        };

    const size_t temporal_dt2_overlap =
        count_overlap(
            temporal_dt2_keys,
            previous_temporal_dt2_place_keys_);
    const size_t temporal_dt5_overlap =
        count_overlap(
            temporal_dt5_keys,
            previous_temporal_dt5_place_keys_);
    const size_t temporal_dt10_overlap =
        count_overlap(
            temporal_dt10_keys,
            previous_temporal_dt10_place_keys_);

    std::fprintf(
        stderr,
        "[PlaceTemporalPolicyShadow] current=%zu t=%.9f "
        "verified_count=%zu "
        "eligible_dt2=%zu eligible_dt5=%zu eligible_dt10=%zu "
        "overlap_dt2=%zu overlap_dt5=%zu overlap_dt10=%zu "
        "groups_dt2_r0p5=%zu groups_dt2_r1=%zu groups_dt2_r2=%zu "
        "groups_dt5_r0p5=%zu groups_dt5_r1=%zu groups_dt5_r2=%zu "
        "groups_dt10_r0p5=%zu groups_dt10_r1=%zu groups_dt10_r2=%zu "
        "policy_applied=0 temporal_excluded=0 state_mutation=0\n",
        frame->id(), frame->image->t,
        verified_keys.size(),
        temporal_dt2_keys.size(),
        temporal_dt5_keys.size(),
        temporal_dt10_keys.size(),
        temporal_dt2_overlap,
        temporal_dt5_overlap,
        temporal_dt10_overlap,
        count_time_groups(temporal_dt2_reference_times, 0.5),
        count_time_groups(temporal_dt2_reference_times, 1.0),
        count_time_groups(temporal_dt2_reference_times, 2.0),
        count_time_groups(temporal_dt5_reference_times, 0.5),
        count_time_groups(temporal_dt5_reference_times, 1.0),
        count_time_groups(temporal_dt5_reference_times, 2.0),
        count_time_groups(temporal_dt10_reference_times, 0.5),
        count_time_groups(temporal_dt10_reference_times, 1.0),
        count_time_groups(temporal_dt10_reference_times, 2.0));

    struct NeighborhoodDiagnostic {
        double reference_t_min = 0.0;
        double reference_t_max = 0.0;
        size_t reference_frame_min =
            std::numeric_limits<size_t>::max();
        size_t reference_frame_max = 0;
        size_t member_count = 0;
        size_t best_rank =
            std::numeric_limits<size_t>::max();
        double best_distance =
            std::numeric_limits<double>::infinity();
        size_t max_inliers = 0;
        double max_inlier_ratio = 0.0;
        double min_abs_timestamp_separation =
            std::numeric_limits<double>::infinity();
        const VerifiedCandidateTemporalDiagnostic *
            representative = nullptr;
        size_t event_id = 0;
        size_t consecutive_age = 1;
        bool persisted_from_previous = false;
        size_t previous_current_frame_id = 0;
        size_t current_frame_gap = 0;
    };

    const auto better_representative =
        [](const VerifiedCandidateTemporalDiagnostic *candidate,
           const VerifiedCandidateTemporalDiagnostic *current) {
            if (!current)
                return true;
            if (candidate->inlier_count != current->inlier_count)
                return candidate->inlier_count > current->inlier_count;
            if (candidate->inlier_ratio != current->inlier_ratio)
                return candidate->inlier_ratio > current->inlier_ratio;
            if (candidate->rank != current->rank)
                return candidate->rank < current->rank;
            if (candidate->distance != current->distance)
                return candidate->distance < current->distance;
            return candidate->reference_frame_id <
                   current->reference_frame_id;
        };

    const auto build_neighborhoods =
        [&verified_temporal_diagnostics,
         &better_representative](
            double minimum_abs_separation_s) {
            std::vector<
                const VerifiedCandidateTemporalDiagnostic *>
                eligible;
            for (const auto &verified :
                 verified_temporal_diagnostics) {
                if (verified.abs_timestamp_separation >=
                    minimum_abs_separation_s) {
                    eligible.emplace_back(&verified);
                }
            }

            std::sort(
                eligible.begin(), eligible.end(),
                [](const auto *a, const auto *b) {
                    if (a->reference_timestamp !=
                        b->reference_timestamp) {
                        return a->reference_timestamp <
                               b->reference_timestamp;
                    }
                    return a->rank < b->rank;
                });

            std::vector<NeighborhoodDiagnostic> groups;
            for (const auto *verified : eligible) {
                if (groups.empty() ||
                    verified->reference_timestamp -
                            groups.back().reference_t_max >
                        2.0) {
                    NeighborhoodDiagnostic group;
                    group.reference_t_min =
                        verified->reference_timestamp;
                    group.reference_t_max =
                        verified->reference_timestamp;
                    group.reference_frame_min =
                        verified->reference_frame_id;
                    group.reference_frame_max =
                        verified->reference_frame_id;
                    groups.emplace_back(group);
                }

                NeighborhoodDiagnostic &group =
                    groups.back();
                group.reference_t_min =
                    std::min(
                        group.reference_t_min,
                        verified->reference_timestamp);
                group.reference_t_max =
                    std::max(
                        group.reference_t_max,
                        verified->reference_timestamp);
                group.reference_frame_min =
                    std::min(
                        group.reference_frame_min,
                        verified->reference_frame_id);
                group.reference_frame_max =
                    std::max(
                        group.reference_frame_max,
                        verified->reference_frame_id);
                ++group.member_count;
                group.best_rank =
                    std::min(
                        group.best_rank,
                        verified->rank);
                group.best_distance =
                    std::min(
                        group.best_distance,
                        verified->distance);
                group.max_inliers =
                    std::max(
                        group.max_inliers,
                        verified->inlier_count);
                group.max_inlier_ratio =
                    std::max(
                        group.max_inlier_ratio,
                        verified->inlier_ratio);
                group.min_abs_timestamp_separation =
                    std::min(
                        group.min_abs_timestamp_separation,
                        verified->abs_timestamp_separation);
                if (better_representative(
                        verified, group.representative)) {
                    group.representative = verified;
                }
            }
            return groups;
        };

    const auto assign_event_persistence =
        [this, frame](
            std::vector<NeighborhoodDiagnostic> &groups,
            const std::vector<
                PlaceNeighborhoodEventShadowState> &previous) {
            std::vector<char> previous_used(
                previous.size(), 0);
            std::vector<PlaceNeighborhoodEventShadowState>
                next;
            next.reserve(groups.size());

            const auto interval_gap =
                [](double a_min, double a_max,
                   double b_min, double b_max) {
                    if (a_max < b_min)
                        return b_min - a_max;
                    if (b_max < a_min)
                        return a_min - b_max;
                    return 0.0;
                };

            for (NeighborhoodDiagnostic &group : groups) {
                size_t best_previous = previous.size();
                double best_gap =
                    std::numeric_limits<double>::infinity();
                double best_center_delta =
                    std::numeric_limits<double>::infinity();
                const double group_center =
                    0.5 *
                    (group.reference_t_min +
                     group.reference_t_max);

                for (size_t i = 0;
                     i < previous.size();
                     ++i) {
                    if (previous_used[i])
                        continue;

                    const double gap =
                        interval_gap(
                            group.reference_t_min,
                            group.reference_t_max,
                            previous[i].reference_t_min,
                            previous[i].reference_t_max);
                    if (gap > 2.0)
                        continue;

                    const double previous_center =
                        0.5 *
                        (previous[i].reference_t_min +
                         previous[i].reference_t_max);
                    const double center_delta =
                        std::abs(
                            group_center -
                            previous_center);
                    if (gap < best_gap ||
                        (gap == best_gap &&
                         center_delta <
                             best_center_delta)) {
                        best_previous = i;
                        best_gap = gap;
                        best_center_delta = center_delta;
                    }
                }

                if (best_previous < previous.size()) {
                    const auto &matched =
                        previous[best_previous];
                    previous_used[best_previous] = 1;
                    group.event_id = matched.event_id;
                    group.consecutive_age =
                        matched.consecutive_age + 1;
                    group.persisted_from_previous = true;
                    group.previous_current_frame_id =
                        matched.last_current_frame_id;
                    group.current_frame_gap =
                        frame->id() >=
                                matched.last_current_frame_id
                            ? frame->id() -
                                  matched.last_current_frame_id
                            : matched.last_current_frame_id -
                                  frame->id();
                } else {
                    group.event_id =
                        next_place_neighborhood_event_id_++;
                }

                PlaceNeighborhoodEventShadowState state;
                state.event_id = group.event_id;
                state.reference_t_min =
                    group.reference_t_min;
                state.reference_t_max =
                    group.reference_t_max;
                state.consecutive_age =
                    group.consecutive_age;
                state.last_current_frame_id =
                    frame->id();
                next.emplace_back(state);
            }

            return next;
        };

    const auto log_neighborhoods =
        [this, frame](
            double threshold_s,
            const std::vector<NeighborhoodDiagnostic> &groups) {
            size_t persistent_groups = 0;
            size_t max_consecutive_age = 0;
            size_t total_members = 0;
            for (const auto &group : groups) {
                if (group.persisted_from_previous)
                    ++persistent_groups;
                max_consecutive_age =
                    std::max(
                        max_consecutive_age,
                        group.consecutive_age);
                total_members += group.member_count;
            }

            std::fprintf(
                stderr,
                "[PlaceNeighborhoodSetShadow] "
                "current=%zu t=%.9f threshold_s=%.1f "
                "group_count=%zu persistent_group_count=%zu "
                "max_consecutive_age=%zu member_count=%zu "
                "neighborhood_gap_s=2.000 "
                "policy_applied=0 temporal_excluded=0 "
                "state_mutation=0\n",
                frame->id(), frame->image->t,
                threshold_s,
                groups.size(),
                persistent_groups,
                max_consecutive_age,
                total_members);

            for (size_t i = 0;
                 i < groups.size();
                 ++i) {
                const auto &group = groups[i];
                std::fprintf(
                    stderr,
                    "[PlaceNeighborhoodEventShadow] "
                    "current=%zu t=%.9f threshold_s=%.1f "
                    "group_index=%zu event_id=%zu "
                    "member_count=%zu "
                    "reference_t_min=%.9f "
                    "reference_t_max=%.9f "
                    "reference_frame_min=%zu "
                    "reference_frame_max=%zu "
                    "best_rank=%zu best_distance=%.9f "
                    "max_inliers=%zu "
                    "max_inlier_ratio=%.6f "
                    "min_abs_timestamp_separation=%.9f "
                    "persisted_from_previous=%d "
                    "consecutive_age=%zu "
                    "previous_current=%zu "
                    "current_frame_gap=%zu "
                    "neighborhood_gap_s=2.000 "
                    "policy_applied=0 temporal_excluded=0 "
                    "state_mutation=0\n",
                    frame->id(), frame->image->t,
                    threshold_s,
                    i,
                    group.event_id,
                    group.member_count,
                    group.reference_t_min,
                    group.reference_t_max,
                    group.reference_frame_min,
                    group.reference_frame_max,
                    group.best_rank,
                    group.best_distance,
                    group.max_inliers,
                    group.max_inlier_ratio,
                    group.min_abs_timestamp_separation,
                    group.persisted_from_previous ? 1 : 0,
                    group.consecutive_age,
                    group.previous_current_frame_id,
                    group.current_frame_gap);
                if (threshold_s == 5.0 &&
                    group.consecutive_age == 3 &&
                    group.representative) {
                    const auto &representative =
                        *group.representative;
                    std::fprintf(
                        stderr,
                        "[PlaceConfirmedEventRepresentativeShadow] "
                        "current=%zu t=%.9f threshold_s=5.0 "
                        "event_id=%zu confirmation_age=%zu "
                        "member_count=%zu "
                        "reference_t_min=%.9f "
                        "reference_t_max=%.9f "
                        "reference_frame_min=%zu "
                        "reference_frame_max=%zu "
                        "representative_key=%llu "
                        "representative_frame=%zu "
                        "representative_t=%.9f "
                        "representative_rank=%zu "
                        "representative_distance=%.9f "
                        "correspondences=%zu "
                        "inliers=%zu inlier_ratio=%.6f "
                        "pnp_p=%.9f,%.9f,%.9f "
                        "pnp_q=%.9f,%.9f,%.9f,%.9f "
                        "translation_delta_m=%.9f "
                        "rotation_delta_deg=%.9f "
                        "representative_dt10_eligible=%d "
                        "selection=max_inliers_then_ratio_then_rank_"
                        "then_distance_then_frame "
                        "confirmed_shadow=1 acceptance_applied=0 "
                        "state_mutation=0\n",
                        frame->id(), frame->image->t,
                        group.event_id,
                        group.consecutive_age,
                        group.member_count,
                        group.reference_t_min,
                        group.reference_t_max,
                        group.reference_frame_min,
                        group.reference_frame_max,
                        static_cast<unsigned long long>(
                            representative.key),
                        representative.reference_frame_id,
                        representative.reference_timestamp,
                        representative.rank,
                        representative.distance,
                        representative.correspondence_count,
                        representative.inlier_count,
                        representative.inlier_ratio,
                        representative.pnp_camera_pose.p.x(),
                        representative.pnp_camera_pose.p.y(),
                        representative.pnp_camera_pose.p.z(),
                        representative.pnp_camera_pose.q.x(),
                        representative.pnp_camera_pose.q.y(),
                        representative.pnp_camera_pose.q.z(),
                        representative.pnp_camera_pose.q.w(),
                        representative.translation_delta_m,
                        representative.rotation_delta_deg,
                        representative.abs_timestamp_separation >= 10.0
                            ? 1
                            : 0);
                    const PoseState authoritative_camera_pose =
                        frame->get_pose(frame->camera);

                    std::unique_ptr<Frame> recovery_frame =
                        frame->clone();
                    recovery_frame->tag(FT_FIX_POSE) = false;

                    recovery_frame->pose.q =
                        representative.pnp_camera_pose.q *
                        recovery_frame->camera.q_cs.conjugate();
                    recovery_frame->pose.q.normalize();
                    recovery_frame->pose.p =
                        representative.pnp_camera_pose.p -
                        recovery_frame->pose.q *
                            recovery_frame->camera.p_cs;

                    const PoseState pnp_seed_camera_pose =
                        recovery_frame->get_pose(
                            recovery_frame->camera);

                    const auto reprojection_rmse_px =
                        [frame, &representative](
                            const PoseState &camera_pose) {
                            if (representative
                                    .inlier_landmarks_world.empty() ||
                                representative
                                        .inlier_landmarks_world.size() !=
                                    representative
                                        .inlier_observations_pixel.size()) {
                                return std::numeric_limits<
                                    double>::quiet_NaN();
                            }

                            double squared_error = 0.0;
                            size_t positive_depth_count = 0;
                            for (size_t j = 0;
                                 j < representative
                                         .inlier_landmarks_world.size();
                                 ++j) {
                                const vector<3> point_camera =
                                    camera_pose.q.conjugate() *
                                    (representative
                                         .inlier_landmarks_world[j] -
                                     camera_pose.p);
                                if (!point_camera.allFinite() ||
                                    point_camera.z() <= 1.0e-6) {
                                    continue;
                                }

                                const vector<2> projected =
                                    apply_k(
                                        point_camera,
                                        frame->K);
                                const double error =
                                    (projected -
                                     representative
                                         .inlier_observations_pixel[j])
                                        .norm();
                                squared_error += error * error;
                                ++positive_depth_count;
                            }

                            if (positive_depth_count == 0) {
                                return std::numeric_limits<
                                    double>::quiet_NaN();
                            }
                            return std::sqrt(
                                squared_error /
                                static_cast<double>(
                                    positive_depth_count));
                        };

                    const double pnp_seed_rmse_px =
                        reprojection_rmse_px(
                            pnp_seed_camera_pose);

                    auto recovery_solver = Solver::create();
                    recovery_solver->add_frame_states(
                        recovery_frame.get(), false);

                    size_t fixed_world_factor_count = 0;
                    for (size_t j = 0;
                         j < representative
                                 .inlier_landmarks_world.size();
                         ++j) {
                        recovery_solver
                            ->add_learned_world_reprojection(
                                recovery_frame.get(),
                                representative
                                    .inlier_landmarks_world[j],
                                representative
                                    .inlier_observations_pixel[j]);
                        ++fixed_world_factor_count;
                    }

                    const bool recovery_usable =
                        fixed_world_factor_count >= 6 &&
                        recovery_solver->solve();

                    const PoseState recovery_body_pose =
                        recovery_frame->pose;
                    const PoseState recovery_camera_pose =
                        recovery_frame->get_pose(
                            recovery_frame->camera);
                    const double recovery_rmse_px =
                        reprojection_rmse_px(
                            recovery_camera_pose);

                    const double seed_delta_authoritative_t =
                        (pnp_seed_camera_pose.p -
                         authoritative_camera_pose.p)
                            .norm();
                    const double seed_delta_authoritative_r =
                        camera_rotation_delta_deg(
                            authoritative_camera_pose,
                            pnp_seed_camera_pose);
                    const double recovery_delta_authoritative_t =
                        (recovery_camera_pose.p -
                         authoritative_camera_pose.p)
                            .norm();
                    const double recovery_delta_authoritative_r =
                        camera_rotation_delta_deg(
                            authoritative_camera_pose,
                            recovery_camera_pose);
                    const double recovery_delta_seed_t =
                        (recovery_camera_pose.p -
                         pnp_seed_camera_pose.p)
                            .norm();
                    const double recovery_delta_seed_r =
                        camera_rotation_delta_deg(
                            pnp_seed_camera_pose,
                            recovery_camera_pose);

                    const bool candidate_finite =
                        recovery_body_pose.p.allFinite() &&
                        recovery_body_pose.q.coeffs().allFinite() &&
                        recovery_camera_pose.p.allFinite() &&
                        recovery_camera_pose.q.coeffs().allFinite() &&
                        std::isfinite(recovery_rmse_px);

                    const bool temporal_eligible =
                        representative.abs_timestamp_separation >=
                        5.0;
                    const bool event_confirmed =
                        group.consecutive_age >= 3;
                    const bool representative_geometry_verified =
                        representative.inlier_count >=
                            min_verified_inliers &&
                        representative.inlier_ratio >=
                            min_verified_inlier_ratio;
                    const bool factor_count_sufficient =
                        fixed_world_factor_count >= 6;
                    const bool raw_rmse_improved =
                        std::isfinite(pnp_seed_rmse_px) &&
                        std::isfinite(recovery_rmse_px) &&
                        recovery_rmse_px <=
                            pnp_seed_rmse_px;

                    const bool acceptance_candidate =
                        temporal_eligible &&
                        event_confirmed &&
                        representative_geometry_verified &&
                        factor_count_sufficient &&
                        recovery_usable &&
                        candidate_finite;

                    const char *acceptance_reason =
                        !temporal_eligible
                            ? "temporal_ineligible"
                        : !event_confirmed
                            ? "event_unconfirmed"
                        : !representative_geometry_verified
                            ? "geometry_unverified"
                        : !factor_count_sufficient
                            ? "insufficient_factors"
                        : !recovery_usable
                            ? "solver_unusable"
                        : !candidate_finite
                            ? "candidate_nonfinite"
                            : "eligible";


                    // 0112a: export the exact body-frame relative
                    // place edge used by the shadow 4-DoF graph experiments.
                    // The reference endpoint is the archived optimized body
                    // pose; the current endpoint is the recovered body pose.
                    // This is diagnostic only and does not mutate estimator
                    // or place-recovery state.
                    const ArchivedKeyframe *graph_reference =
                        keyframe_archive_.get(
                            representative.reference_frame_id);
                    if (!graph_reference) {
                        std::fprintf(
                            stderr,
                            "[PlaceGraphEdgeShadow] "
                            "current=%zu t=%.9f event_id=%zu "
                            "reference_frame=%zu "
                            "reject=reference_not_archived "
                            "acceptance_candidate=%d "
                            "state_mutation=0\n",
                            frame->id(), frame->image->t,
                            group.event_id,
                            representative.reference_frame_id,
                            acceptance_candidate ? 1 : 0);
                    } else {
                        const PoseState &graph_reference_body_pose =
                            graph_reference->body_pose;
                        const vector<3> graph_relative_t =
                            graph_reference_body_pose.q.conjugate() *
                            (recovery_body_pose.p -
                             graph_reference_body_pose.p);
                        const matrix<3> graph_reference_R =
                            graph_reference_body_pose.q.matrix();
                        const matrix<3> graph_recovery_R =
                            recovery_body_pose.q.matrix();
                        const double graph_reference_yaw_rad =
                            std::atan2(
                                graph_reference_R(1, 0),
                                graph_reference_R(0, 0));
                        const double graph_recovery_yaw_rad =
                            std::atan2(
                                graph_recovery_R(1, 0),
                                graph_recovery_R(0, 0));
                        const double graph_relative_yaw_deg =
                            std::remainder(
                                (graph_recovery_yaw_rad -
                                 graph_reference_yaw_rad) *
                                    180.0 / M_PI,
                                360.0);
                        const bool graph_edge_finite =
                            graph_reference_body_pose.p.allFinite() &&
                            graph_reference_body_pose.q.coeffs()
                                .allFinite() &&
                            recovery_body_pose.p.allFinite() &&
                            recovery_body_pose.q.coeffs().allFinite() &&
                            graph_relative_t.allFinite() &&
                            std::isfinite(graph_relative_yaw_deg);

                        double graph_residual_gain = 1.0;
                        if (acceptance_candidate && graph_edge_finite) {
                            const auto conditioning =
                                pnp_directional_conditioning(
                                    recovery_body_pose,
                                    frame->camera,
                                    frame->K,
                                    graph_reference_body_pose,
                                    representative.inlier_landmarks_world,
                                    representative
                                        .inlier_observations_pixel);
                            if (conditioning.finite) {
                                graph_residual_gain =
                                    conditioning.graph_residual_gain;
                            }
                            std::fprintf(
                                stderr,
                                "[PlacePnPConditioningShadow] "
                                "current=%zu t=%.9f event_id=%zu "
                                "reference_frame=%zu inliers=%zu "
                                "observations=%zu finite=%d "
                                "reprojection_rmse_px=%.9f "
                                "information_condition=%.9g "
                                "sigma_t_ref_m=%.9g,%.9g,%.9g "
                                "sigma_r_world_deg=%.9g,%.9g,%.9g "
                                "pixel_sigma_hat_px=%.9g "
                                "graph_cov_condition=%.9g "
                                "graph_sigma_rms=%.9g "
                                "graph_residual_gain=%.9g "
                                "graph_cov4_unit="
                                "%.9g,%.9g,%.9g,%.9g,"
                                "%.9g,%.9g,%.9g,%.9g,"
                                "%.9g,%.9g,%.9g,%.9g,"
                                "%.9g,%.9g,%.9g,%.9g "
                                "graph_cov_eigenvalues="
                                "%.9g,%.9g,%.9g,%.9g "
                                "graph_weak_mode="
                                "%.9g,%.9g,%.9g,%.9g "
                                "graph_units=tx_m,ty_m,tz_m,yaw_deg_x0p1 "
                                "unit_pixel_sigma=1 state_mutation=0\n",
                                frame->id(), frame->image->t,
                                group.event_id,
                                representative.reference_frame_id,
                                representative.inlier_count,
                                conditioning.observation_count,
                                conditioning.finite ? 1 : 0,
                                conditioning.reprojection_rmse_px,
                                conditioning.information_condition,
                                conditioning.sigma_translation_ref_m.x(),
                                conditioning.sigma_translation_ref_m.y(),
                                conditioning.sigma_translation_ref_m.z(),
                                conditioning.sigma_rotation_world_deg.x(),
                                conditioning.sigma_rotation_world_deg.y(),
                                conditioning.sigma_rotation_world_deg.z(),
                                conditioning.estimated_pixel_sigma_px,
                                conditioning.graph_covariance_condition,
                                conditioning.graph_sigma_rms,
                                conditioning.graph_residual_gain,
                                conditioning.graph_covariance_unit(0, 0),
                                conditioning.graph_covariance_unit(0, 1),
                                conditioning.graph_covariance_unit(0, 2),
                                conditioning.graph_covariance_unit(0, 3),
                                conditioning.graph_covariance_unit(1, 0),
                                conditioning.graph_covariance_unit(1, 1),
                                conditioning.graph_covariance_unit(1, 2),
                                conditioning.graph_covariance_unit(1, 3),
                                conditioning.graph_covariance_unit(2, 0),
                                conditioning.graph_covariance_unit(2, 1),
                                conditioning.graph_covariance_unit(2, 2),
                                conditioning.graph_covariance_unit(2, 3),
                                conditioning.graph_covariance_unit(3, 0),
                                conditioning.graph_covariance_unit(3, 1),
                                conditioning.graph_covariance_unit(3, 2),
                                conditioning.graph_covariance_unit(3, 3),
                                conditioning.graph_covariance_eigenvalues(0),
                                conditioning.graph_covariance_eigenvalues(1),
                                conditioning.graph_covariance_eigenvalues(2),
                                conditioning.graph_covariance_eigenvalues(3),
                                conditioning.graph_weak_mode(0),
                                conditioning.graph_weak_mode(1),
                                conditioning.graph_weak_mode(2),
                                conditioning.graph_weak_mode(3));
                        }

                        std::fprintf(
                            stderr,
                            "[PlaceGraphEdgeShadow] "
                            "current=%zu t=%.9f event_id=%zu "
                            "reference_frame=%zu reference_t=%.9f "
                            "reference_body_p=%.9f,%.9f,%.9f "
                            "reference_body_q=%.9f,%.9f,%.9f,%.9f "
                            "recovered_body_p=%.9f,%.9f,%.9f "
                            "recovered_body_q=%.9f,%.9f,%.9f,%.9f "
                            "relative_t_ref=%.9f,%.9f,%.9f "
                            "relative_yaw_deg=%.9f "
                            "pnp_inliers=%zu pnp_ratio=%.6f "
                            "acceptance_candidate=%d edge_finite=%d "
                            "state_mutation=0\n",
                            frame->id(), frame->image->t,
                            group.event_id,
                            representative.reference_frame_id,
                            graph_reference->timestamp,
                            graph_reference_body_pose.p.x(),
                            graph_reference_body_pose.p.y(),
                            graph_reference_body_pose.p.z(),
                            graph_reference_body_pose.q.x(),
                            graph_reference_body_pose.q.y(),
                            graph_reference_body_pose.q.z(),
                            graph_reference_body_pose.q.w(),
                            recovery_body_pose.p.x(),
                            recovery_body_pose.p.y(),
                            recovery_body_pose.p.z(),
                            recovery_body_pose.q.x(),
                            recovery_body_pose.q.y(),
                            recovery_body_pose.q.z(),
                            recovery_body_pose.q.w(),
                            graph_relative_t.x(),
                            graph_relative_t.y(),
                            graph_relative_t.z(),
                            graph_relative_yaw_deg,
                            representative.inlier_count,
                            representative.inlier_ratio,
                            acceptance_candidate ? 1 : 0,
                            graph_edge_finite ? 1 : 0);
                        if (relocalization_v1_profile_enabled() &&
                            acceptance_candidate &&
                            graph_edge_finite) {
                            PlaceGraph4DoFEdge graph_edge;
                            graph_edge.event_id = group.event_id;
                            graph_edge.reference_frame_id =
                                representative.reference_frame_id;
                            graph_edge.current_frame_id = frame->id();
                            graph_edge.relative_translation =
                                graph_relative_t;
                            graph_edge.relative_yaw_deg =
                                graph_relative_yaw_deg;
                            graph_edge.residual_gain =
                                graph_residual_gain;

                            const auto existing_edge =
                                std::find_if(
                                    place_graph_4dof_edges_.begin(),
                                    place_graph_4dof_edges_.end(),
                                    [&graph_edge](
                                        const PlaceGraph4DoFEdge &edge) {
                                        return edge.event_id ==
                                               graph_edge.event_id;
                                    });
                            if (existing_edge ==
                                place_graph_4dof_edges_.end()) {
                                place_graph_4dof_edges_.emplace_back(
                                    graph_edge);
                            } else {
                                *existing_edge = graph_edge;
                            }

                            const PlaceGraph4DoFSolveResult
                                graph_result =
                                    solve_place_graph_4dof_shadow(
                                        keyframe_archive_,
                                        place_graph_4dof_edges_);
                            const PlaceGraph4DoFCorrection
                                *current_correction =
                                    graph_result.find_correction(
                                        frame->id());
                            // Output-only candidate. No VIO or graph state is
                            // modified; rejected/unusable solves do not replace
                            // the previous valid drift correction.
                            if (graph_result.usable && current_correction) {
                                latest_global_correction_ =
                                    *current_correction;
                                latest_global_correction_available_ = true;
                            }
                            const double graph_nan =
                                std::numeric_limits<double>::quiet_NaN();
                            const double current_yaw_correction_deg =
                                current_correction
                                    ? current_correction->yaw_deg
                                    : graph_nan;
                            const vector<3> current_translation =
                                current_correction
                                    ? current_correction->translation
                                    : vector<3>::Constant(graph_nan);
                            const double current_translation_norm =
                                current_correction
                                    ? current_translation.norm()
                                    : graph_nan;

                            std::fprintf(
                                stderr,
                                "[PlaceGraph4DoFSolveShadow] "
                                "current=%zu t=%.9f event_id=%zu "
                                "nodes=%zu odometry_edges=%zu "
                                "loop_edges=%zu iterations=%zu "
                                "initial_cost=%.12g final_cost=%.12g "
                                "usable=%d "
                                "current_correction_available=%d "
                                "current_yaw_correction_deg=%.9f "
                                "current_translation=%.9f,%.9f,%.9f "
                                "current_translation_norm=%.9f "
                                "state_mutation=0\n",
                                frame->id(), frame->image->t,
                                group.event_id,
                                graph_result.node_count,
                                graph_result.odometry_edge_count,
                                graph_result.loop_edge_count,
                                graph_result.iterations,
                                graph_result.initial_cost,
                                graph_result.final_cost,
                                graph_result.usable ? 1 : 0,
                                current_correction ? 1 : 0,
                                current_yaw_correction_deg,
                                current_translation.x(),
                                current_translation.y(),
                                current_translation.z(),
                                current_translation_norm);

                            for (const PlaceGraph4DoFEdge &edge :
                                 place_graph_4dof_edges_) {
                                const PlaceGraph4DoFCorrection
                                    *edge_correction =
                                        graph_result.find_correction(
                                            edge.current_frame_id);
                                const double edge_yaw_deg =
                                    edge_correction
                                        ? edge_correction->yaw_deg
                                        : graph_nan;
                                const vector<3> edge_translation =
                                    edge_correction
                                        ? edge_correction->translation
                                        : vector<3>::Constant(graph_nan);
                                const double edge_translation_norm =
                                    edge_correction
                                        ? edge_translation.norm()
                                        : graph_nan;
                                std::fprintf(
                                    stderr,
                                    "[PlaceGraph4DoFEventCorrectionShadow] "
                                    "solve_event_id=%zu edge_event_id=%zu "
                                    "frame_id=%zu correction_available=%d "
                                    "yaw_correction_deg=%.9f "
                                    "translation=%.9f,%.9f,%.9f "
                                    "translation_norm=%.9f "
                                    "state_mutation=0\n",
                                    group.event_id,
                                    edge.event_id,
                                    edge.current_frame_id,
                                    edge_correction ? 1 : 0,
                                    edge_yaw_deg,
                                    edge_translation.x(),
                                    edge_translation.y(),
                                    edge_translation.z(),
                                    edge_translation_norm);
                            }

                            if (graph_result.usable) {
                                for (const auto &node_correction :
                                     graph_result.corrections) {
                                    std::fprintf(
                                        stderr,
                                        "[PlaceGraph4DoFNodeCorrectionShadow] "
                                        "solve_event_id=%zu frame_id=%zu "
                                        "t=%.9f yaw_correction_deg=%.9f "
                                        "translation=%.9f,%.9f,%.9f "
                                        "translation_norm=%.9f "
                                        "state_mutation=0\n",
                                        group.event_id,
                                        node_correction.frame_id,
                                        node_correction.timestamp,
                                        node_correction.yaw_deg,
                                        node_correction.translation.x(),
                                        node_correction.translation.y(),
                                        node_correction.translation.z(),
                                        node_correction.translation.norm());
                                }
                            }
                        }
                    }

                    std::fprintf(
                        stderr,
                        "[PlaceRecoverySolveShadow] "
                        "current=%zu t=%.9f event_id=%zu "
                        "confirmation_age=%zu "
                        "representative_key=%llu "
                        "representative_frame=%zu "
                        "fixed_world_factors=%zu "
                        "pnp_inliers=%zu pnp_ratio=%.6f "
                        "solver_usable=%d candidate_finite=%d "
                        "pnp_seed_rmse_px=%.6f "
                        "recovery_rmse_px=%.6f "
                        "seed_delta_authoritative_t=%.9f "
                        "seed_delta_authoritative_r_deg=%.9f "
                        "recovery_delta_authoritative_t=%.9f "
                        "recovery_delta_authoritative_r_deg=%.9f "
                        "recovery_delta_seed_t=%.9f "
                        "recovery_delta_seed_r_deg=%.9f "
                        "candidate_body_p=%.9f,%.9f,%.9f "
                        "candidate_body_q=%.9f,%.9f,%.9f,%.9f "
                        "candidate_camera_p=%.9f,%.9f,%.9f "
                        "candidate_camera_q=%.9f,%.9f,%.9f,%.9f "
                        "fixed_world_cauchy=2.448 "
                        "motion_source=authoritative_unchanged "
                        "acceptance_applied=0 state_mutation=0\n",
                        frame->id(), frame->image->t,
                        group.event_id,
                        group.consecutive_age,
                        static_cast<unsigned long long>(
                            representative.key),
                        representative.reference_frame_id,
                        fixed_world_factor_count,
                        representative.inlier_count,
                        representative.inlier_ratio,
                        recovery_usable ? 1 : 0,
                        candidate_finite ? 1 : 0,
                        pnp_seed_rmse_px,
                        recovery_rmse_px,
                        seed_delta_authoritative_t,
                        seed_delta_authoritative_r,
                        recovery_delta_authoritative_t,
                        recovery_delta_authoritative_r,
                        recovery_delta_seed_t,
                        recovery_delta_seed_r,
                        recovery_body_pose.p.x(),
                        recovery_body_pose.p.y(),
                        recovery_body_pose.p.z(),
                        recovery_body_pose.q.x(),
                        recovery_body_pose.q.y(),
                        recovery_body_pose.q.z(),
                        recovery_body_pose.q.w(),
                        recovery_camera_pose.p.x(),
                        recovery_camera_pose.p.y(),
                        recovery_camera_pose.p.z(),
                        recovery_camera_pose.q.x(),
                        recovery_camera_pose.q.y(),
                        recovery_camera_pose.q.z(),
                        recovery_camera_pose.q.w());

                    std::fprintf(
                        stderr,
                        "[PlaceRecoveryAcceptanceShadow] "
                        "current=%zu t=%.9f event_id=%zu "
                        "confirmation_age=%zu "
                        "representative_key=%llu "
                        "representative_frame=%zu "
                        "temporal_eligible=%d "
                        "event_confirmed=%d "
                        "geometry_verified=%d "
                        "factor_count_sufficient=%d "
                        "solver_usable=%d candidate_finite=%d "
                        "raw_rmse_improved=%d "
                        "acceptance_candidate=%d "
                        "reason=%s "
                        "pnp_inliers=%zu pnp_ratio=%.6f "
                        "fixed_world_factors=%zu "
                        "pnp_seed_rmse_px=%.6f "
                        "recovery_rmse_px=%.6f "
                        "recovery_delta_authoritative_t=%.9f "
                        "recovery_delta_authoritative_r_deg=%.9f "
                        "recovery_delta_seed_t=%.9f "
                        "recovery_delta_seed_r_deg=%.9f "
                        "vio_disagreement_gate=0 "
                        "rmse_gate=0 "
                        "acceptance_applied=0 state_mutation=0\n",
                        frame->id(), frame->image->t,
                        group.event_id,
                        group.consecutive_age,
                        static_cast<unsigned long long>(
                            representative.key),
                        representative.reference_frame_id,
                        temporal_eligible ? 1 : 0,
                        event_confirmed ? 1 : 0,
                        representative_geometry_verified ? 1 : 0,
                        factor_count_sufficient ? 1 : 0,
                        recovery_usable ? 1 : 0,
                        candidate_finite ? 1 : 0,
                        raw_rmse_improved ? 1 : 0,
                        acceptance_candidate ? 1 : 0,
                        acceptance_reason,
                        representative.inlier_count,
                        representative.inlier_ratio,
                        fixed_world_factor_count,
                        pnp_seed_rmse_px,
                        recovery_rmse_px,
                        recovery_delta_authoritative_t,
                        recovery_delta_authoritative_r,
                        recovery_delta_seed_t,
                        recovery_delta_seed_r);

                }
            }
        };

    std::vector<NeighborhoodDiagnostic> dt5_neighborhoods =
        build_neighborhoods(5.0);
    std::vector<NeighborhoodDiagnostic> dt10_neighborhoods =
        build_neighborhoods(10.0);

    std::vector<PlaceNeighborhoodEventShadowState>
        next_dt5_neighborhood_events =
            assign_event_persistence(
                dt5_neighborhoods,
                previous_dt5_neighborhood_events_);
    std::vector<PlaceNeighborhoodEventShadowState>
        next_dt10_neighborhood_events =
            assign_event_persistence(
                dt10_neighborhoods,
                previous_dt10_neighborhood_events_);

    log_neighborhoods(5.0, dt5_neighborhoods);
    log_neighborhoods(10.0, dt10_neighborhoods);

    previous_dt5_neighborhood_events_ =
        std::move(next_dt5_neighborhood_events);
    previous_dt10_neighborhood_events_ =
        std::move(next_dt10_neighborhood_events);

    const size_t invalid_frame_id =
        static_cast<size_t>(-1);
    const bool previous_available =
        previous_verified_current_frame_id_ !=
        invalid_frame_id;
    size_t exact_overlap = 0;
    if (previous_available) {
        for (PlaceKey key : verified_keys) {
            if (std::find(
                    previous_verified_place_keys_.begin(),
                    previous_verified_place_keys_.end(),
                    key) !=
                previous_verified_place_keys_.end()) {
                ++exact_overlap;
            }
        }
    }

    if (verified_keys.empty()) {
        std::fprintf(
            stderr,
            "[PlaceVerifiedSetShadow] current=%zu t=%.9f "
            "candidate_count=%zu verified_count=0 multi_verified=0 "
            "previous_available=%d previous_current=%zu "
            "previous_verified_count=%zu exact_key_overlap=%zu "
            "state_mutation=0\n",
            frame->id(), frame->image->t,
            candidates.size(),
            previous_available ? 1 : 0,
            previous_available
                ? previous_verified_current_frame_id_
                : 0,
            previous_verified_place_keys_.size(),
            exact_overlap);
    } else {
        std::fprintf(
            stderr,
            "[PlaceVerifiedSetShadow] current=%zu t=%.9f "
            "candidate_count=%zu verified_count=%zu multi_verified=%d "
            "rank_min=%zu rank_max=%zu "
            "reference_frame_min=%zu reference_frame_max=%zu "
            "max_frame_separation=%zu "
            "max_abs_timestamp_separation=%.9f "
            "previous_available=%d previous_current=%zu "
            "previous_verified_count=%zu exact_key_overlap=%zu "
            "state_mutation=0\n",
            frame->id(), frame->image->t,
            candidates.size(), verified_keys.size(),
            verified_keys.size() > 1 ? 1 : 0,
            verified_rank_min, verified_rank_max,
            verified_reference_frame_min,
            verified_reference_frame_max,
            verified_max_frame_separation,
            verified_max_abs_timestamp_separation,
            previous_available ? 1 : 0,
            previous_available
                ? previous_verified_current_frame_id_
                : 0,
            previous_verified_place_keys_.size(),
            exact_overlap);
    }

    previous_temporal_dt2_place_keys_ =
        std::move(temporal_dt2_keys);
    previous_temporal_dt5_place_keys_ =
        std::move(temporal_dt5_keys);
    previous_temporal_dt10_place_keys_ =
        std::move(temporal_dt10_keys);
    previous_verified_current_frame_id_ = frame->id();
    previous_verified_place_keys_ = std::move(verified_keys);
}

void SlidingWindowTracker::
diagnose_pending_retrieved_place_candidates() {
    if (pending_place_retrieval_candidates_.empty())
        return;

    std::vector<PendingPlaceRetrieval> pending;
    pending.swap(pending_place_retrieval_candidates_);

    for (const PendingPlaceRetrieval &entry : pending) {
        Frame *frame = nullptr;
        for (size_t i = 0; i < map->frame_num(); ++i) {
            Frame *candidate_frame = map->get_frame(i);
            if (candidate_frame &&
                candidate_frame->id() == entry.frame_id) {
                frame = candidate_frame;
                break;
            }
        }

        if (!frame || !frame->image) {
            std::fprintf(
                stderr,
                "[PlaceRetrievalVerificationShadow] current=%zu "
                "reject=current_not_active "
                "state_mutation=0\n",
                entry.frame_id);
            continue;
        }

        diagnose_retrieved_place_candidates(
            frame, entry.candidates);

        const bool place_ready =
            place_keyframes_.find(
                static_cast<PlaceKey>(frame->id())) != nullptr;
        const ArchivedKeyframe *archived =
            keyframe_archive_.get(frame->id());
        const bool local_ready =
            !relocalization_v1_profile_enabled() ||
            (archived &&
             archived->local_descriptors_complete);
        if (place_ready && local_ready) {
            frame->image
                ->retain_place_recognition_source(false);
        }
    }
}

void SlidingWindowTracker::extract_place_descriptors() {
    if (!relocalization_v1_profile_enabled())
        return;

    if (!detail) {
        std::fprintf(stderr,
                     "[PlaceDescriptorShadow] reject=no_detail\n");
        return;
    }

    PlaceDescriptorExtractor *extractor =
        detail->place_descriptor_extractor();
    if (!extractor) {
        std::fprintf(stderr,
                     "[PlaceDescriptorShadow] reject=no_extractor\n");
        return;
    }

    const bool retrieval_enabled =
        relocalization_v1_profile_enabled();
    PlaceDatabase *database =
        retrieval_enabled ? detail->place_database() : nullptr;
    const size_t retrieval_top_k =
        retrieval_enabled ? place_retrieval_top_k() : 0;

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        if (!frame || !frame->tag(FT_KEYFRAME) || !frame->image)
            continue;

        const PlaceKey key =
            static_cast<PlaceKey>(frame->id());
        if (place_keyframes_.find(key))
            continue;

        const bool source_available_before =
            frame->image->has_place_recognition_source();
        if (!source_available_before) {
            std::fprintf(
                stderr,
                "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                "reject=source_unavailable source_available_before=0\n",
                frame->id(), frame->image->t);
            continue;
        }

        const auto begin = std::chrono::steady_clock::now();
        try {
            PlaceDescriptor descriptor =
                extractor->extract(*frame->image);

            bool finite = !descriptor.empty();
            double squared_norm = 0.0;
            for (float value : descriptor.values) {
                finite = finite && std::isfinite(value);
                const double value_double =
                    static_cast<double>(value);
                squared_norm += value_double * value_double;
            }
            const double norm = std::sqrt(squared_norm);
            const size_t dimension = descriptor.dimension();

            if (!finite ||
                dimension != extractor->dimension() ||
                !std::isfinite(norm) ||
                norm <= 1.0e-12) {
                std::fprintf(
                    stderr,
                    "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                    "reject=invalid_descriptor dimension=%zu "
                    "expected_dimension=%zu norm=%.9f finite=%d\n",
                    frame->id(), frame->image->t,
                    dimension, extractor->dimension(), norm,
                    finite ? 1 : 0);
                continue;
            }

            PlaceKeyframe place_keyframe;
            place_keyframe.key = key;
            place_keyframe.frame_id = frame->id();
            place_keyframe.timestamp = frame->image->t;
            place_keyframe.descriptor = std::move(descriptor);

            bool database_compatible = false;
            size_t database_size_before = 0;
            std::vector<PlaceCandidate> candidates;
            double query_ms = 0.0;

            if (retrieval_enabled) {
                if (!database) {
                    std::fprintf(
                        stderr,
                        "[PlaceRetrievalShadow] current=%zu t=%.9f "
                        "reject=no_database state_mutation=0\n",
                        frame->id(), frame->image->t);
                } else if (database->dimension() != dimension) {
                    std::fprintf(
                        stderr,
                        "[PlaceRetrievalShadow] current=%zu t=%.9f "
                        "reject=database_dimension_mismatch "
                        "descriptor_dimension=%zu database_dimension=%zu "
                        "state_mutation=0\n",
                        frame->id(), frame->image->t,
                        dimension, database->dimension());
                } else {
                    database_compatible = true;
                    database_size_before = database->size();

                    const auto query_begin =
                        std::chrono::steady_clock::now();
                    candidates = database->search(
                        place_keyframe.descriptor,
                        retrieval_top_k);
                    query_ms =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() -
                            query_begin)
                            .count();

                    for (size_t rank = 0;
                         rank < candidates.size();
                         ++rank) {
                        const PlaceCandidate &candidate =
                            candidates[rank];
                        const PlaceKeyframe *historical =
                            place_keyframes_.find(candidate.key);
                        if (!historical) {
                            if (relocalization_verbose_diagnostics())
                            std::fprintf(
                                stderr,
                                "[PlaceRetrievalShadow] current=%zu "
                                "t=%.9f rank=%zu key=%llu "
                                "distance=%.9f metadata_found=0 "
                                "temporal_excluded=0\n",
                                frame->id(), frame->image->t,
                                rank,
                                static_cast<unsigned long long>(
                                    candidate.key),
                                candidate.distance);
                            continue;
                        }

                        const size_t frame_separation =
                            frame->id() >= historical->frame_id
                                ? frame->id() -
                                      historical->frame_id
                                : historical->frame_id -
                                      frame->id();
                        const double timestamp_separation =
                            frame->image->t -
                            historical->timestamp;

                        if (relocalization_verbose_diagnostics())
                        std::fprintf(
                            stderr,
                            "[PlaceRetrievalShadow] current=%zu "
                            "t=%.9f rank=%zu key=%llu "
                            "candidate_frame_id=%zu "
                            "distance=%.9f candidate_t=%.9f "
                            "frame_separation=%zu "
                            "timestamp_separation=%.9f "
                            "metadata_found=1 temporal_excluded=0\n",
                            frame->id(), frame->image->t,
                            rank,
                            static_cast<unsigned long long>(
                                candidate.key),
                            historical->frame_id,
                            candidate.distance,
                            historical->timestamp,
                            frame_separation,
                            timestamp_separation);
                    }
                }
            }

            if (database_compatible) {
                const auto index_begin =
                    std::chrono::steady_clock::now();
                const bool index_success =
                    database->add(
                        key, place_keyframe.descriptor);
                const double index_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        index_begin)
                        .count();

                if (relocalization_verbose_diagnostics())
                std::fprintf(
                    stderr,
                    "[PlaceRetrievalShadow] current=%zu t=%.9f "
                    "database_size=%zu top_k=%zu "
                    "query_success=1 candidate_count=%zu "
                    "query_ms=%.3f index_success=%d "
                    "database_size_after=%zu index_ms=%.3f "
                    "state_mutation=0\n",
                    frame->id(), frame->image->t,
                    database_size_before, retrieval_top_k,
                    candidates.size(), query_ms,
                    index_success ? 1 : 0,
                    database->size(), index_ms);

                if (!index_success)
                    continue;
            }

            if (!place_keyframes_.add(place_keyframe)) {
                std::fprintf(
                    stderr,
                    "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                    "reject=duplicate_key\n",
                    frame->id(), frame->image->t);
                continue;
            }

            const bool retrieval_verification_pending =
                retrieval_enabled &&
                relocalization_v1_profile_enabled() &&
                !candidates.empty();
            if (retrieval_verification_pending) {
                PendingPlaceRetrieval pending;
                pending.frame_id = frame->id();
                pending.candidates = candidates;
                pending_place_retrieval_candidates_.emplace_back(
                    std::move(pending));
            }

            bool local_ready = true;
            if (relocalization_v1_profile_enabled()) {
                const ArchivedKeyframe *archived =
                    keyframe_archive_.get(frame->id());
                local_ready =
                    archived &&
                    archived->local_descriptors_complete;
            }
            if (local_ready &&
                !retrieval_verification_pending) {
                frame->image
                    ->retain_place_recognition_source(false);
            }

            const bool source_available_after =
                frame->image->has_place_recognition_source();
            const double extract_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin)
                    .count();

            if (relocalization_verbose_diagnostics())
            std::fprintf(
                stderr,
                "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                "dimension=%zu norm=%.9f finite=1 "
                "source_available_before=1 source_available_after=%d "
                "extract_ms=%.3f descriptor_count=%zu\n",
                frame->id(), frame->image->t, dimension, norm,
                source_available_after ? 1 : 0,
                extract_ms, place_keyframes_.size());
        } catch (const std::exception &e) {
            std::fprintf(
                stderr,
                "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                "reject=extract_exception error=%s\n",
                frame->id(), frame->image->t, e.what());
        }
    }
}

bool SlidingWindowTracker::track() {

    if (config->parsac_flag()) {
        if (judge_track_status()) {
            update_track_status();
        }
    }

    localize_newframe();

    if (manage_keyframe()) {
        track_landmark();
        refine_window();
        // Stage owned pixels BEFORE place recognition can free its source.
        // This is fast CPU copying only; no Torch/CUDA calls on the tracker.
        if (semantic_session_id_ != 0 && detail &&
            detail->semantic_keyframe_sink()) {
            for (size_t i = 0; i < map->frame_num(); ++i) {
                Frame *kf = map->get_frame(i);
                if (!kf || !kf->tag(FT_KEYFRAME) || !kf->image ||
                    keyframe_archive_.contains(kf->id()) ||
                    staged_semantic_images_.count(kf->id()))
                    continue;
                SemanticPixelBuffer pixels;
                if (kf->image->copy_semantic_source(pixels) && pixels.valid()) {
                    if (detail->semantic_keyframe_sink()->wants_metric_depth()) {
                        SemanticMetricDepthBuffer depth;
                        if (kf->image->copy_semantic_metric_depth(depth) &&
                            depth.valid() && depth.width == pixels.width &&
                            depth.height == pixels.height)
                            staged_semantic_depths_.emplace(kf->id(),
                                                            std::move(depth));
                    }
                    staged_semantic_images_.emplace(kf->id(),
                                                    std::move(pixels));
                }
            }
        }
        extract_place_descriptors();
        archive_optimized_keyframes();
        staged_semantic_images_.clear();
        staged_semantic_depths_.clear();
        diagnose_pending_retrieved_place_candidates();
        slide_window();
    } else {
        refine_subwindow();
    }


    inspect_debug(sliding_window_landmarks, landmarks) {
        std::vector<Landmark> points;
        points.reserve(map->track_num());
        for (size_t i = 0; i < map->track_num(); ++i) {
            if (Track *track = map->get_track(i)) {
                if (track->tag(TT_VALID)) {
                    Landmark point;
                    point.p = track->get_landmark_point();
                    point.triangulated = track->tag(TT_TRIANGULATED);
                    points.push_back(point);
                }
            }
        }
        landmarks = std::move(points);
    }

    return true;
}

void SlidingWindowTracker::localize_newframe() {
    Frame *frame_i = map->get_frame(map->frame_num() - 2);
    if (!frame_i->subframes.empty()) {
        frame_i = frame_i->subframes.back().get();
    }
    Frame *frame_j = map->get_frame(map->frame_num() - 1);

    // Ordinary XRSLAM localization remains authoritative.
    auto solver = Solver::create();
    solver->add_frame_states(frame_j);

    if (config->has_imu()) {
        solver->put_factor(Solver::create_preintegration_prior_factor(
            frame_i, frame_j, frame_j->preintegration));
    }

    for (size_t k = 0; k < frame_j->keypoint_num(); ++k) {
        if (Track *track = frame_j->get_track(k)) {
            if (track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC)) {
                solver->put_factor(
                    Solver::create_reprojection_prior_factor(frame_j, track));
            }
        }
    }

    solver->solve();
}

bool SlidingWindowTracker::manage_keyframe() {
    Frame *keyframe_i = map->get_frame(map->frame_num() - 2);
    Frame *newframe_j = map->get_frame(map->frame_num() - 1);

    if (!keyframe_i->subframes.empty()) {
        if (keyframe_i->subframes.back()->tag(FT_NO_TRANSLATION)) {
            if (newframe_j->tag(FT_NO_TRANSLATION)) {
                // [T]...........<-[R]
                //  +-[R]-[R]-[R]
                // ==>
                // [T]
                //  +-[R-R]-[R]-[R]
            } else {
                // [T]...........<-[T]
                //  +-[R]-[R]-[R]
                // ==>
                // [T]........[R]-[T]
                //  +-[R]-[R]
                keyframe_i->subframes.back()->tag(FT_KEYFRAME) = true;
                map->attach_frame(std::move(keyframe_i->subframes.back()),
                                  map->frame_num() - 1);
                keyframe_i->subframes.pop_back();
                newframe_j->tag(FT_KEYFRAME) = true;
                return true;
            }
        } else {
            if (newframe_j->tag(FT_NO_TRANSLATION)) {
                // [T]...........<-[R]
                //  +-[T]-[T]-[T]
                // ==>
                // [T]........[T]
                //  +-[T]-[T]  +-[R]
                std::unique_ptr<Frame> frame_lifted =
                    std::move(keyframe_i->subframes.back());
                keyframe_i->subframes.pop_back();
                frame_lifted->tag(FT_KEYFRAME) = true;
                frame_lifted->subframes.emplace_back(
                    map->detach_frame(map->frame_num() - 1));
                map->attach_frame(std::move(frame_lifted));
                return true;
            } else {
                if (keyframe_i->subframes.size() >=
                    config->sliding_window_subframe_size()) {
                    // [T]...........<-[T]
                    //  +-[T]-[T]-[T]
                    // ==>
                    // [T]............[T]
                    //  +-[T]-[T]-[T]
                    newframe_j->tag(FT_KEYFRAME) = true;
                    return true;
                }
            }
        }
    }
    size_t mapped_landmark_count = 0;
    for (size_t k = 0; k < newframe_j->keypoint_num(); ++k) {
        if (Track *track = newframe_j->get_track(k)) {
            if (track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC)) {
                mapped_landmark_count++;
            }
        }
    }

    bool is_keyframe = mapped_landmark_count <
                       config->sliding_window_force_keyframe_landmarks();

#if defined(XRSLAM_IOS)
    is_keyframe = is_keyframe || !newframe_j->tag(FT_NO_TRANSLATION);
#endif

    if (is_keyframe) {
        newframe_j->tag(FT_KEYFRAME) = true;
        return true;
    } else {
        keyframe_i->subframes.emplace_back(
            map->detach_frame(map->frame_num() - 1));
        return false;
    }
}

void SlidingWindowTracker::track_landmark() {
    Frame *newframe_j = map->get_frame(map->frame_num() - 1);

    for (size_t k = 0; k < newframe_j->keypoint_num(); ++k) {
        if (Track *track = newframe_j->get_track(k)) {
            if (!track->tag(TT_TRIANGULATED)) {
                if (auto p = track->triangulate()) {
                    track->set_landmark_point(p.value());
                    track->tag(TT_TRIANGULATED) = true;
                    track->tag(TT_VALID) = true;
                    track->tag(TT_STATIC) = true;
                } else {
                    // outlier
                    track->landmark.inv_depth = -1.0;
                    track->tag(TT_TRIANGULATED) = false;
                    track->tag(TT_VALID) = false;
                }
            }
        }
    }
}

void SlidingWindowTracker::refine_window() {
    Frame *keyframe_i = map->get_frame(map->frame_num() - 2);
    Frame *keyframe_j = map->get_frame(map->frame_num() - 1);

    auto solver = Solver::create();
    if (!map->marginalization_factor) {
        map->marginalization_factor =
            Solver::create_marginalization_factor(map.get());
    }
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        solver->add_frame_states(frame);
    }
    std::unordered_set<Track *> visited_tracks;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Track *track = frame->get_track(j);
            if (!track)
                continue;
            if (visited_tracks.count(track) > 0)
                continue;
            visited_tracks.insert(track);
            if (!track->tag(TT_VALID))
                continue;
            if (!track->tag(TT_STATIC))
                continue;
            if (!track->first_frame()->tag(FT_KEYFRAME))
                continue;
            solver->add_track_states(track);
        }
    }

    solver->add_factor(map->marginalization_factor.get());

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Track *track = frame->get_track(j);
            if (!track)
                continue;
            if (!track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC))
                continue;
            if (!track->first_frame()->tag(FT_KEYFRAME))
                continue;
            if (frame == track->first_frame())
                continue;
            solver->add_factor(frame->reprojection_error_factors[j].get());
        }
    }

    if (config->has_imu()) {
        for (size_t j = 1; j < map->frame_num(); ++j) {
            Frame *frame_i = map->get_frame(j - 1);
            Frame *frame_j = map->get_frame(j);

            frame_j->keyframe_preintegration = frame_j->preintegration;
            if (!frame_i->subframes.empty()) {
                std::vector<ImuData> imu_data;
                for (size_t k = 0; k < frame_i->subframes.size(); ++k) {
                    auto &sub_imu_data = frame_i->subframes[k]->preintegration.data;
                    imu_data.insert(imu_data.end(), sub_imu_data.begin(),
                                    sub_imu_data.end());
                }
                frame_j->keyframe_preintegration.data.insert(
                    frame_j->keyframe_preintegration.data.begin(),
                    imu_data.begin(), imu_data.end());
            }

            if (frame_j->keyframe_preintegration.integrate(
                    frame_j->image->t, frame_i->motion.bg, frame_i->motion.ba,
                    true, true)) {
                solver->put_factor(Solver::create_preintegration_error_factor(
                    frame_i, frame_j, frame_j->keyframe_preintegration));
            }
        }
    }

    solver->solve();

    for (size_t k = 0; k < map->track_num(); ++k) {
        Track *track = map->get_track(k);
        if (track->tag(TT_TRIANGULATED)) {
            bool is_valid = true;
            auto x = track->get_landmark_point();
            double rpe = 0.0;
            double rpe_count = 0.0;
            for (const auto &[frame, keypoint_index] : track->keypoint_map()) {
                if (!frame->tag(FT_KEYFRAME))
                    continue;
                PoseState pose = frame->get_pose(frame->camera);
                vector<3> y = pose.q.conjugate() * (x - pose.p);
                if (y.z() <= 1.0e-3 || y.z() > 50) { // todo
                    is_valid = false;
                    break;
                }
                rpe += (apply_k(y, frame->K) -
                        apply_k(frame->get_keypoint(keypoint_index), frame->K))
                           .norm();
                rpe_count += 1.0;
            }
            is_valid = is_valid && (rpe / std::max(rpe_count, 1.0) < 3.0);
            track->tag(TT_VALID) = is_valid;
        } else {
            track->landmark.inv_depth = -1.0;
        }
    }

    for (size_t k = 0; k < map->track_num(); ++k) {
        Track *track = map->get_track(k);
        if (!track->tag(TT_VALID))
            track->tag(TT_TRASH) = true;
    }
}

void SlidingWindowTracker::slide_window() {
    while (map->frame_num() > config->sliding_window_size()) {
        Frame *frame = map->get_frame(0);
        for (size_t i = 0; i < frame->subframes.size(); ++i) {
            map->untrack_frame(frame->subframes[i].get());
        }
        map->marginalize_frame(0);
    }
}

void SlidingWindowTracker::refine_subwindow() {
    Frame *frame = map->get_frame(map->frame_num() - 1);
    if (frame->subframes.empty())
        return;
    if (frame->subframes[0]->tag(FT_NO_TRANSLATION)) {
        if (frame->subframes.size() >= 9) {
            for (size_t i = frame->subframes.size() / 3; i > 0; --i) {
                Frame *tgt_frame = frame->subframes[i * 3 - 1].get();
                std::vector<ImuData> imu_data;
                for (size_t j = i * 3 - 1; j > (i - 1) * 3; --j) {
                    Frame *src_frame = frame->subframes[j - 1].get();
                    imu_data.insert(imu_data.begin(),
                                    src_frame->preintegration.data.begin(),
                                    src_frame->preintegration.data.end());
                    map->untrack_frame(src_frame);
                    frame->subframes.erase(frame->subframes.begin() + (j - 1));
                }
                tgt_frame->preintegration.data.insert(
                    tgt_frame->preintegration.data.begin(), imu_data.begin(),
                    imu_data.end());
            }
        }

        auto solver = Solver::create();
        frame->tag(FT_FIX_POSE) = true;
        frame->tag(FT_FIX_MOTION) = true;

        solver->add_frame_states(frame);
        for (size_t i = 0; i < frame->subframes.size(); ++i) {
            Frame *subframe = frame->subframes[i].get();
            solver->add_frame_states(subframe);
            Frame *prev_frame =
                (i == 0 ? frame : frame->subframes[i - 1].get());
            if (config->has_imu()) {
                subframe->preintegration.integrate(
                    subframe->image->t, prev_frame->motion.bg,
                    prev_frame->motion.ba, true, true);
                solver->put_factor(Solver::create_preintegration_error_factor(
                    prev_frame, subframe, subframe->preintegration));
            }
        }

        Frame *last_subframe = frame->subframes.back().get();
        for (size_t k = 0; k < last_subframe->keypoint_num(); ++k) {
            if (Track *track = last_subframe->get_track(k)) {
                if (track->tag(TT_VALID)) {
                    if (track->tag(TT_TRIANGULATED)) {
                        if (track->tag(TT_STATIC))
                            solver->put_factor(
                                Solver::create_reprojection_prior_factor(
                                    last_subframe, track));
                    } else {
                        solver->put_factor(Solver::create_rotation_prior_factor(
                            last_subframe, track));
                    }
                }
            }
        }

        solver->solve();
        frame->tag(FT_FIX_POSE) = false;
        frame->tag(FT_FIX_MOTION) = false;
    } else {
        auto solver = Solver::create();
        frame->tag(FT_FIX_POSE) = true;
        frame->tag(FT_FIX_MOTION) = true;
        solver->add_frame_states(frame);
        for (size_t i = 0; i < frame->subframes.size(); ++i) {
            Frame *subframe = frame->subframes[i].get();
            solver->add_frame_states(subframe);
            Frame *prev_frame =
                (i == 0 ? frame : frame->subframes[i - 1].get());
            if (config->has_imu()) {
                subframe->preintegration.integrate(
                    subframe->image->t, prev_frame->motion.bg,
                    prev_frame->motion.ba, true, true);
                solver->put_factor(Solver::create_preintegration_error_factor(
                    prev_frame, subframe, subframe->preintegration));
            }
            for (size_t k = 0; k < subframe->keypoint_num(); ++k) {
                if (Track *track = subframe->get_track(k)) {
                    if (track->all_tagged(TT_VALID, TT_TRIANGULATED,
                                          TT_STATIC)) {
                        if (track->first_frame()->tag(FT_KEYFRAME)) {
                            solver->put_factor(
                                Solver::create_reprojection_prior_factor(
                                    subframe, track));
                        } else if (track->first_frame()->id() > frame->id()) {
                            solver->add_factor(
                                subframe->reprojection_error_factors[k].get());
                        }
                    }
                }
            }
        }
        solver->solve();
        frame->tag(FT_FIX_POSE) = false;
        frame->tag(FT_FIX_MOTION) = false;
    }
} // namespace xrslam

std::tuple<double, PoseState, MotionState>
SlidingWindowTracker::get_latest_state() const {
    const Frame *frame = map->get_frame(map->frame_num() - 1);
    if (!frame->subframes.empty()) {
        frame = frame->subframes.back().get();
    }
    return {frame->image->t, frame->pose, frame->motion};
}

matrix<3> compute_essential_matrix(matrix<3> &R, vector<3> &t) {
    matrix<3> t_ = matrix<3>::Zero();

    t_(0, 1) = -t(2);
    t_(0, 2) = t(1);
    t_(1, 0) = t(2);
    t_(1, 2) = -t(0);
    t_(2, 0) = -t(1);
    t_(2, 1) = t(0);

    matrix<3> E = t_ * R;
    return E;
}

double compute_epipolar_dist(matrix<3> F, vector<2> &pt1, vector<2> &pt2) {
    vector<3> l = F * pt1.homogeneous();
    double dist =
        std::abs(pt2.homogeneous().transpose() * l) / l.segment<2>(0).norm();
    return dist;
}

bool SlidingWindowTracker::check_frames_rpe(Track *track, const vector<3> &x) {
    std::vector<matrix<3, 4>> Ps;
    std::vector<vector<3>> ps;

    bool is_valid = true;
    double rpe = 0.0;
    double rpe_count = 0.0;
    for (const auto &[frame, keypoint_index] : track->keypoint_map()) {
        if (!frame->tag(FT_KEYFRAME))
            continue;
        PoseState pose = frame->get_pose(frame->camera);
        vector<3> y = pose.q.conjugate() * (x - pose.p);
        if (y.z() <= 1.0e-3 || y.z() > 50) { // todo
            is_valid = false;
            break;
        }
        rpe += (apply_k(y, frame->K) -
                apply_k(frame->get_keypoint(keypoint_index), frame->K))
                   .norm();
        rpe_count += 1.0;
    }
    is_valid = is_valid && (rpe / std::max(rpe_count, 1.0) < 3.0);

    return is_valid;
}

bool SlidingWindowTracker::filter_parsac_2d2d(
    Frame *frame_i, Frame *frame_j, std::vector<char> &mask,
    std::vector<size_t> &pts_to_index) {

    std::vector<vector<2>> pts1, pts2;

    for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
        if (Track *track = frame_i->get_track(ki)) {
            if (size_t kj = track->get_keypoint_index(frame_j)) {
                if (kj != nil()) {
                    pts1.push_back(frame_i->get_keypoint(ki).hnormalized());
                    pts2.push_back(frame_j->get_keypoint(kj).hnormalized());
                    pts_to_index.push_back(kj);
                }
            }
        }
    }

    if (pts1.size() < 10)
        return false;

    matrix<3> E =
        find_essential_matrix_parsac(pts1, pts2, mask, m_th / frame_i->K(0, 0));

    return true;
}

void SlidingWindowTracker::predict_RT(Frame *frame_i, Frame *frame_j,
                                      matrix<3> &R, vector<3> &t) {

    auto camera = frame_i->camera;
    auto imu = frame_i->imu;

    matrix<4> Pwc = matrix<4>::Identity();
    matrix<4> PwI = matrix<4>::Identity();
    matrix<4> Pwi = matrix<4>::Identity();
    matrix<4> Pwj = matrix<4>::Identity();

    Pwc.block<3, 3>(0, 0) = camera.q_cs.toRotationMatrix();
    Pwc.block<3, 1>(0, 3) = camera.p_cs;
    PwI.block<3, 3>(0, 0) = imu.q_cs.toRotationMatrix();
    PwI.block<3, 1>(0, 3) = imu.p_cs;
    Pwi.block<3, 3>(0, 0) = frame_i->pose.q.toRotationMatrix();
    Pwi.block<3, 1>(0, 3) = frame_i->pose.p;
    Pwj.block<3, 3>(0, 0) = frame_j->pose.q.toRotationMatrix();
    Pwj.block<3, 1>(0, 3) = frame_j->pose.p;

    matrix<4> Pji = Pwj.inverse() * Pwi;

    matrix<4> P = (Pwc.inverse() * PwI * Pji * PwI.inverse() * Pwc);

    R = P.block<3, 3>(0, 0);
    t = P.block(0, 3, 3, 1);
}

bool SlidingWindowTracker::judge_track_status() {

    Frame *curr_frame = map->get_frame(map->frame_num() - 1);
    Frame *keyframe = map->get_frame(map->frame_num() - 2);
    Frame *last_frame = keyframe;
    if (!keyframe->subframes.empty()) {
        last_frame = keyframe->subframes.back().get();
    }

    curr_frame->preintegration.integrate(curr_frame->image->t,
                                         last_frame->motion.bg,
                                         last_frame->motion.ba, true, true);
    curr_frame->preintegration.predict(last_frame, curr_frame);

    m_P2D.clear();
    m_P3D.clear();
    m_lens.clear();
    m_indices_map = std::vector<int>(curr_frame->keypoint_num(), -1);

    for (size_t k = 0; k < curr_frame->keypoint_num(); ++k) {
        if (Track *track = curr_frame->get_track(k)) {
            if (track->all_tagged(TT_VALID, TT_TRIANGULATED)) {
                const vector<3> &bearing = curr_frame->get_keypoint(k);
                const vector<3> &landmark = track->get_landmark_point();
                m_P2D.push_back(bearing.hnormalized());
                m_P3D.push_back(landmark);
                m_lens.push_back(std::max(track->m_life, size_t(0)));
                m_indices_map[k] = m_P3D.size() - 1;
            }
        }
    }

    if (m_P2D.size() < 20)
        return false;

    const PoseState &pose = curr_frame->get_pose(curr_frame->camera);

    std::vector<char> mask;
    matrix<3> Rcw = pose.q.inverse().toRotationMatrix();
    vector<3> tcw = pose.q.inverse() * pose.p * (-1.0);
    matrix<4> T_IMU =
        find_pnp_matrix_parsac_imu(m_P3D, m_P2D, m_lens, Rcw, tcw, 0.20, 1.0,
                                   mask, 1.0 / curr_frame->K(0, 0));

    matrix<3> R;
    vector<3> t;
    predict_RT(keyframe, curr_frame, R, t);

    // check rpe
    {
        std::vector<vector<2>> P2D_inliers, P2D_outliers;
        std::vector<vector<3>> P3D_inliers, P3D_outliers;

        for (int i = 0; i < m_P2D.size(); ++i) {
            if (mask[i]) {
                P2D_inliers.push_back(m_P2D[i]);
                P3D_inliers.push_back(m_P3D[i]);
            } else {
                P2D_outliers.push_back(m_P2D[i]);
                P3D_outliers.push_back(m_P3D[i]);
            }
        }

        std::vector<double> inlier_errs, outlier_errs;
        double inlier_errs_sum = 0, outlier_errs_sum = 0;
        for (int i = 0; i < P2D_inliers.size(); i++) {
            vector<3> p = pose.q.conjugate() * (P3D_inliers[i] - pose.p);
            double proj_err =
                (apply_k(p, curr_frame->K) -
                 apply_k(P2D_inliers[i].homogeneous(), curr_frame->K))
                    .norm();
            inlier_errs.push_back(proj_err);
            inlier_errs_sum += proj_err;
        }

        for (int i = 0; i < P2D_outliers.size(); i++) {
            vector<3> p = pose.q.conjugate() * (P3D_outliers[i] - pose.p);
            double proj_err =
                (apply_k(p, curr_frame->K) -
                 apply_k(P2D_outliers[i].homogeneous(), curr_frame->K))
                    .norm();
            outlier_errs.push_back(proj_err);
            outlier_errs_sum += proj_err;
        }
    }

    matrix<3> E = compute_essential_matrix(R, t);
    matrix<3> F =
        keyframe->K.transpose().inverse() * E * curr_frame->K.inverse();

    std::vector<vector<2>> inlier_set1, inlier_set2;
    std::vector<vector<2>> outlier_set1, outlier_set2;
    for (size_t i = 0; i < curr_frame->keypoint_num(); ++i) {
        if (m_indices_map[i] != -1) {
            if (size_t j =
                    curr_frame->get_track(i)->get_keypoint_index(keyframe);
                j != nil()) {
                if (mask[m_indices_map[i]]) {
                    inlier_set1.push_back(
                        apply_k(keyframe->get_keypoint(j), keyframe->K));
                    inlier_set2.push_back(
                        apply_k(curr_frame->get_keypoint(i), curr_frame->K));
                } else {
                    outlier_set1.push_back(
                        apply_k(keyframe->get_keypoint(j), keyframe->K));
                    outlier_set2.push_back(
                        apply_k(curr_frame->get_keypoint(i), curr_frame->K));
                }
            }
        }
    }

    std::vector<double> inliers_dist, outliers_dist;

    for (int i = 0; i < inlier_set1.size(); i++) {
        vector<2> &p1 = inlier_set1[i];
        vector<2> &p2 = inlier_set2[i];
        double err = compute_epipolar_dist(F, p1, p2) +
                     compute_epipolar_dist(F.transpose(), p2, p1);
        inliers_dist.push_back(err);
    }

    for (int i = 0; i < outlier_set1.size(); i++) {
        vector<2> &p1 = outlier_set1[i];
        vector<2> &p2 = outlier_set2[i];
        double err = compute_epipolar_dist(F, p1, p2) +
                     compute_epipolar_dist(F.transpose(), p2, p1);
        outliers_dist.push_back(err);
    }

    size_t min_num = 20;
    if (inliers_dist.size() < min_num || outliers_dist.size() < min_num)
        return false;

    std::sort(inliers_dist.begin(), inliers_dist.end());
    std::sort(outliers_dist.begin(), outliers_dist.end());

    double th1 = inliers_dist[size_t(inliers_dist.size() * 0.5)];
    double th2 = outliers_dist[size_t(outliers_dist.size() * 0.5)];

    if (th2 < th1 * 2) // mean there is ambiguity
        return false;

    m_th = (th1 + th2) / 2;

    for (size_t k = 0; k < curr_frame->keypoint_num(); ++k) {
        if (Track *track = curr_frame->get_track(k)) {
            // track->tag(TT_STATIC) = true;
            if (m_indices_map[k] != -1) {
                if (mask[m_indices_map[k]]) {
                    curr_frame->get_track(k)->tag(TT_OUTLIER) = false;
                    curr_frame->get_track(k)->tag(TT_STATIC) = true;
                } else {
                    curr_frame->get_track(k)->tag(TT_OUTLIER) = true;
                    curr_frame->get_track(k)->tag(TT_STATIC) = false;
                }
            }
        }
    }

    return true;
}

void SlidingWindowTracker::update_track_status() {

    Frame *curr_frame = map->get_frame(map->frame_num() - 1);
    size_t frame_id = feature_tracking_map->frame_index_by_id(curr_frame->id());

    if (frame_id == nil())
        return;

    Frame *old_frame = feature_tracking_map->get_frame(frame_id);

    std::vector<size_t> outlier_cnts(curr_frame->keypoint_num(), 0);
    std::vector<size_t> matches_cnts(curr_frame->keypoint_num(), 0);
    size_t start_idx = std::min(
        map->frame_num() - 1,
        std::max(map->frame_num() - 1 - config->parsac_keyframe_check_size(),
                 size_t(0)));
    for (size_t i = start_idx; i < map->frame_num() - 1; i++) {
        std::vector<char> mask;
        std::vector<size_t> pts_to_index;
        if (filter_parsac_2d2d(map->get_frame(i), curr_frame, mask,
                               pts_to_index)) {
            for (size_t j = 0; j < mask.size(); j++) {
                if (!mask[j]) {
                    outlier_cnts[pts_to_index[j]] += 1;
                }
                matches_cnts[pts_to_index[j]] += 1;
            }
        }
    }

    for (size_t i = 0; i < curr_frame->keypoint_num(); i++) {
        if (Track *curr_track = curr_frame->get_track(i)) {
            if (size_t j = curr_track->get_keypoint_index(old_frame)) {
                if (j != nil()) {
                    Track *old_track = old_frame->get_track(j);
                    size_t outlier_th = map->frame_num() / 2;
                    if (outlier_cnts[i] > outlier_th / 2 &&
                        outlier_cnts[i] > 0.8 * matches_cnts[i]) {
                        curr_track->tag(TT_STATIC) = false;
                    }
                    if (!old_track->tag(TT_STATIC) ||
                        !curr_track->tag(TT_STATIC)) {
                        curr_track->tag(TT_STATIC) = false;
                        old_track->tag(TT_STATIC) = false;
                    }
                }
            }
        }
    }
}

} // namespace xrslam
