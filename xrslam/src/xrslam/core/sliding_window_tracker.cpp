#include <xrslam/common.h>
#include <xrslam/core/detail.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/core/keyframe_archive.h>
#include <xrslam/core/frontend_worker.h>
#include <xrslam/core/recovery_learned_map.h>
#include <xrslam/core/recovery_pose_cache.h>
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
#include <cstdlib>
#include <exception>
#include <limits>

namespace xrslam {

namespace {

bool learned_recovery_shadow_enabled() {
    const char *value = std::getenv(
        "XRSLAM_LEARNED_RECOVERY_SHADOW");
    return value && std::string(value) == "1";
}

bool place_retrieval_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_PLACE_RETRIEVAL_SHADOW");
    return value && std::string(value) == "1";
}

bool place_recovery_commit_enabled() {
    const char *value =
        std::getenv("XRSLAM_PLACE_RECOVERY_COMMIT");
    return value && std::string(value) == "1";
}

bool place_recovery_factor_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_PLACE_RECOVERY_FACTOR_SHADOW");
    return value && std::string(value) == "1";
}

bool place_descriptor_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_PLACE_DESCRIPTOR_SHADOW");
    return (value && std::string(value) == "1") ||
           place_retrieval_shadow_enabled();
}

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

bool orb_pnp_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_ORB_PNP_SHADOW");
    return value && std::string(value) == "1";
}

bool orb_association_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_ORB_ASSOCIATION_SHADOW");
    return (value && std::string(value) == "1") ||
           orb_pnp_shadow_enabled();
}

bool local_descriptor_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_LOCAL_DESCRIPTOR_SHADOW");
    return (value && std::string(value) == "1") ||
           orb_association_shadow_enabled();
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

bool keyframe_archive_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_KEYFRAME_ARCHIVE_SHADOW");
    return (value && std::string(value) == "1") ||
           local_descriptor_shadow_enabled();
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

double recovery_observation_rmse_px(
    const Frame *frame,
    const RecoveryLearnedObservationSet &observations) {
    if (!frame || observations.landmarks_world.empty() ||
        observations.landmarks_world.size() !=
            observations.observations_pixel.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PoseState camera_pose = frame->get_pose(frame->camera);
    double squared_error = 0.0;
    size_t count = 0;
    for (size_t i = 0;
         i < observations.landmarks_world.size(); ++i) {
        const vector<3> point_camera =
            camera_pose.q.conjugate() *
            (observations.landmarks_world[i] - camera_pose.p);
        if (!point_camera.allFinite() ||
            point_camera.z() <= 1.0e-6) {
            continue;
        }
        const vector<2> projected =
            apply_k(point_camera, frame->K);
        const double error =
            (projected - observations.observations_pixel[i]).norm();
        squared_error += error * error;
        ++count;
    }

    if (count == 0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(
        squared_error / static_cast<double>(count));
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
    if (!keyframe_archive_shadow_enabled())
        return;

    LocalDescriptorExtractor *local_extractor = nullptr;
    if (local_descriptor_shadow_enabled()) {
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
                            !place_descriptor_shadow_enabled() ||
                            place_keyframes_.find(
                                static_cast<PlaceKey>(
                                    frame->id()));
                        if (place_ready &&
                            !orb_association_shadow_enabled()) {
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

        if (orb_association_shadow_enabled()) {
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
                    !place_descriptor_shadow_enabled() ||
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
    if (!orb_association_shadow_enabled() ||
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

        if (orb_pnp_shadow_enabled()) {
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
    if (!place_retrieval_shadow_enabled() ||
        !orb_pnp_shadow_enabled() ||
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
                    const PoseState authoritative_body_pose =
                        frame->pose;
                    const MotionState authoritative_motion =
                        frame->motion;
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

                    if (acceptance_candidate) {
                        const PoseState live_body_pose_after_shadow =
                            frame->pose;
                        const MotionState live_motion_after_shadow =
                            frame->motion;

                        const double commit_delta_t =
                            (recovery_body_pose.p -
                             authoritative_body_pose.p)
                                .norm();
                        const double commit_delta_r =
                            camera_rotation_delta_deg(
                                authoritative_body_pose,
                                recovery_body_pose);

                        const double live_pose_delta_t =
                            (live_body_pose_after_shadow.p -
                             authoritative_body_pose.p)
                                .norm();
                        const double live_q_coeff_delta =
                            (live_body_pose_after_shadow.q.coeffs() -
                             authoritative_body_pose.q.coeffs())
                                .norm();
                        const double live_pose_delta_r =
                            camera_rotation_delta_deg(
                                authoritative_body_pose,
                                live_body_pose_after_shadow);
                        const double live_v_delta =
                            (live_motion_after_shadow.v -
                             authoritative_motion.v)
                                .norm();
                        const double live_bg_delta =
                            (live_motion_after_shadow.bg -
                             authoritative_motion.bg)
                                .norm();
                        const double live_ba_delta =
                            (live_motion_after_shadow.ba -
                             authoritative_motion.ba)
                                .norm();

                        const bool live_state_unchanged =
                            live_pose_delta_t == 0.0 &&
                            live_q_coeff_delta == 0.0 &&
                            live_v_delta == 0.0 &&
                            live_bg_delta == 0.0 &&
                            live_ba_delta == 0.0;

                        const bool commit_enabled =
                            place_recovery_commit_enabled();
                        const bool target_is_latest =
                            map->frame_num() > 0 &&
                            map->get_frame(map->frame_num() - 1) ==
                                frame;
                        const bool target_has_subframes =
                            !frame->subframes.empty();
                        const bool commit_target_ready =
                            target_is_latest &&
                            !target_has_subframes;
                        const bool commit_preconditions_met =
                            commit_target_ready &&
                            live_state_unchanged;

                        std::fprintf(
                            stderr,
                            "[PlaceRecoveryCommitDryRun] "
                            "current=%zu t=%.9f event_id=%zu "
                            "acceptance_candidate=1 "
                            "commit_scope=pose_only "
                            "commit_enabled=%d "
                            "target_is_latest=%d "
                            "target_has_subframes=%d "
                            "commit_target_ready=%d "
                            "commit_preconditions_met=%d "
                            "authoritative_body_p=%.9f,%.9f,%.9f "
                            "authoritative_body_q=%.9f,%.9f,%.9f,%.9f "
                            "proposed_body_p=%.9f,%.9f,%.9f "
                            "proposed_body_q=%.9f,%.9f,%.9f,%.9f "
                            "commit_delta_t=%.9f "
                            "commit_delta_r_deg=%.9f "
                            "preserved_v=%.9f,%.9f,%.9f "
                            "preserved_bg=%.9f,%.9f,%.9f "
                            "preserved_ba=%.9f,%.9f,%.9f "
                            "live_pose_delta_t=%.9f "
                            "live_q_coeff_delta=%.9f "
                            "live_pose_delta_r_deg=%.9f "
                            "live_v_delta=%.9f "
                            "live_bg_delta=%.9f "
                            "live_ba_delta=%.9f "
                            "live_state_unchanged=%d "
                            "would_commit_pose=1 "
                            "would_preserve_v=1 "
                            "would_preserve_bg=1 "
                            "would_preserve_ba=1 "
                            "estimator_reset=0 loop_constraint=0 "
                            "commit_applied=0 state_mutation=0\n",
                            frame->id(), frame->image->t,
                            group.event_id,
                            commit_enabled ? 1 : 0,
                            target_is_latest ? 1 : 0,
                            target_has_subframes ? 1 : 0,
                            commit_target_ready ? 1 : 0,
                            commit_preconditions_met ? 1 : 0,
                            authoritative_body_pose.p.x(),
                            authoritative_body_pose.p.y(),
                            authoritative_body_pose.p.z(),
                            authoritative_body_pose.q.x(),
                            authoritative_body_pose.q.y(),
                            authoritative_body_pose.q.z(),
                            authoritative_body_pose.q.w(),
                            recovery_body_pose.p.x(),
                            recovery_body_pose.p.y(),
                            recovery_body_pose.p.z(),
                            recovery_body_pose.q.x(),
                            recovery_body_pose.q.y(),
                            recovery_body_pose.q.z(),
                            recovery_body_pose.q.w(),
                            commit_delta_t,
                            commit_delta_r,
                            authoritative_motion.v.x(),
                            authoritative_motion.v.y(),
                            authoritative_motion.v.z(),
                            authoritative_motion.bg.x(),
                            authoritative_motion.bg.y(),
                            authoritative_motion.bg.z(),
                            authoritative_motion.ba.x(),
                            authoritative_motion.ba.y(),
                            authoritative_motion.ba.z(),
                            live_pose_delta_t,
                            live_q_coeff_delta,
                            live_pose_delta_r,
                            live_v_delta,
                            live_bg_delta,
                            live_ba_delta,
                            live_state_unchanged ? 1 : 0);

                        // 0110k coherent-commit dry run. Construct the exact
                        // gravity-preserving world-gauge transform that a
                        // future controlled commit would apply to every
                        // active top-level frame and attached subframe.
                        // Nothing below mutates live estimator state.
                        const quaternion full_correction_q =
                            (recovery_body_pose.q *
                             authoritative_body_pose.q
                                 .conjugate())
                                .normalized();
                        const matrix<3> full_correction_R =
                            full_correction_q.matrix();
                        const double coherent_yaw_rad =
                            std::atan2(
                                full_correction_R(1, 0),
                                full_correction_R(0, 0));
                        quaternion coherent_yaw_q;
                        coherent_yaw_q =
                            Eigen::AngleAxisd(
                                coherent_yaw_rad,
                                Eigen::Vector3d::UnitZ());
                        coherent_yaw_q.normalize();
                        const vector<3> coherent_translation =
                            recovery_body_pose.p -
                            coherent_yaw_q *
                                authoritative_body_pose.p;

                        struct CoherentCommitDryRunState {
                            Frame *frame = nullptr;
                            bool subframe = false;
                            PoseState before_pose;
                            MotionState before_motion;
                            PoseState proposed_pose;
                            MotionState proposed_motion;
                        };

                        std::vector<CoherentCommitDryRunState>
                            coherent_states;
                        coherent_states.reserve(
                            map->frame_num() * 2);

                        std::vector<std::unique_ptr<Frame>>
                            coherent_top_level_owners;
                        std::unordered_map<Frame *, Frame *>
                            coherent_top_level_clones;
                        coherent_top_level_owners.reserve(
                            map->frame_num());
                        coherent_top_level_clones.reserve(
                            map->frame_num());

                        size_t coherent_top_level_count = 0;
                        size_t coherent_subframe_count = 0;
                        bool coherent_source_finite = true;
                        bool coherent_proposed_finite = true;

                        const auto coherent_pose_finite =
                            [](const PoseState &pose) {
                                return pose.p.allFinite() &&
                                       pose.q.coeffs()
                                           .allFinite() &&
                                       std::isfinite(
                                           pose.q.norm()) &&
                                       pose.q.norm() >
                                           1.0e-12;
                            };
                        const auto coherent_motion_finite =
                            [](const MotionState &motion) {
                                return motion.v.allFinite() &&
                                       motion.bg.allFinite() &&
                                       motion.ba.allFinite();
                            };
                        const auto coherent_transform_state =
                            [&coherent_yaw_q,
                             &coherent_translation](
                                Frame *source,
                                bool subframe) {
                                CoherentCommitDryRunState
                                    snapshot;
                                snapshot.frame = source;
                                snapshot.subframe = subframe;
                                snapshot.before_pose =
                                    source->pose;
                                snapshot.before_motion =
                                    source->motion;
                                snapshot.proposed_pose =
                                    source->pose;
                                snapshot.proposed_motion =
                                    source->motion;
                                snapshot.proposed_pose.q =
                                    coherent_yaw_q *
                                    source->pose.q;
                                snapshot.proposed_pose.q
                                    .normalize();
                                snapshot.proposed_pose.p =
                                    coherent_yaw_q *
                                        source->pose.p +
                                    coherent_translation;
                                snapshot.proposed_motion.v =
                                    coherent_yaw_q *
                                    source->motion.v;
                                return snapshot;
                            };

                        for (size_t i = 0;
                             i < map->frame_num();
                             ++i) {
                            Frame *top = map->get_frame(i);
                            if (!top)
                                continue;

                            CoherentCommitDryRunState
                                top_snapshot =
                                    coherent_transform_state(
                                        top, false);
                            coherent_source_finite =
                                coherent_source_finite &&
                                coherent_pose_finite(
                                    top_snapshot.before_pose) &&
                                coherent_motion_finite(
                                    top_snapshot.before_motion);
                            coherent_proposed_finite =
                                coherent_proposed_finite &&
                                coherent_pose_finite(
                                    top_snapshot
                                        .proposed_pose) &&
                                coherent_motion_finite(
                                    top_snapshot
                                        .proposed_motion);
                            coherent_states.emplace_back(
                                top_snapshot);
                            ++coherent_top_level_count;

                            std::unique_ptr<Frame> owner =
                                top->clone();
                            Frame *shadow = owner.get();
                            shadow->pose =
                                top_snapshot.proposed_pose;
                            shadow->motion =
                                top_snapshot.proposed_motion;
                            coherent_top_level_clones[top] =
                                shadow;
                            coherent_top_level_owners
                                .emplace_back(
                                    std::move(owner));

                            for (const auto &sub_owner :
                                 top->subframes) {
                                Frame *sub =
                                    sub_owner.get();
                                if (!sub)
                                    continue;
                                CoherentCommitDryRunState
                                    sub_snapshot =
                                        coherent_transform_state(
                                            sub, true);
                                coherent_source_finite =
                                    coherent_source_finite &&
                                    coherent_pose_finite(
                                        sub_snapshot
                                            .before_pose) &&
                                    coherent_motion_finite(
                                        sub_snapshot
                                            .before_motion);
                                coherent_proposed_finite =
                                    coherent_proposed_finite &&
                                    coherent_pose_finite(
                                        sub_snapshot
                                            .proposed_pose) &&
                                    coherent_motion_finite(
                                        sub_snapshot
                                            .proposed_motion);
                                coherent_states.emplace_back(
                                    sub_snapshot);
                                ++coherent_subframe_count;
                            }
                        }

                        const auto coherent_rotation_delta_deg =
                            [](const quaternion &a,
                               const quaternion &b) {
                                quaternion delta =
                                    a.conjugate() * b;
                                delta.normalize();
                                return 2.0 *
                                       std::atan2(
                                           delta.vec().norm(),
                                           std::abs(delta.w())) *
                                       180.0 / M_PI;
                            };

                        double coherent_max_relative_t_error =
                            0.0;
                        double coherent_max_relative_r_error =
                            0.0;
                        for (size_t i = 0;
                             i < coherent_states.size();
                             ++i) {
                            for (size_t j = i + 1;
                                 j < coherent_states.size();
                                 ++j) {
                                const auto &a =
                                    coherent_states[i];
                                const auto &b =
                                    coherent_states[j];

                                const vector<3>
                                    before_relative_p =
                                        a.before_pose.q
                                            .conjugate() *
                                        (b.before_pose.p -
                                         a.before_pose.p);
                                const vector<3>
                                    proposed_relative_p =
                                        a.proposed_pose.q
                                            .conjugate() *
                                        (b.proposed_pose.p -
                                         a.proposed_pose.p);
                                coherent_max_relative_t_error =
                                    std::max(
                                        coherent_max_relative_t_error,
                                        (proposed_relative_p -
                                         before_relative_p)
                                            .norm());

                                const quaternion
                                    before_relative_q =
                                        a.before_pose.q
                                            .conjugate() *
                                        b.before_pose.q;
                                const quaternion
                                    proposed_relative_q =
                                        a.proposed_pose.q
                                            .conjugate() *
                                        b.proposed_pose.q;
                                coherent_max_relative_r_error =
                                    std::max(
                                        coherent_max_relative_r_error,
                                        coherent_rotation_delta_deg(
                                            before_relative_q,
                                            proposed_relative_q));
                            }
                        }

                        double
                            coherent_max_velocity_transform_error =
                                0.0;
                        double
                            coherent_max_velocity_norm_error =
                                0.0;
                        double coherent_max_bg_delta = 0.0;
                        double coherent_max_ba_delta = 0.0;
                        for (const auto &snapshot :
                             coherent_states) {
                            coherent_max_velocity_transform_error =
                                std::max(
                                    coherent_max_velocity_transform_error,
                                    (snapshot.proposed_motion.v -
                                     coherent_yaw_q *
                                         snapshot.before_motion.v)
                                        .norm());
                            coherent_max_velocity_norm_error =
                                std::max(
                                    coherent_max_velocity_norm_error,
                                    std::abs(
                                        snapshot
                                            .proposed_motion.v
                                            .norm() -
                                        snapshot
                                            .before_motion.v
                                            .norm()));
                            coherent_max_bg_delta =
                                std::max(
                                    coherent_max_bg_delta,
                                    (snapshot
                                         .proposed_motion.bg -
                                     snapshot.before_motion.bg)
                                        .norm());
                            coherent_max_ba_delta =
                                std::max(
                                    coherent_max_ba_delta,
                                    (snapshot
                                         .proposed_motion.ba -
                                     snapshot.before_motion.ba)
                                        .norm());
                        }

                        PoseState coherent_target_pose =
                            authoritative_body_pose;
                        MotionState coherent_target_motion =
                            authoritative_motion;
                        bool coherent_target_found = false;
                        for (const auto &snapshot :
                             coherent_states) {
                            if (snapshot.frame == frame) {
                                coherent_target_pose =
                                    snapshot.proposed_pose;
                                coherent_target_motion =
                                    snapshot.proposed_motion;
                                coherent_target_found = true;
                                break;
                            }
                        }

                        PoseState
                            expected_coherent_target_pose;
                        expected_coherent_target_pose.q =
                            coherent_yaw_q *
                            authoritative_body_pose.q;
                        expected_coherent_target_pose.q.normalize();
                        expected_coherent_target_pose.p =
                            recovery_body_pose.p;
                        const double
                            coherent_target_position_error =
                                (coherent_target_pose.p -
                                 recovery_body_pose.p)
                                    .norm();
                        const double
                            coherent_target_orientation_formula_error =
                                coherent_rotation_delta_deg(
                                    expected_coherent_target_pose.q,
                                    coherent_target_pose.q);
                        const double
                            coherent_target_to_full_recovery_r =
                                camera_rotation_delta_deg(
                                    recovery_body_pose,
                                    coherent_target_pose);
                        const double
                            coherent_target_v_transform_error =
                                (coherent_target_motion.v -
                                 coherent_yaw_q *
                                     authoritative_motion.v)
                                    .norm();

                        std::vector<Frame *>
                            live_prior_frames;
                        if (map->marginalization_factor) {
                            live_prior_frames =
                                map->marginalization_factor
                                    ->linearization_frames();
                        }
                        std::vector<Frame *>
                            coherent_prior_frames;
                        coherent_prior_frames.reserve(
                            live_prior_frames.size());
                        bool coherent_prior_mapping_complete =
                            map->marginalization_factor !=
                            nullptr;
                        for (Frame *prior_frame :
                             live_prior_frames) {
                            const auto it =
                                coherent_top_level_clones
                                    .find(prior_frame);
                            if (it ==
                                coherent_top_level_clones
                                    .end()) {
                                coherent_prior_mapping_complete =
                                    false;
                                break;
                            }
                            coherent_prior_frames
                                .emplace_back(it->second);
                        }

                        std::unique_ptr<MarginalizationFactor>
                            coherent_rebased_prior;
                        if (coherent_prior_mapping_complete) {
                            coherent_rebased_prior =
                                map->marginalization_factor
                                    ->clone_rebased_world(
                                        coherent_prior_frames,
                                        coherent_yaw_q,
                                        coherent_translation);
                        }
                        const bool
                            coherent_marginalization_rebase_success =
                                coherent_rebased_prior != nullptr;
                        bool coherent_rebased_prior_frames_match =
                            coherent_marginalization_rebase_success &&
                            coherent_rebased_prior
                                    ->linearization_frames()
                                    .size() ==
                                coherent_prior_frames.size();
                        if (coherent_rebased_prior_frames_match) {
                            const auto &rebased_frames =
                                coherent_rebased_prior
                                    ->linearization_frames();
                            for (size_t i = 0;
                                 i < rebased_frames.size();
                                 ++i) {
                                if (rebased_frames[i] !=
                                    coherent_prior_frames[i]) {
                                    coherent_rebased_prior_frames_match =
                                        false;
                                    break;
                                }
                            }
                        }

                        std::vector<std::pair<Track *, double>>
                            coherent_inverse_depth_snapshot;
                        coherent_inverse_depth_snapshot.reserve(
                            map->track_num());
                        for (size_t i = 0;
                             i < map->track_num();
                             ++i) {
                            Track *track =
                                map->get_track(i);
                            if (track) {
                                coherent_inverse_depth_snapshot
                                    .emplace_back(
                                        track,
                                        track->landmark
                                            .inv_depth);
                            }
                        }

                        double coherent_live_max_pose_t_delta =
                            0.0;
                        double coherent_live_max_q_coeff_delta =
                            0.0;
                        double coherent_live_max_v_delta = 0.0;
                        double coherent_live_max_bg_delta = 0.0;
                        double coherent_live_max_ba_delta = 0.0;
                        for (const auto &snapshot :
                             coherent_states) {
                            coherent_live_max_pose_t_delta =
                                std::max(
                                    coherent_live_max_pose_t_delta,
                                    (snapshot.frame->pose.p -
                                     snapshot.before_pose.p)
                                        .norm());
                            coherent_live_max_q_coeff_delta =
                                std::max(
                                    coherent_live_max_q_coeff_delta,
                                    (snapshot.frame
                                         ->pose.q.coeffs() -
                                     snapshot
                                         .before_pose.q
                                         .coeffs())
                                        .norm());
                            coherent_live_max_v_delta =
                                std::max(
                                    coherent_live_max_v_delta,
                                    (snapshot.frame->motion.v -
                                     snapshot.before_motion.v)
                                        .norm());
                            coherent_live_max_bg_delta =
                                std::max(
                                    coherent_live_max_bg_delta,
                                    (snapshot.frame->motion.bg -
                                     snapshot.before_motion.bg)
                                        .norm());
                            coherent_live_max_ba_delta =
                                std::max(
                                    coherent_live_max_ba_delta,
                                    (snapshot.frame->motion.ba -
                                     snapshot.before_motion.ba)
                                        .norm());
                        }

                        double
                            coherent_live_max_inv_depth_delta =
                                0.0;
                        bool
                            coherent_inverse_depth_unchanged =
                                true;
                        for (const auto &[track, before] :
                             coherent_inverse_depth_snapshot) {
                            const double after =
                                track->landmark.inv_depth;
                            const bool equal =
                                before == after ||
                                (std::isnan(before) &&
                                 std::isnan(after));
                            coherent_inverse_depth_unchanged =
                                coherent_inverse_depth_unchanged &&
                                equal;
                            if (std::isfinite(before) &&
                                std::isfinite(after)) {
                                coherent_live_max_inv_depth_delta =
                                    std::max(
                                        coherent_live_max_inv_depth_delta,
                                        std::abs(after - before));
                            }
                        }

                        const bool
                            coherent_all_live_state_unchanged =
                                coherent_live_max_pose_t_delta ==
                                    0.0 &&
                                coherent_live_max_q_coeff_delta ==
                                    0.0 &&
                                coherent_live_max_v_delta ==
                                    0.0 &&
                                coherent_live_max_bg_delta ==
                                    0.0 &&
                                coherent_live_max_ba_delta ==
                                    0.0 &&
                                coherent_inverse_depth_unchanged;

                        const bool
                            coherent_relative_geometry_invariant =
                                coherent_max_relative_t_error <=
                                    1.0e-9 &&
                                coherent_max_relative_r_error <=
                                    1.0e-7;
                        const bool
                            coherent_motion_transform_valid =
                                coherent_max_velocity_transform_error <=
                                    1.0e-12 &&
                                coherent_max_bg_delta == 0.0 &&
                                coherent_max_ba_delta == 0.0;
                        const bool
                            coherent_target_transform_valid =
                                coherent_target_found &&
                                coherent_target_position_error <=
                                    1.0e-12 &&
                                coherent_target_orientation_formula_error <=
                                    1.0e-7 &&
                                coherent_target_v_transform_error <=
                                    1.0e-12;
                        const bool coherent_target_ready =
                            target_is_latest;
                        const bool coherent_dry_run_ready =
                            coherent_target_ready &&
                            coherent_source_finite &&
                            coherent_proposed_finite &&
                            coherent_relative_geometry_invariant &&
                            coherent_motion_transform_valid &&
                            coherent_target_transform_valid &&
                            coherent_prior_mapping_complete &&
                            coherent_marginalization_rebase_success &&
                            coherent_rebased_prior_frames_match &&
                            coherent_all_live_state_unchanged;

                        std::fprintf(
                            stderr,
                            "[PlaceRecoveryCoherentCommitDryRun] "
                            "current=%zu t=%.9f event_id=%zu "
                            "acceptance_candidate=1 "
                            "commit_enabled=%d "
                            "commit_scope=active_window_gauge "
                            "gravity_policy=yaw_translation "
                            "velocity_transform=world_yaw "
                            "bias_transform=unchanged "
                            "inverse_depth_transform=unchanged "
                            "target_is_latest=%d "
                            "target_has_subframes=%d "
                            "supports_target_subframes=1 "
                            "coherent_target_ready=%d "
                            "coherent_dry_run_ready=%d "
                            "top_level_frames=%zu "
                            "subframes=%zu active_states=%zu "
                            "prior_frames=%zu "
                            "yaw_correction_deg=%.9f "
                            "full_recovery_correction_r_deg=%.9f "
                            "translation=%.9f,%.9f,%.9f "
                            "source_state_finite=%d "
                            "proposed_state_finite=%d "
                            "relative_geometry_invariant=%d "
                            "max_relative_t_error=%.12g "
                            "max_relative_r_error_deg=%.12g "
                            "motion_transform_valid=%d "
                            "max_velocity_transform_error=%.12g "
                            "max_velocity_norm_error=%.12g "
                            "max_bg_delta=%.12g "
                            "max_ba_delta=%.12g "
                            "target_found=%d "
                            "target_position_error=%.12g "
                            "target_orientation_formula_error_deg=%.12g "
                            "target_to_full_recovery_r_deg=%.9f "
                            "target_v_transform_error=%.12g "
                            "prior_mapping_complete=%d "
                            "marginalization_rebase_success=%d "
                            "rebased_prior_frames_match=%d "
                            "inverse_depth_states=%zu "
                            "inverse_depth_unchanged=%d "
                            "live_max_inv_depth_delta=%.12g "
                            "live_max_pose_t_delta=%.12g "
                            "live_max_q_coeff_delta=%.12g "
                            "live_max_v_delta=%.12g "
                            "live_max_bg_delta=%.12g "
                            "live_max_ba_delta=%.12g "
                            "live_state_unchanged=%d "
                            "would_transform_top_level=1 "
                            "would_transform_subframes=1 "
                            "would_rebase_marginalization=1 "
                            "would_mutate_inverse_depth=0 "
                            "estimator_reset=0 loop_constraint=0 "
                            "commit_applied=0 state_mutation=0\n",
                            frame->id(),
                            frame->image->t,
                            group.event_id,
                            commit_enabled ? 1 : 0,
                            target_is_latest ? 1 : 0,
                            target_has_subframes ? 1 : 0,
                            coherent_target_ready ? 1 : 0,
                            coherent_dry_run_ready ? 1 : 0,
                            coherent_top_level_count,
                            coherent_subframe_count,
                            coherent_states.size(),
                            live_prior_frames.size(),
                            std::abs(coherent_yaw_rad) *
                                180.0 / M_PI,
                            commit_delta_r,
                            coherent_translation.x(),
                            coherent_translation.y(),
                            coherent_translation.z(),
                            coherent_source_finite ? 1 : 0,
                            coherent_proposed_finite ? 1 : 0,
                            coherent_relative_geometry_invariant
                                ? 1
                                : 0,
                            coherent_max_relative_t_error,
                            coherent_max_relative_r_error,
                            coherent_motion_transform_valid
                                ? 1
                                : 0,
                            coherent_max_velocity_transform_error,
                            coherent_max_velocity_norm_error,
                            coherent_max_bg_delta,
                            coherent_max_ba_delta,
                            coherent_target_found ? 1 : 0,
                            coherent_target_position_error,
                            coherent_target_orientation_formula_error,
                            coherent_target_to_full_recovery_r,
                            coherent_target_v_transform_error,
                            coherent_prior_mapping_complete
                                ? 1
                                : 0,
                            coherent_marginalization_rebase_success
                                ? 1
                                : 0,
                            coherent_rebased_prior_frames_match
                                ? 1
                                : 0,
                            coherent_inverse_depth_snapshot.size(),
                            coherent_inverse_depth_unchanged
                                ? 1
                                : 0,
                            coherent_live_max_inv_depth_delta,
                            coherent_live_max_pose_t_delta,
                            coherent_live_max_q_coeff_delta,
                            coherent_live_max_v_delta,
                            coherent_live_max_bg_delta,
                            coherent_live_max_ba_delta,
                            coherent_all_live_state_unchanged
                                ? 1
                                : 0);

                        if (commit_enabled) {
                            // Prepare a second rebased prior bound to the
                            // actual live frame storage. The 0110k shadow
                            // prior is clone-bound and must never be moved
                            // into the estimator.
                            std::unique_ptr<MarginalizationFactor>
                                live_rebased_prior;
                            if (coherent_dry_run_ready &&
                                map->marginalization_factor) {
                                live_rebased_prior =
                                    map->marginalization_factor
                                        ->clone_rebased_world(
                                            live_prior_frames,
                                            coherent_yaw_q,
                                            coherent_translation);
                            }

                            const bool live_rebased_prior_ready =
                                live_rebased_prior != nullptr;
                            const bool coherent_commit_preconditions_met =
                                coherent_dry_run_ready &&
                                live_rebased_prior_ready;

                            if (!coherent_commit_preconditions_met) {
                                const char *skip_reason =
                                    !target_is_latest
                                        ? "target_not_latest"
                                    : !live_state_unchanged
                                        ? "precommit_state_changed"
                                    : !coherent_dry_run_ready
                                        ? "coherent_dry_run_failed"
                                    : !live_rebased_prior_ready
                                        ? "live_marginalization_rebase_failed"
                                    : "unknown";

                                std::fprintf(
                                    stderr,
                                    "[PlaceRecoveryCommit] "
                                    "current=%zu t=%.9f event_id=%zu "
                                    "commit_scope=active_window_gauge "
                                    "gravity_policy=yaw_translation "
                                    "velocity_transform=world_yaw "
                                    "bias_transform=unchanged "
                                    "inverse_depth_transform=unchanged "
                                    "commit_enabled=1 "
                                    "commit_preconditions_met=0 "
                                    "reason=%s "
                                    "target_is_latest=%d "
                                    "target_has_subframes=%d "
                                    "coherent_dry_run_ready=%d "
                                    "live_rebased_prior_ready=%d "
                                    "estimator_reset=0 loop_constraint=0 "
                                    "commit_applied=0 state_mutation=0\n",
                                    frame->id(), frame->image->t,
                                    group.event_id,
                                    skip_reason,
                                    target_is_latest ? 1 : 0,
                                    target_has_subframes ? 1 : 0,
                                    coherent_dry_run_ready ? 1 : 0,
                                    live_rebased_prior_ready ? 1 : 0);
                            } else {
                                MarginalizationFactor
                                    *prepared_live_prior =
                                        live_rebased_prior.get();

                                double max_applied_pose_t = 0.0;
                                double max_applied_pose_r_deg = 0.0;
                                double max_applied_v_delta = 0.0;
                                double max_applied_bg_delta = 0.0;
                                double max_applied_ba_delta = 0.0;
                                for (const auto &snapshot :
                                     coherent_states) {
                                    max_applied_pose_t =
                                        std::max(
                                            max_applied_pose_t,
                                            (snapshot.proposed_pose.p -
                                             snapshot.before_pose.p)
                                                .norm());
                                    max_applied_pose_r_deg =
                                        std::max(
                                            max_applied_pose_r_deg,
                                            coherent_rotation_delta_deg(
                                                snapshot.before_pose.q,
                                                snapshot.proposed_pose.q));
                                    max_applied_v_delta =
                                        std::max(
                                            max_applied_v_delta,
                                            (snapshot
                                                 .proposed_motion.v -
                                             snapshot.before_motion.v)
                                                .norm());
                                    max_applied_bg_delta =
                                        std::max(
                                            max_applied_bg_delta,
                                            (snapshot
                                                 .proposed_motion.bg -
                                             snapshot.before_motion.bg)
                                                .norm());
                                    max_applied_ba_delta =
                                        std::max(
                                            max_applied_ba_delta,
                                            (snapshot
                                                 .proposed_motion.ba -
                                             snapshot.before_motion.ba)
                                                .norm());
                                }

                                // All fallible preparation is complete.
                                // Apply exactly the state vector validated by
                                // 0110k, then replace ownership of the
                                // stale-gauge marginalization prior.
                                size_t applied_top_level_frames = 0;
                                size_t applied_subframes = 0;
                                for (const auto &snapshot :
                                     coherent_states) {
                                    snapshot.frame->pose =
                                        snapshot.proposed_pose;
                                    snapshot.frame->motion =
                                        snapshot.proposed_motion;
                                    if (snapshot.subframe)
                                        ++applied_subframes;
                                    else
                                        ++applied_top_level_frames;
                                }
                                map->marginalization_factor =
                                    std::move(live_rebased_prior);

                                const bool prior_replaced =
                                    map->marginalization_factor &&
                                    map->marginalization_factor.get() ==
                                        prepared_live_prior;

                                bool committed_prior_frames_match =
                                    prior_replaced &&
                                    map->marginalization_factor
                                            ->linearization_frames()
                                            .size() ==
                                        live_prior_frames.size();
                                if (committed_prior_frames_match) {
                                    const auto &committed_prior_frames =
                                        map->marginalization_factor
                                            ->linearization_frames();
                                    for (size_t i = 0;
                                         i < committed_prior_frames.size();
                                         ++i) {
                                        if (committed_prior_frames[i] !=
                                            live_prior_frames[i]) {
                                            committed_prior_frames_match =
                                                false;
                                            break;
                                        }
                                    }
                                }

                                double max_pose_proposal_error_t = 0.0;
                                double max_pose_proposal_error_r_deg = 0.0;
                                double max_motion_proposal_error_v = 0.0;
                                double max_motion_proposal_error_bg = 0.0;
                                double max_motion_proposal_error_ba = 0.0;
                                for (const auto &snapshot :
                                     coherent_states) {
                                    max_pose_proposal_error_t =
                                        std::max(
                                            max_pose_proposal_error_t,
                                            (snapshot.frame->pose.p -
                                             snapshot.proposed_pose.p)
                                                .norm());
                                    max_pose_proposal_error_r_deg =
                                        std::max(
                                            max_pose_proposal_error_r_deg,
                                            coherent_rotation_delta_deg(
                                                snapshot.proposed_pose.q,
                                                snapshot.frame->pose.q));
                                    max_motion_proposal_error_v =
                                        std::max(
                                            max_motion_proposal_error_v,
                                            (snapshot.frame->motion.v -
                                             snapshot
                                                 .proposed_motion.v)
                                                .norm());
                                    max_motion_proposal_error_bg =
                                        std::max(
                                            max_motion_proposal_error_bg,
                                            (snapshot.frame->motion.bg -
                                             snapshot
                                                 .proposed_motion.bg)
                                                .norm());
                                    max_motion_proposal_error_ba =
                                        std::max(
                                            max_motion_proposal_error_ba,
                                            (snapshot.frame->motion.ba -
                                             snapshot
                                                 .proposed_motion.ba)
                                                .norm());
                                }

                                bool inverse_depth_unchanged_after_commit =
                                    true;
                                double max_inv_depth_delta_after_commit =
                                    0.0;
                                for (const auto &[track, before] :
                                     coherent_inverse_depth_snapshot) {
                                    const double after =
                                        track->landmark.inv_depth;
                                    const bool equal =
                                        before == after ||
                                        (std::isnan(before) &&
                                         std::isnan(after));
                                    inverse_depth_unchanged_after_commit =
                                        inverse_depth_unchanged_after_commit &&
                                        equal;
                                    if (std::isfinite(before) &&
                                        std::isfinite(after)) {
                                        max_inv_depth_delta_after_commit =
                                            std::max(
                                                max_inv_depth_delta_after_commit,
                                                std::abs(after - before));
                                    }
                                }

                                // Rebase prior recovery reconciliation
                                // snapshots into the same new gauge. Their
                                // fixed-world recovery landmarks intentionally
                                // remain in the archive/global frame.
                                size_t reconciliation_states_rebased = 0;
                                for (auto &existing :
                                     active_place_recovery_commit_reconciliations_) {
                                    existing.precommit_body_pose.q =
                                        coherent_yaw_q *
                                        existing.precommit_body_pose.q;
                                    existing.precommit_body_pose.q.normalize();
                                    existing.precommit_body_pose.p =
                                        coherent_yaw_q *
                                            existing.precommit_body_pose.p +
                                        coherent_translation;

                                    existing.committed_body_pose.q =
                                        coherent_yaw_q *
                                        existing.committed_body_pose.q;
                                    existing.committed_body_pose.q.normalize();
                                    existing.committed_body_pose.p =
                                        coherent_yaw_q *
                                            existing.committed_body_pose.p +
                                        coherent_translation;
                                    existing.committed_motion.v =
                                        coherent_yaw_q *
                                        existing.committed_motion.v;
                                    ++reconciliation_states_rebased;
                                }

                                const double target_applied_delta_t =
                                    (frame->pose.p -
                                     authoritative_body_pose.p)
                                        .norm();
                                const double target_applied_delta_r =
                                    coherent_rotation_delta_deg(
                                        authoritative_body_pose.q,
                                        frame->pose.q);
                                const double target_to_full_recovery_t =
                                    (frame->pose.p -
                                     recovery_body_pose.p)
                                        .norm();
                                const double target_to_full_recovery_r =
                                    coherent_rotation_delta_deg(
                                        recovery_body_pose.q,
                                        frame->pose.q);
                                const double target_v_transform_error =
                                    (frame->motion.v -
                                     coherent_yaw_q *
                                         authoritative_motion.v)
                                        .norm();
                                const double target_bg_delta =
                                    (frame->motion.bg -
                                     authoritative_motion.bg)
                                        .norm();
                                const double target_ba_delta =
                                    (frame->motion.ba -
                                     authoritative_motion.ba)
                                        .norm();

                                const bool committed_state_matches_proposal =
                                    max_pose_proposal_error_t == 0.0 &&
                                    max_pose_proposal_error_r_deg <=
                                        1.0e-12 &&
                                    max_motion_proposal_error_v == 0.0 &&
                                    max_motion_proposal_error_bg == 0.0 &&
                                    max_motion_proposal_error_ba == 0.0;
                                const bool coherent_commit_postconditions =
                                    prior_replaced &&
                                    committed_prior_frames_match &&
                                    committed_state_matches_proposal &&
                                    inverse_depth_unchanged_after_commit &&
                                    target_to_full_recovery_t <= 1.0e-12 &&
                                    target_v_transform_error <= 1.0e-12 &&
                                    target_bg_delta == 0.0 &&
                                    target_ba_delta == 0.0;

                                std::fprintf(
                                    stderr,
                                    "[PlaceRecoveryCommit] "
                                    "current=%zu t=%.9f event_id=%zu "
                                    "commit_scope=active_window_gauge "
                                    "gravity_policy=yaw_translation "
                                    "velocity_transform=world_yaw "
                                    "bias_transform=unchanged "
                                    "inverse_depth_transform=unchanged "
                                    "commit_enabled=1 "
                                    "commit_preconditions_met=1 "
                                    "reason=committed "
                                    "coherent_dry_run_ready=1 "
                                    "top_level_frames=%zu "
                                    "subframes=%zu active_states=%zu "
                                    "prior_frames=%zu "
                                    "yaw_correction_deg=%.9f "
                                    "translation=%.9f,%.9f,%.9f "
                                    "target_applied_delta_t=%.9f "
                                    "target_applied_delta_r_deg=%.9f "
                                    "target_to_full_recovery_t=%.12g "
                                    "target_to_full_recovery_r_deg=%.9f "
                                    "target_v_transform_error=%.12g "
                                    "target_bg_delta=%.12g "
                                    "target_ba_delta=%.12g "
                                    "max_applied_pose_t=%.9f "
                                    "max_applied_pose_r_deg=%.9f "
                                    "max_applied_v_delta=%.9f "
                                    "max_applied_bg_delta=%.12g "
                                    "max_applied_ba_delta=%.12g "
                                    "applied_top_level_frames=%zu "
                                    "applied_subframes=%zu "
                                    "prior_replaced=%d "
                                    "committed_prior_frames_match=%d "
                                    "max_pose_proposal_error_t=%.12g "
                                    "max_pose_proposal_error_r_deg=%.12g "
                                    "max_motion_proposal_error_v=%.12g "
                                    "max_motion_proposal_error_bg=%.12g "
                                    "max_motion_proposal_error_ba=%.12g "
                                    "inverse_depth_states=%zu "
                                    "inverse_depth_unchanged=%d "
                                    "max_inv_depth_delta=%.12g "
                                    "reconciliation_states_rebased=%zu "
                                    "commit_postconditions=%d "
                                    "full_recovery_orientation_applied=0 "
                                    "estimator_reset=0 loop_constraint=0 "
                                    "archive_refresh_deferred=1 "
                                    "commit_applied=1 state_mutation=1\n",
                                    frame->id(), frame->image->t,
                                    group.event_id,
                                    coherent_top_level_count,
                                    coherent_subframe_count,
                                    coherent_states.size(),
                                    live_prior_frames.size(),
                                    std::abs(coherent_yaw_rad) *
                                        180.0 / M_PI,
                                    coherent_translation.x(),
                                    coherent_translation.y(),
                                    coherent_translation.z(),
                                    target_applied_delta_t,
                                    target_applied_delta_r,
                                    target_to_full_recovery_t,
                                    target_to_full_recovery_r,
                                    target_v_transform_error,
                                    target_bg_delta,
                                    target_ba_delta,
                                    max_applied_pose_t,
                                    max_applied_pose_r_deg,
                                    max_applied_v_delta,
                                    max_applied_bg_delta,
                                    max_applied_ba_delta,
                                    applied_top_level_frames,
                                    applied_subframes,
                                    prior_replaced ? 1 : 0,
                                    committed_prior_frames_match ? 1 : 0,
                                    max_pose_proposal_error_t,
                                    max_pose_proposal_error_r_deg,
                                    max_motion_proposal_error_v,
                                    max_motion_proposal_error_bg,
                                    max_motion_proposal_error_ba,
                                    coherent_inverse_depth_snapshot.size(),
                                    inverse_depth_unchanged_after_commit
                                        ? 1
                                        : 0,
                                    max_inv_depth_delta_after_commit,
                                    reconciliation_states_rebased,
                                    coherent_commit_postconditions ? 1 : 0);

                                PlaceRecoveryCommitReconciliationState
                                    reconciliation;
                                reconciliation.event_id = group.event_id;
                                reconciliation.frame_id = frame->id();
                                reconciliation.timestamp = frame->image->t;
                                reconciliation.precommit_body_pose =
                                    authoritative_body_pose;
                                reconciliation.committed_body_pose =
                                    frame->pose;
                                reconciliation.committed_motion =
                                    frame->motion;
                                reconciliation.recovery_landmarks_world =
                                    representative
                                        .inlier_landmarks_world;
                                reconciliation.recovery_observations_pixel =
                                    representative
                                        .inlier_observations_pixel;
                                active_place_recovery_commit_reconciliations_
                                    .emplace_back(
                                        std::move(reconciliation));
                            }
                        }
                    }
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
            !local_descriptor_shadow_enabled() ||
            (archived &&
             archived->local_descriptors_complete);
        if (place_ready && local_ready) {
            frame->image
                ->retain_place_recognition_source(false);
        }
    }
}

void SlidingWindowTracker::
diagnose_place_recovery_commit_reconciliation() {
    if (active_place_recovery_commit_reconciliations_.empty())
        return;

    const size_t latest_refined_frame_id =
        map->frame_num() > 0
            ? map->get_frame(map->frame_num() - 1)->id()
            : 0;

    std::vector<PlaceRecoveryCommitReconciliationState> active;
    active.reserve(
        active_place_recovery_commit_reconciliations_.size());

    for (auto &state :
         active_place_recovery_commit_reconciliations_) {
        Frame *target = nullptr;
        size_t target_top_index = nil();
        bool target_is_top_level = false;

        for (size_t i = 0;
             i < map->frame_num() && !target;
             ++i) {
            Frame *top = map->get_frame(i);
            if (!top)
                continue;

            if (top->id() == state.frame_id) {
                target = top;
                target_is_top_level = true;
                target_top_index = i;
                break;
            }

            for (const auto &subframe : top->subframes) {
                if (subframe &&
                    subframe->id() == state.frame_id) {
                    target = subframe.get();
                    break;
                }
            }
        }

        if (!target) {
            std::fprintf(
                stderr,
                "[PlaceRecoveryCommitReconcile] "
                "event_id=%zu commit_frame=%zu "
                "commit_t=%.9f target_active=0 "
                "samples_emitted=%zu retired=1 "
                "state_mutation=0\n",
                state.event_id,
                state.frame_id,
                state.timestamp,
                state.samples_emitted);
            continue;
        }

        if (place_recovery_factor_shadow_enabled() &&
            !state.factor_shadow_emitted) {
            state.factor_shadow_emitted = true;

            if (!target_is_top_level ||
                target_top_index == nil() ||
                state.recovery_landmarks_world.empty() ||
                state.recovery_landmarks_world.size() !=
                    state.recovery_observations_pixel.size()) {
                const char *reject_reason =
                    !target_is_top_level
                        ? "target_not_top_level"
                    : target_top_index == nil()
                        ? "target_index_unavailable"
                    : state.recovery_landmarks_world.empty()
                        ? "recovery_observations_empty"
                        : "recovery_observation_size_mismatch";

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryFactorIntegrationShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f "
                    "reject=%s "
                    "topology=localize_single_frame "
                    "marginalization_prior=0 "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    reject_reason);
            } else {
                struct FactorShadowResult {
                    bool usable = false;
                    bool imu_prior = false;
                    size_t conventional_visual_factors = 0;
                    size_t recovery_factors = 0;
                    size_t self_anchor_skipped = 0;
                    PoseState pose;
                    MotionState motion;
                    double recovery_rmse_px =
                        std::numeric_limits<double>::quiet_NaN();
                };

                const auto recovery_rmse_px =
                    [&state, target](
                        const PoseState &body_pose) {
                        PoseState camera_pose;
                        camera_pose.q =
                            body_pose.q * target->camera.q_cs;
                        camera_pose.p =
                            body_pose.p +
                            body_pose.q * target->camera.p_cs;

                        double squared_error = 0.0;
                        size_t positive_depth_count = 0;
                        for (size_t i = 0;
                             i < state.recovery_landmarks_world.size();
                             ++i) {
                            const vector<3> point_camera =
                                camera_pose.q.conjugate() *
                                (state.recovery_landmarks_world[i] -
                                 camera_pose.p);
                            if (!point_camera.allFinite() ||
                                point_camera.z() <= 1.0e-6) {
                                continue;
                            }

                            const vector<2> projected =
                                apply_k(point_camera, target->K);
                            const double error =
                                (projected -
                                 state.recovery_observations_pixel[i])
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

                const auto run_factor_shadow =
                    [this, target, target_top_index, &state,
                     &recovery_rmse_px](
                        bool use_recovery_seed,
                        bool include_visual,
                        bool include_imu,
                        bool include_recovery) {
                        FactorShadowResult result;
                        result.pose = target->pose;
                        result.motion = target->motion;

                        auto shadow_map =
                            std::make_unique<Map>();
                        std::unique_ptr<Frame> target_owner =
                            target->clone();
                        Frame *shadow_target = target_owner.get();
                        shadow_target->tag(FT_FIX_POSE) = false;
                        shadow_target->tag(FT_FIX_MOTION) = false;
                        if (use_recovery_seed) {
                            shadow_target->pose =
                                state.committed_body_pose;
                        }
                        shadow_map->attach_frame(
                            std::move(target_owner));

                        auto solver = Solver::create();
                        solver->add_frame_states(
                            shadow_target, include_imu);

                        PreIntegrator shadow_preintegration;
                        if (include_imu &&
                            config->has_imu() &&
                            target_top_index > 0) {
                            Frame *previous =
                                map->get_frame(
                                    target_top_index - 1);
                            shadow_preintegration =
                                target->keyframe_preintegration;
                            if (previous &&
                                shadow_preintegration.integrate(
                                    target->image->t,
                                    previous->motion.bg,
                                    previous->motion.ba,
                                    true, true)) {
                                solver->put_factor(
                                    Solver::
                                        create_preintegration_prior_factor(
                                            previous,
                                            shadow_target,
                                            shadow_preintegration));
                                result.imu_prior = true;
                            }
                        }

                        if (include_visual) {
                            std::vector<Frame *> anchor_sources;
                            std::vector<Frame *> anchor_copies;

                            const auto get_anchor_copy =
                                [&shadow_map,
                                 &anchor_sources,
                                 &anchor_copies](
                                    Frame *source) {
                                    for (size_t i = 0;
                                         i < anchor_sources.size();
                                         ++i) {
                                        if (anchor_sources[i] ==
                                            source) {
                                            return anchor_copies[i];
                                        }
                                    }

                                    std::unique_ptr<Frame> owner =
                                        source->clone();
                                    Frame *copy = owner.get();
                                    shadow_map->attach_frame(
                                        std::move(owner));
                                    anchor_sources.emplace_back(
                                        source);
                                    anchor_copies.emplace_back(copy);
                                    return copy;
                                };

                            for (size_t k = 0;
                                 k < target->keypoint_num();
                                 ++k) {
                                Track *source_track =
                                    target->get_track(k);
                                if (!source_track ||
                                    !source_track->all_tagged(
                                        TT_VALID,
                                        TT_TRIANGULATED,
                                        TT_STATIC)) {
                                    continue;
                                }

                                Frame *source_anchor =
                                    source_track->first_frame();
                                if (!source_anchor ||
                                    source_anchor == target) {
                                    ++result.self_anchor_skipped;
                                    continue;
                                }

                                const size_t anchor_index =
                                    source_track
                                        ->get_keypoint_index(
                                            source_anchor);
                                if (anchor_index == nil() ||
                                    anchor_index >=
                                        source_anchor
                                            ->keypoint_num()) {
                                    continue;
                                }

                                Frame *shadow_anchor =
                                    get_anchor_copy(source_anchor);
                                Track *shadow_track =
                                    shadow_map->create_track();
                                shadow_track->add_keypoint(
                                    shadow_anchor,
                                    anchor_index);
                                shadow_track->add_keypoint(
                                    shadow_target, k);
                                shadow_track->landmark =
                                    source_track->landmark;
                                shadow_track->tag(TT_VALID) = true;
                                shadow_track->tag(
                                    TT_TRIANGULATED) = true;
                                shadow_track->tag(TT_STATIC) = true;
                                shadow_track->tag(
                                    TT_FIX_INVD) = true;

                                solver->put_factor(
                                    Solver::
                                        create_reprojection_prior_factor(
                                            shadow_target,
                                            shadow_track));
                                ++result
                                      .conventional_visual_factors;
                            }
                        }

                        if (include_recovery) {
                            for (size_t i = 0;
                                 i < state
                                         .recovery_landmarks_world.size();
                                 ++i) {
                                solver
                                    ->add_learned_world_reprojection(
                                        shadow_target,
                                        state
                                            .recovery_landmarks_world[i],
                                        state
                                            .recovery_observations_pixel[i]);
                                ++result.recovery_factors;
                            }
                        }

                        result.usable = solver->solve();
                        result.pose = shadow_target->pose;
                        result.motion = shadow_target->motion;
                        result.recovery_rmse_px =
                            recovery_rmse_px(
                                shadow_target->pose);
                        return result;
                    };

                const double current_recovery_rmse_px =
                    recovery_rmse_px(target->pose);
                const FactorShadowResult baseline =
                    run_factor_shadow(
                        false, true, true, false);
                const FactorShadowResult augmented =
                    run_factor_shadow(
                        false, true, true, true);

                // Recovery-seeded ablation. All four cases begin from the
                // exact accepted recovery pose. They differ only in which
                // ordinary local constraints are reintroduced, while the
                // recovery fixed-world factors remain identical:
                //   A: recovery only
                //   B: recovery + conventional visual
                //   C: recovery + incoming IMU
                //   D: recovery + conventional visual + incoming IMU
                const FactorShadowResult seeded_recovery_only =
                    run_factor_shadow(
                        true, false, false, true);
                const FactorShadowResult seeded_recovery_visual =
                    run_factor_shadow(
                        true, true, false, true);
                const FactorShadowResult seeded_recovery_imu =
                    run_factor_shadow(
                        true, false, true, true);
                const FactorShadowResult seeded_recovery_visual_imu =
                    run_factor_shadow(
                        true, true, true, true);

                const double baseline_delta_current_t =
                    (baseline.pose.p - target->pose.p).norm();
                const double baseline_delta_current_r =
                    camera_rotation_delta_deg(
                        target->pose, baseline.pose);
                const double augmented_delta_current_t =
                    (augmented.pose.p - target->pose.p).norm();
                const double augmented_delta_current_r =
                    camera_rotation_delta_deg(
                        target->pose, augmented.pose);
                const double augmented_delta_baseline_t =
                    (augmented.pose.p - baseline.pose.p).norm();
                const double augmented_delta_baseline_r =
                    camera_rotation_delta_deg(
                        baseline.pose, augmented.pose);

                const double augmented_to_committed_t =
                    (augmented.pose.p -
                     state.committed_body_pose.p)
                        .norm();
                const double augmented_to_committed_r =
                    camera_rotation_delta_deg(
                        state.committed_body_pose,
                        augmented.pose);
                const double augmented_to_precommit_t =
                    (augmented.pose.p -
                     state.precommit_body_pose.p)
                        .norm();
                const double augmented_to_precommit_r =
                    camera_rotation_delta_deg(
                        state.precommit_body_pose,
                        augmented.pose);

                const double baseline_v_delta =
                    (baseline.motion.v - target->motion.v).norm();
                const double baseline_bg_delta =
                    (baseline.motion.bg - target->motion.bg).norm();
                const double baseline_ba_delta =
                    (baseline.motion.ba - target->motion.ba).norm();
                const double augmented_v_delta =
                    (augmented.motion.v - target->motion.v).norm();
                const double augmented_bg_delta =
                    (augmented.motion.bg - target->motion.bg).norm();
                const double augmented_ba_delta =
                    (augmented.motion.ba - target->motion.ba).norm();

                const auto seeded_to_committed_t =
                    [&state](const FactorShadowResult &result) {
                        return (result.pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto seeded_to_committed_r =
                    [&state](const FactorShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.pose);
                    };
                const auto seeded_to_precommit_t =
                    [&state](const FactorShadowResult &result) {
                        return (result.pose.p -
                                state.precommit_body_pose.p)
                            .norm();
                    };
                const auto seeded_to_precommit_r =
                    [&state](const FactorShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.precommit_body_pose,
                            result.pose);
                    };

                const double visual_effect_from_recovery_only_t =
                    (seeded_recovery_visual.pose.p -
                     seeded_recovery_only.pose.p)
                        .norm();
                const double visual_effect_from_recovery_only_r =
                    camera_rotation_delta_deg(
                        seeded_recovery_only.pose,
                        seeded_recovery_visual.pose);
                const double imu_effect_from_recovery_only_t =
                    (seeded_recovery_imu.pose.p -
                     seeded_recovery_only.pose.p)
                        .norm();
                const double imu_effect_from_recovery_only_r =
                    camera_rotation_delta_deg(
                        seeded_recovery_only.pose,
                        seeded_recovery_imu.pose);
                const double combined_effect_from_recovery_only_t =
                    (seeded_recovery_visual_imu.pose.p -
                     seeded_recovery_only.pose.p)
                        .norm();
                const double combined_effect_from_recovery_only_r =
                    camera_rotation_delta_deg(
                        seeded_recovery_only.pose,
                        seeded_recovery_visual_imu.pose);

                const double seeded_recovery_imu_v_delta =
                    (seeded_recovery_imu.motion.v -
                     target->motion.v)
                        .norm();
                const double seeded_recovery_imu_bg_delta =
                    (seeded_recovery_imu.motion.bg -
                     target->motion.bg)
                        .norm();
                const double seeded_recovery_imu_ba_delta =
                    (seeded_recovery_imu.motion.ba -
                     target->motion.ba)
                        .norm();
                const double seeded_recovery_visual_imu_v_delta =
                    (seeded_recovery_visual_imu.motion.v -
                     target->motion.v)
                        .norm();
                const double seeded_recovery_visual_imu_bg_delta =
                    (seeded_recovery_visual_imu.motion.bg -
                     target->motion.bg)
                        .norm();
                const double seeded_recovery_visual_imu_ba_delta =
                    (seeded_recovery_visual_imu.motion.ba -
                     target->motion.ba)
                        .norm();

                std::fprintf(
                    stderr,
                    "[PlaceRecoverySeededAblationShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f "
                    "latest_refined_frame=%zu "
                    "seed=recovery_committed_pose "
                    "motion_seed=current_authoritative "
                    "marginalization_prior=0 "
                    "recovery_factors=%zu "
                    "conventional_visual_factors=%zu "
                    "recovery_only_usable=%d "
                    "recovery_only_rmse_px=%.9f "
                    "recovery_only_to_committed_t=%.9f "
                    "recovery_only_to_committed_r_deg=%.9f "
                    "recovery_only_to_precommit_t=%.9f "
                    "recovery_only_to_precommit_r_deg=%.9f "
                    "recovery_visual_usable=%d "
                    "recovery_visual_rmse_px=%.9f "
                    "recovery_visual_to_committed_t=%.9f "
                    "recovery_visual_to_committed_r_deg=%.9f "
                    "recovery_visual_to_precommit_t=%.9f "
                    "recovery_visual_to_precommit_r_deg=%.9f "
                    "recovery_imu_usable=%d "
                    "recovery_imu_prior=%d "
                    "recovery_imu_rmse_px=%.9f "
                    "recovery_imu_to_committed_t=%.9f "
                    "recovery_imu_to_committed_r_deg=%.9f "
                    "recovery_imu_to_precommit_t=%.9f "
                    "recovery_imu_to_precommit_r_deg=%.9f "
                    "recovery_visual_imu_usable=%d "
                    "recovery_visual_imu_prior=%d "
                    "recovery_visual_imu_rmse_px=%.9f "
                    "recovery_visual_imu_to_committed_t=%.9f "
                    "recovery_visual_imu_to_committed_r_deg=%.9f "
                    "recovery_visual_imu_to_precommit_t=%.9f "
                    "recovery_visual_imu_to_precommit_r_deg=%.9f "
                    "visual_effect_from_recovery_only_t=%.9f "
                    "visual_effect_from_recovery_only_r_deg=%.9f "
                    "imu_effect_from_recovery_only_t=%.9f "
                    "imu_effect_from_recovery_only_r_deg=%.9f "
                    "combined_effect_from_recovery_only_t=%.9f "
                    "combined_effect_from_recovery_only_r_deg=%.9f "
                    "recovery_imu_v_delta=%.9f "
                    "recovery_imu_bg_delta=%.9f "
                    "recovery_imu_ba_delta=%.9f "
                    "recovery_visual_imu_v_delta=%.9f "
                    "recovery_visual_imu_bg_delta=%.9f "
                    "recovery_visual_imu_ba_delta=%.9f "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    seeded_recovery_only.recovery_factors,
                    seeded_recovery_visual
                        .conventional_visual_factors,
                    seeded_recovery_only.usable ? 1 : 0,
                    seeded_recovery_only.recovery_rmse_px,
                    seeded_to_committed_t(
                        seeded_recovery_only),
                    seeded_to_committed_r(
                        seeded_recovery_only),
                    seeded_to_precommit_t(
                        seeded_recovery_only),
                    seeded_to_precommit_r(
                        seeded_recovery_only),
                    seeded_recovery_visual.usable ? 1 : 0,
                    seeded_recovery_visual.recovery_rmse_px,
                    seeded_to_committed_t(
                        seeded_recovery_visual),
                    seeded_to_committed_r(
                        seeded_recovery_visual),
                    seeded_to_precommit_t(
                        seeded_recovery_visual),
                    seeded_to_precommit_r(
                        seeded_recovery_visual),
                    seeded_recovery_imu.usable ? 1 : 0,
                    seeded_recovery_imu.imu_prior ? 1 : 0,
                    seeded_recovery_imu.recovery_rmse_px,
                    seeded_to_committed_t(
                        seeded_recovery_imu),
                    seeded_to_committed_r(
                        seeded_recovery_imu),
                    seeded_to_precommit_t(
                        seeded_recovery_imu),
                    seeded_to_precommit_r(
                        seeded_recovery_imu),
                    seeded_recovery_visual_imu.usable ? 1 : 0,
                    seeded_recovery_visual_imu.imu_prior ? 1 : 0,
                    seeded_recovery_visual_imu.recovery_rmse_px,
                    seeded_to_committed_t(
                        seeded_recovery_visual_imu),
                    seeded_to_committed_r(
                        seeded_recovery_visual_imu),
                    seeded_to_precommit_t(
                        seeded_recovery_visual_imu),
                    seeded_to_precommit_r(
                        seeded_recovery_visual_imu),
                    visual_effect_from_recovery_only_t,
                    visual_effect_from_recovery_only_r,
                    imu_effect_from_recovery_only_t,
                    imu_effect_from_recovery_only_r,
                    combined_effect_from_recovery_only_t,
                    combined_effect_from_recovery_only_r,
                    seeded_recovery_imu_v_delta,
                    seeded_recovery_imu_bg_delta,
                    seeded_recovery_imu_ba_delta,
                    seeded_recovery_visual_imu_v_delta,
                    seeded_recovery_visual_imu_bg_delta,
                    seeded_recovery_visual_imu_ba_delta);

                // Coherent-trajectory IMU shadow. A global yaw+translation
                // transform is a gauge symmetry of the inertial model because
                // world gravity remains on -Z. A full 3D rotation is not:
                // roll/pitch rotates the local trajectory relative to gravity.
                //
                // Apply the same correction to the previous and current
                // cloned states, including rotating world-frame velocity.
                // Biases remain unchanged because they live in the IMU/body
                // frame. Compare IMU-only and IMU+recovery solves for both
                // the full correction and its gravity-preserving yaw part.
                struct CoherentImuShadowResult {
                    bool usable = false;
                    bool imu_prior = false;
                    size_t recovery_factors = 0;
                    PoseState seed_pose;
                    MotionState seed_motion;
                    PoseState pose;
                    MotionState motion;
                    double recovery_rmse_px =
                        std::numeric_limits<double>::quiet_NaN();
                };

                quaternion full_correction_q =
                    state.committed_body_pose.q *
                    target->pose.q.conjugate();
                full_correction_q.normalize();
                const vector<3> full_correction_p =
                    state.committed_body_pose.p -
                    full_correction_q * target->pose.p;

                const matrix<3> full_correction_R =
                    full_correction_q.matrix();
                const double yaw_correction_rad =
                    std::atan2(
                        full_correction_R(1, 0),
                        full_correction_R(0, 0));
                quaternion yaw_correction_q;
                yaw_correction_q =
                    Eigen::AngleAxisd(
                        yaw_correction_rad,
                        Eigen::Vector3d::UnitZ());
                yaw_correction_q.normalize();
                const vector<3> yaw_correction_p =
                    state.committed_body_pose.p -
                    yaw_correction_q * target->pose.p;

                const auto run_coherent_imu_shadow =
                    [this, target, target_top_index, &state,
                     &recovery_rmse_px](
                        const quaternion &correction_q,
                        const vector<3> &correction_p,
                        bool include_recovery) {
                        CoherentImuShadowResult result;
                        if (!config->has_imu() ||
                            target_top_index == 0 ||
                            !target->image) {
                            return result;
                        }

                        Frame *previous =
                            map->get_frame(target_top_index - 1);
                        if (!previous)
                            return result;

                        std::unique_ptr<Frame> previous_owner =
                            previous->clone();
                        std::unique_ptr<Frame> target_owner =
                            target->clone();
                        Frame *shadow_previous =
                            previous_owner.get();
                        Frame *shadow_target =
                            target_owner.get();

                        const auto transform_state =
                            [&correction_q, &correction_p](
                                Frame *frame) {
                                frame->pose.q =
                                    correction_q *
                                    frame->pose.q;
                                frame->pose.q.normalize();
                                frame->pose.p =
                                    correction_q *
                                        frame->pose.p +
                                    correction_p;
                                frame->motion.v =
                                    correction_q *
                                    frame->motion.v;
                            };

                        transform_state(shadow_previous);
                        transform_state(shadow_target);
                        shadow_target->tag(FT_FIX_POSE) = false;
                        shadow_target->tag(FT_FIX_MOTION) = false;

                        result.seed_pose = shadow_target->pose;
                        result.seed_motion = shadow_target->motion;
                        result.pose = shadow_target->pose;
                        result.motion = shadow_target->motion;

                        PreIntegrator shadow_preintegration =
                            target->keyframe_preintegration;
                        if (!shadow_preintegration.integrate(
                                target->image->t,
                                shadow_previous->motion.bg,
                                shadow_previous->motion.ba,
                                true, true)) {
                            return result;
                        }

                        auto solver = Solver::create();
                        solver->add_frame_states(shadow_target);
                        solver->put_factor(
                            Solver::
                                create_preintegration_prior_factor(
                                    shadow_previous,
                                    shadow_target,
                                    shadow_preintegration));
                        result.imu_prior = true;

                        if (include_recovery) {
                            for (size_t i = 0;
                                 i < state
                                         .recovery_landmarks_world
                                         .size();
                                 ++i) {
                                solver
                                    ->add_learned_world_reprojection(
                                        shadow_target,
                                        state
                                            .recovery_landmarks_world[i],
                                        state
                                            .recovery_observations_pixel[i]);
                                ++result.recovery_factors;
                            }
                        }

                        result.usable = solver->solve();
                        result.pose = shadow_target->pose;
                        result.motion = shadow_target->motion;
                        result.recovery_rmse_px =
                            recovery_rmse_px(
                                shadow_target->pose);
                        return result;
                    };

                const CoherentImuShadowResult full_imu =
                    run_coherent_imu_shadow(
                        full_correction_q,
                        full_correction_p,
                        false);
                const CoherentImuShadowResult full_imu_recovery =
                    run_coherent_imu_shadow(
                        full_correction_q,
                        full_correction_p,
                        true);
                const CoherentImuShadowResult yaw_imu =
                    run_coherent_imu_shadow(
                        yaw_correction_q,
                        yaw_correction_p,
                        false);
                const CoherentImuShadowResult yaw_imu_recovery =
                    run_coherent_imu_shadow(
                        yaw_correction_q,
                        yaw_correction_p,
                        true);

                const auto coherent_seed_to_committed_t =
                    [&state](
                        const CoherentImuShadowResult &result) {
                        return (result.seed_pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto coherent_seed_to_committed_r =
                    [&state](
                        const CoherentImuShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.seed_pose);
                    };
                const auto coherent_final_delta_seed_t =
                    [](const CoherentImuShadowResult &result) {
                        return (result.pose.p -
                                result.seed_pose.p)
                            .norm();
                    };
                const auto coherent_final_delta_seed_r =
                    [](const CoherentImuShadowResult &result) {
                        return camera_rotation_delta_deg(
                            result.seed_pose,
                            result.pose);
                    };
                const auto coherent_final_to_committed_t =
                    [&state](
                        const CoherentImuShadowResult &result) {
                        return (result.pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto coherent_final_to_committed_r =
                    [&state](
                        const CoherentImuShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.pose);
                    };
                const auto coherent_v_delta =
                    [](const CoherentImuShadowResult &result) {
                        return (result.motion.v -
                                result.seed_motion.v)
                            .norm();
                    };
                const auto coherent_bg_delta =
                    [](const CoherentImuShadowResult &result) {
                        return (result.motion.bg -
                                result.seed_motion.bg)
                            .norm();
                    };
                const auto coherent_ba_delta =
                    [](const CoherentImuShadowResult &result) {
                        return (result.motion.ba -
                                result.seed_motion.ba)
                            .norm();
                    };

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryCoherentImuShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f latest_refined_frame=%zu "
                    "topology=coherent_previous_current "
                    "marginalization_prior=0 visual_factors=0 "
                    "full_correction_r_deg=%.9f "
                    "yaw_correction_deg=%.9f "
                    "full_seed_to_committed_t=%.9f "
                    "full_seed_to_committed_r_deg=%.9f "
                    "yaw_seed_to_committed_t=%.9f "
                    "yaw_seed_to_committed_r_deg=%.9f "
                    "full_imu_usable=%d full_imu_prior=%d "
                    "full_imu_delta_seed_t=%.9f "
                    "full_imu_delta_seed_r_deg=%.9f "
                    "full_imu_to_committed_t=%.9f "
                    "full_imu_to_committed_r_deg=%.9f "
                    "full_imu_recovery_usable=%d "
                    "full_imu_recovery_prior=%d "
                    "full_imu_recovery_factors=%zu "
                    "full_imu_recovery_rmse_px=%.9f "
                    "full_imu_recovery_delta_seed_t=%.9f "
                    "full_imu_recovery_delta_seed_r_deg=%.9f "
                    "full_imu_recovery_to_committed_t=%.9f "
                    "full_imu_recovery_to_committed_r_deg=%.9f "
                    "full_imu_recovery_v_delta=%.9f "
                    "full_imu_recovery_bg_delta=%.9f "
                    "full_imu_recovery_ba_delta=%.9f "
                    "yaw_imu_usable=%d yaw_imu_prior=%d "
                    "yaw_imu_delta_seed_t=%.9f "
                    "yaw_imu_delta_seed_r_deg=%.9f "
                    "yaw_imu_to_committed_t=%.9f "
                    "yaw_imu_to_committed_r_deg=%.9f "
                    "yaw_imu_recovery_usable=%d "
                    "yaw_imu_recovery_prior=%d "
                    "yaw_imu_recovery_factors=%zu "
                    "yaw_imu_recovery_rmse_px=%.9f "
                    "yaw_imu_recovery_delta_seed_t=%.9f "
                    "yaw_imu_recovery_delta_seed_r_deg=%.9f "
                    "yaw_imu_recovery_to_committed_t=%.9f "
                    "yaw_imu_recovery_to_committed_r_deg=%.9f "
                    "yaw_imu_recovery_v_delta=%.9f "
                    "yaw_imu_recovery_bg_delta=%.9f "
                    "yaw_imu_recovery_ba_delta=%.9f "
                    "bias_transform=unchanged "
                    "velocity_transform=world_rotated "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    camera_rotation_delta_deg(
                        target->pose,
                        state.committed_body_pose),
                    std::abs(yaw_correction_rad) *
                        180.0 / M_PI,
                    coherent_seed_to_committed_t(full_imu),
                    coherent_seed_to_committed_r(full_imu),
                    coherent_seed_to_committed_t(yaw_imu),
                    coherent_seed_to_committed_r(yaw_imu),
                    full_imu.usable ? 1 : 0,
                    full_imu.imu_prior ? 1 : 0,
                    coherent_final_delta_seed_t(full_imu),
                    coherent_final_delta_seed_r(full_imu),
                    coherent_final_to_committed_t(full_imu),
                    coherent_final_to_committed_r(full_imu),
                    full_imu_recovery.usable ? 1 : 0,
                    full_imu_recovery.imu_prior ? 1 : 0,
                    full_imu_recovery.recovery_factors,
                    full_imu_recovery.recovery_rmse_px,
                    coherent_final_delta_seed_t(
                        full_imu_recovery),
                    coherent_final_delta_seed_r(
                        full_imu_recovery),
                    coherent_final_to_committed_t(
                        full_imu_recovery),
                    coherent_final_to_committed_r(
                        full_imu_recovery),
                    coherent_v_delta(full_imu_recovery),
                    coherent_bg_delta(full_imu_recovery),
                    coherent_ba_delta(full_imu_recovery),
                    yaw_imu.usable ? 1 : 0,
                    yaw_imu.imu_prior ? 1 : 0,
                    coherent_final_delta_seed_t(yaw_imu),
                    coherent_final_delta_seed_r(yaw_imu),
                    coherent_final_to_committed_t(yaw_imu),
                    coherent_final_to_committed_r(yaw_imu),
                    yaw_imu_recovery.usable ? 1 : 0,
                    yaw_imu_recovery.imu_prior ? 1 : 0,
                    yaw_imu_recovery.recovery_factors,
                    yaw_imu_recovery.recovery_rmse_px,
                    coherent_final_delta_seed_t(
                        yaw_imu_recovery),
                    coherent_final_delta_seed_r(
                        yaw_imu_recovery),
                    coherent_final_to_committed_t(
                        yaw_imu_recovery),
                    coherent_final_to_committed_r(
                        yaw_imu_recovery),
                    coherent_v_delta(yaw_imu_recovery),
                    coherent_bg_delta(yaw_imu_recovery),
                    coherent_ba_delta(yaw_imu_recovery));

                // Whole-window coherent IMU shadow. Transform every active
                // top-level frame with the same correction, rotate world-frame
                // velocity, keep body-frame biases unchanged, and reintroduce
                // every adjacent keyframe-preintegration edge. The oldest
                // cloned pose+motion is fixed only to remove gauge freedom.
                // Ordinary visual factors and marginalization remain excluded.
                struct CoherentWindowImuShadowResult {
                    bool usable = false;
                    bool complete = false;
                    size_t window_frames = 0;
                    size_t target_index =
                        static_cast<size_t>(-1);
                    size_t imu_edges = 0;
                    size_t recovery_factors = 0;
                    size_t first_frame_id = 0;
                    size_t last_frame_id = 0;
                    PoseState target_seed_pose;
                    MotionState target_seed_motion;
                    PoseState target_pose;
                    MotionState target_motion;
                    double recovery_rmse_px =
                        std::numeric_limits<double>::quiet_NaN();
                    double max_pose_delta_seed_t = 0.0;
                    double max_pose_delta_seed_r_deg = 0.0;
                    double max_v_delta = 0.0;
                    double max_bg_delta = 0.0;
                    double max_ba_delta = 0.0;
                    double latest_pose_delta_seed_t = 0.0;
                    double latest_pose_delta_seed_r_deg = 0.0;
                };

                const auto run_coherent_window_imu_shadow =
                    [this, target_top_index, &state,
                     &recovery_rmse_px](
                        const quaternion &correction_q,
                        const vector<3> &correction_p,
                        bool include_recovery) {
                        CoherentWindowImuShadowResult result;
                        const size_t window_frames =
                            map->frame_num();
                        result.window_frames = window_frames;
                        result.target_index = target_top_index;

                        if (!config->has_imu() ||
                            window_frames < 2 ||
                            target_top_index == 0 ||
                            target_top_index >= window_frames) {
                            return result;
                        }

                        std::vector<std::unique_ptr<Frame>>
                            shadow_owners;
                        std::vector<Frame *> shadow_frames;
                        std::vector<PoseState> seed_poses;
                        std::vector<MotionState> seed_motions;
                        shadow_owners.reserve(window_frames);
                        shadow_frames.reserve(window_frames);
                        seed_poses.reserve(window_frames);
                        seed_motions.reserve(window_frames);

                        for (size_t i = 0;
                             i < window_frames; ++i) {
                            Frame *source = map->get_frame(i);
                            if (!source || !source->image)
                                return result;

                            std::unique_ptr<Frame> owner =
                                source->clone();
                            Frame *shadow = owner.get();

                            shadow->pose.q =
                                correction_q * shadow->pose.q;
                            shadow->pose.q.normalize();
                            shadow->pose.p =
                                correction_q * shadow->pose.p +
                                correction_p;
                            shadow->motion.v =
                                correction_q * shadow->motion.v;

                            shadow->tag(FT_FIX_POSE) =
                                (i == 0);
                            shadow->tag(FT_FIX_MOTION) =
                                (i == 0);

                            seed_poses.emplace_back(
                                shadow->pose);
                            seed_motions.emplace_back(
                                shadow->motion);
                            shadow_frames.emplace_back(shadow);
                            shadow_owners.emplace_back(
                                std::move(owner));
                        }

                        result.first_frame_id =
                            shadow_frames.front()->id();
                        result.last_frame_id =
                            shadow_frames.back()->id();
                        result.target_seed_pose =
                            shadow_frames[
                                target_top_index]->pose;
                        result.target_seed_motion =
                            shadow_frames[
                                target_top_index]->motion;
                        result.target_pose =
                            result.target_seed_pose;
                        result.target_motion =
                            result.target_seed_motion;

                        auto solver = Solver::create();
                        for (Frame *shadow : shadow_frames)
                            solver->add_frame_states(shadow);

                        // Preintegration factors hold references, so reserve
                        // their final storage before creating any factor.
                        std::vector<PreIntegrator>
                            shadow_preintegrations;
                        shadow_preintegrations.reserve(
                            window_frames - 1);

                        for (size_t i = 1;
                             i < window_frames; ++i) {
                            Frame *source_current =
                                map->get_frame(i);
                            if (!source_current ||
                                !source_current->image) {
                                return result;
                            }

                            shadow_preintegrations.emplace_back(
                                source_current
                                    ->keyframe_preintegration);
                            PreIntegrator &preintegration =
                                shadow_preintegrations.back();
                            Frame *shadow_previous =
                                shadow_frames[i - 1];
                            Frame *shadow_current =
                                shadow_frames[i];

                            if (!preintegration.integrate(
                                    shadow_current->image->t,
                                    shadow_previous->motion.bg,
                                    shadow_previous->motion.ba,
                                    true, true)) {
                                return result;
                            }

                            solver->put_factor(
                                Solver::
                                    create_preintegration_error_factor(
                                        shadow_previous,
                                        shadow_current,
                                        preintegration));
                            ++result.imu_edges;
                        }

                        Frame *shadow_target =
                            shadow_frames[
                                target_top_index];
                        if (include_recovery) {
                            for (size_t i = 0;
                                 i < state
                                         .recovery_landmarks_world
                                         .size();
                                 ++i) {
                                solver
                                    ->add_learned_world_reprojection(
                                        shadow_target,
                                        state
                                            .recovery_landmarks_world[i],
                                        state
                                            .recovery_observations_pixel[i]);
                                ++result.recovery_factors;
                            }
                        }

                        result.complete =
                            result.imu_edges + 1 ==
                            window_frames;
                        result.usable = solver->solve();
                        result.target_pose =
                            shadow_target->pose;
                        result.target_motion =
                            shadow_target->motion;
                        result.recovery_rmse_px =
                            recovery_rmse_px(
                                shadow_target->pose);

                        for (size_t i = 0;
                             i < window_frames; ++i) {
                            result.max_pose_delta_seed_t =
                                std::max(
                                    result
                                        .max_pose_delta_seed_t,
                                    (shadow_frames[i]->pose.p -
                                     seed_poses[i].p)
                                        .norm());
                            result.max_pose_delta_seed_r_deg =
                                std::max(
                                    result
                                        .max_pose_delta_seed_r_deg,
                                    camera_rotation_delta_deg(
                                        seed_poses[i],
                                        shadow_frames[i]->pose));
                            result.max_v_delta =
                                std::max(
                                    result.max_v_delta,
                                    (shadow_frames[i]->motion.v -
                                     seed_motions[i].v)
                                        .norm());
                            result.max_bg_delta =
                                std::max(
                                    result.max_bg_delta,
                                    (shadow_frames[i]->motion.bg -
                                     seed_motions[i].bg)
                                        .norm());
                            result.max_ba_delta =
                                std::max(
                                    result.max_ba_delta,
                                    (shadow_frames[i]->motion.ba -
                                     seed_motions[i].ba)
                                        .norm());
                        }

                        result.latest_pose_delta_seed_t =
                            (shadow_frames.back()->pose.p -
                             seed_poses.back().p)
                                .norm();
                        result.latest_pose_delta_seed_r_deg =
                            camera_rotation_delta_deg(
                                seed_poses.back(),
                                shadow_frames.back()->pose);
                        return result;
                    };

                const CoherentWindowImuShadowResult
                    full_window_imu =
                        run_coherent_window_imu_shadow(
                            full_correction_q,
                            full_correction_p,
                            false);
                const CoherentWindowImuShadowResult
                    full_window_imu_recovery =
                        run_coherent_window_imu_shadow(
                            full_correction_q,
                            full_correction_p,
                            true);
                const CoherentWindowImuShadowResult
                    yaw_window_imu =
                        run_coherent_window_imu_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            false);
                const CoherentWindowImuShadowResult
                    yaw_window_imu_recovery =
                        run_coherent_window_imu_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            true);

                const auto window_target_delta_seed_t =
                    [](const CoherentWindowImuShadowResult &result) {
                        return (result.target_pose.p -
                                result.target_seed_pose.p)
                            .norm();
                    };
                const auto window_target_delta_seed_r =
                    [](const CoherentWindowImuShadowResult &result) {
                        return camera_rotation_delta_deg(
                            result.target_seed_pose,
                            result.target_pose);
                    };
                const auto window_target_to_committed_t =
                    [&state](
                        const CoherentWindowImuShadowResult &result) {
                        return (result.target_pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto window_target_to_committed_r =
                    [&state](
                        const CoherentWindowImuShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.target_pose);
                    };
                const auto window_target_v_delta =
                    [](const CoherentWindowImuShadowResult &result) {
                        return (result.target_motion.v -
                                result.target_seed_motion.v)
                            .norm();
                    };

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryCoherentWindowImuShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f latest_refined_frame=%zu "
                    "topology=coherent_active_top_level_window "
                    "oldest_anchor=pose_motion_fixed "
                    "marginalization_prior=0 visual_factors=0 "
                    "window_frames=%zu target_window_index=%zu "
                    "first_frame=%zu last_frame=%zu "
                    "full_correction_r_deg=%.9f "
                    "yaw_correction_deg=%.9f "
                    "full_imu_usable=%d full_imu_complete=%d "
                    "full_imu_edges=%zu "
                    "full_imu_target_delta_seed_t=%.9f "
                    "full_imu_target_delta_seed_r_deg=%.9f "
                    "full_imu_max_delta_seed_t=%.9f "
                    "full_imu_max_delta_seed_r_deg=%.9f "
                    "full_imu_latest_delta_seed_t=%.9f "
                    "full_imu_latest_delta_seed_r_deg=%.9f "
                    "full_imu_target_to_committed_t=%.9f "
                    "full_imu_target_to_committed_r_deg=%.9f "
                    "full_imu_recovery_usable=%d "
                    "full_imu_recovery_complete=%d "
                    "full_imu_recovery_edges=%zu "
                    "full_imu_recovery_factors=%zu "
                    "full_imu_recovery_rmse_px=%.9f "
                    "full_imu_recovery_target_delta_seed_t=%.9f "
                    "full_imu_recovery_target_delta_seed_r_deg=%.9f "
                    "full_imu_recovery_max_delta_seed_t=%.9f "
                    "full_imu_recovery_max_delta_seed_r_deg=%.9f "
                    "full_imu_recovery_target_to_committed_t=%.9f "
                    "full_imu_recovery_target_to_committed_r_deg=%.9f "
                    "full_imu_recovery_target_v_delta=%.9f "
                    "full_imu_recovery_max_v_delta=%.9f "
                    "full_imu_recovery_max_bg_delta=%.9f "
                    "full_imu_recovery_max_ba_delta=%.9f "
                    "yaw_imu_usable=%d yaw_imu_complete=%d "
                    "yaw_imu_edges=%zu "
                    "yaw_imu_target_delta_seed_t=%.9f "
                    "yaw_imu_target_delta_seed_r_deg=%.9f "
                    "yaw_imu_max_delta_seed_t=%.9f "
                    "yaw_imu_max_delta_seed_r_deg=%.9f "
                    "yaw_imu_latest_delta_seed_t=%.9f "
                    "yaw_imu_latest_delta_seed_r_deg=%.9f "
                    "yaw_imu_target_to_committed_t=%.9f "
                    "yaw_imu_target_to_committed_r_deg=%.9f "
                    "yaw_imu_recovery_usable=%d "
                    "yaw_imu_recovery_complete=%d "
                    "yaw_imu_recovery_edges=%zu "
                    "yaw_imu_recovery_factors=%zu "
                    "yaw_imu_recovery_rmse_px=%.9f "
                    "yaw_imu_recovery_target_delta_seed_t=%.9f "
                    "yaw_imu_recovery_target_delta_seed_r_deg=%.9f "
                    "yaw_imu_recovery_max_delta_seed_t=%.9f "
                    "yaw_imu_recovery_max_delta_seed_r_deg=%.9f "
                    "yaw_imu_recovery_target_to_committed_t=%.9f "
                    "yaw_imu_recovery_target_to_committed_r_deg=%.9f "
                    "yaw_imu_recovery_target_v_delta=%.9f "
                    "yaw_imu_recovery_max_v_delta=%.9f "
                    "yaw_imu_recovery_max_bg_delta=%.9f "
                    "yaw_imu_recovery_max_ba_delta=%.9f "
                    "bias_transform=unchanged "
                    "velocity_transform=world_rotated "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    full_window_imu.window_frames,
                    full_window_imu.target_index,
                    full_window_imu.first_frame_id,
                    full_window_imu.last_frame_id,
                    camera_rotation_delta_deg(
                        target->pose,
                        state.committed_body_pose),
                    std::abs(yaw_correction_rad) *
                        180.0 / M_PI,
                    full_window_imu.usable ? 1 : 0,
                    full_window_imu.complete ? 1 : 0,
                    full_window_imu.imu_edges,
                    window_target_delta_seed_t(
                        full_window_imu),
                    window_target_delta_seed_r(
                        full_window_imu),
                    full_window_imu.max_pose_delta_seed_t,
                    full_window_imu
                        .max_pose_delta_seed_r_deg,
                    full_window_imu
                        .latest_pose_delta_seed_t,
                    full_window_imu
                        .latest_pose_delta_seed_r_deg,
                    window_target_to_committed_t(
                        full_window_imu),
                    window_target_to_committed_r(
                        full_window_imu),
                    full_window_imu_recovery.usable ? 1 : 0,
                    full_window_imu_recovery.complete ? 1 : 0,
                    full_window_imu_recovery.imu_edges,
                    full_window_imu_recovery.recovery_factors,
                    full_window_imu_recovery.recovery_rmse_px,
                    window_target_delta_seed_t(
                        full_window_imu_recovery),
                    window_target_delta_seed_r(
                        full_window_imu_recovery),
                    full_window_imu_recovery
                        .max_pose_delta_seed_t,
                    full_window_imu_recovery
                        .max_pose_delta_seed_r_deg,
                    window_target_to_committed_t(
                        full_window_imu_recovery),
                    window_target_to_committed_r(
                        full_window_imu_recovery),
                    window_target_v_delta(
                        full_window_imu_recovery),
                    full_window_imu_recovery.max_v_delta,
                    full_window_imu_recovery.max_bg_delta,
                    full_window_imu_recovery.max_ba_delta,
                    yaw_window_imu.usable ? 1 : 0,
                    yaw_window_imu.complete ? 1 : 0,
                    yaw_window_imu.imu_edges,
                    window_target_delta_seed_t(
                        yaw_window_imu),
                    window_target_delta_seed_r(
                        yaw_window_imu),
                    yaw_window_imu.max_pose_delta_seed_t,
                    yaw_window_imu.max_pose_delta_seed_r_deg,
                    yaw_window_imu.latest_pose_delta_seed_t,
                    yaw_window_imu
                        .latest_pose_delta_seed_r_deg,
                    window_target_to_committed_t(
                        yaw_window_imu),
                    window_target_to_committed_r(
                        yaw_window_imu),
                    yaw_window_imu_recovery.usable ? 1 : 0,
                    yaw_window_imu_recovery.complete ? 1 : 0,
                    yaw_window_imu_recovery.imu_edges,
                    yaw_window_imu_recovery.recovery_factors,
                    yaw_window_imu_recovery.recovery_rmse_px,
                    window_target_delta_seed_t(
                        yaw_window_imu_recovery),
                    window_target_delta_seed_r(
                        yaw_window_imu_recovery),
                    yaw_window_imu_recovery
                        .max_pose_delta_seed_t,
                    yaw_window_imu_recovery
                        .max_pose_delta_seed_r_deg,
                    window_target_to_committed_t(
                        yaw_window_imu_recovery),
                    window_target_to_committed_r(
                        yaw_window_imu_recovery),
                    window_target_v_delta(
                        yaw_window_imu_recovery),
                    yaw_window_imu_recovery.max_v_delta,
                    yaw_window_imu_recovery.max_bg_delta,
                    yaw_window_imu_recovery.max_ba_delta);

                // Marginalization compatibility shadow. Unlike the 0110g
                // IMU-only experiment, do not add a synthetic gauge anchor.
                // Instead bind the actual live marginalization factor to
                // cloned frame storage, preserving its exact stored
                // linearization point and information while keeping live
                // estimator state untouched.
                struct MarginalizationShadowResult {
                    bool usable = false;
                    bool prior_bound = false;
                    bool prior_rebased = false;
                    bool complete = false;
                    size_t window_frames = 0;
                    size_t prior_frames = 0;
                    size_t target_index =
                        static_cast<size_t>(-1);
                    size_t imu_edges = 0;
                    size_t recovery_factors = 0;
                    PoseState target_seed_pose;
                    MotionState target_seed_motion;
                    PoseState target_pose;
                    MotionState target_motion;
                    double recovery_rmse_px =
                        std::numeric_limits<double>::quiet_NaN();
                    double max_pose_delta_seed_t = 0.0;
                    double max_pose_delta_seed_r_deg = 0.0;
                    double max_v_delta = 0.0;
                };

                const auto run_marginalization_shadow =
                    [this, target_top_index, &state,
                     &recovery_rmse_px](
                        const quaternion &correction_q,
                        const vector<3> &correction_p,
                        bool rebase_prior,
                        bool include_recovery) {
                        MarginalizationShadowResult result;
                        result.window_frames = map->frame_num();
                        result.target_index = target_top_index;

                        if (!config->has_imu() ||
                            !map->marginalization_factor ||
                            result.window_frames < 2 ||
                            target_top_index >=
                                result.window_frames) {
                            return result;
                        }

                        std::vector<std::unique_ptr<Frame>>
                            shadow_owners;
                        std::vector<Frame *> shadow_frames;
                        std::vector<PoseState> seed_poses;
                        std::vector<MotionState> seed_motions;
                        shadow_owners.reserve(
                            result.window_frames);
                        shadow_frames.reserve(
                            result.window_frames);
                        seed_poses.reserve(
                            result.window_frames);
                        seed_motions.reserve(
                            result.window_frames);

                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            Frame *source = map->get_frame(i);
                            if (!source || !source->image)
                                return result;

                            std::unique_ptr<Frame> owner =
                                source->clone();
                            Frame *shadow = owner.get();

                            shadow->pose.q =
                                correction_q * shadow->pose.q;
                            shadow->pose.q.normalize();
                            shadow->pose.p =
                                correction_q * shadow->pose.p +
                                correction_p;
                            shadow->motion.v =
                                correction_q * shadow->motion.v;

                            // The real marginalization factor supplies the
                            // gauge anchor in this experiment.
                            shadow->tag(FT_FIX_POSE) = false;
                            shadow->tag(FT_FIX_MOTION) = false;

                            seed_poses.emplace_back(
                                shadow->pose);
                            seed_motions.emplace_back(
                                shadow->motion);
                            shadow_frames.emplace_back(shadow);
                            shadow_owners.emplace_back(
                                std::move(owner));
                        }

                        Frame *shadow_target =
                            shadow_frames[target_top_index];
                        result.target_seed_pose =
                            shadow_target->pose;
                        result.target_seed_motion =
                            shadow_target->motion;
                        result.target_pose =
                            result.target_seed_pose;
                        result.target_motion =
                            result.target_seed_motion;

                        auto solver = Solver::create();
                        for (Frame *shadow : shadow_frames)
                            solver->add_frame_states(shadow);

                        std::vector<PreIntegrator>
                            shadow_preintegrations;
                        shadow_preintegrations.reserve(
                            result.window_frames - 1);

                        for (size_t i = 1;
                             i < result.window_frames;
                             ++i) {
                            Frame *source_current =
                                map->get_frame(i);
                            if (!source_current ||
                                !source_current->image) {
                                return result;
                            }

                            shadow_preintegrations.emplace_back(
                                source_current
                                    ->keyframe_preintegration);
                            PreIntegrator &preintegration =
                                shadow_preintegrations.back();
                            Frame *shadow_previous =
                                shadow_frames[i - 1];
                            Frame *shadow_current =
                                shadow_frames[i];

                            if (!preintegration.integrate(
                                    shadow_current->image->t,
                                    shadow_previous->motion.bg,
                                    shadow_previous->motion.ba,
                                    true, true)) {
                                return result;
                            }

                            solver->put_factor(
                                Solver::
                                    create_preintegration_error_factor(
                                        shadow_previous,
                                        shadow_current,
                                        preintegration));
                            ++result.imu_edges;
                        }

                        const auto &prior_sources =
                            map->marginalization_factor
                                ->linearization_frames();
                        result.prior_frames =
                            prior_sources.size();

                        std::vector<Frame *>
                            shadow_prior_frames;
                        shadow_prior_frames.reserve(
                            prior_sources.size());

                        for (Frame *prior_source :
                             prior_sources) {
                            size_t source_index = nil();
                            for (size_t i = 0;
                                 i < result.window_frames;
                                 ++i) {
                                if (map->get_frame(i) ==
                                    prior_source) {
                                    source_index = i;
                                    break;
                                }
                            }
                            if (source_index == nil())
                                return result;
                            shadow_prior_frames.emplace_back(
                                shadow_frames[source_index]);
                        }

                        std::unique_ptr<MarginalizationFactor>
                            rebased_prior;
                        MarginalizationFactor *prior_factor =
                            map->marginalization_factor.get();
                        if (rebase_prior) {
                            rebased_prior =
                                map->marginalization_factor
                                    ->clone_rebased_world(
                                        shadow_prior_frames,
                                        correction_q,
                                        correction_p);
                            if (!rebased_prior)
                                return result;
                            prior_factor =
                                rebased_prior.get();
                            result.prior_rebased = true;
                        }

                        result.prior_bound =
                            solver
                                ->add_marginalization_factor_for_frames(
                                    prior_factor,
                                    shadow_prior_frames);
                        if (!result.prior_bound)
                            return result;

                        if (include_recovery) {
                            for (size_t i = 0;
                                 i < state
                                         .recovery_landmarks_world
                                         .size();
                                 ++i) {
                                solver
                                    ->add_learned_world_reprojection(
                                        shadow_target,
                                        state
                                            .recovery_landmarks_world[i],
                                        state
                                            .recovery_observations_pixel[i]);
                                ++result.recovery_factors;
                            }
                        }

                        result.complete =
                            result.imu_edges + 1 ==
                                result.window_frames &&
                            result.prior_frames > 0;
                        result.usable = solver->solve();
                        result.target_pose =
                            shadow_target->pose;
                        result.target_motion =
                            shadow_target->motion;
                        result.recovery_rmse_px =
                            recovery_rmse_px(
                                shadow_target->pose);

                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            result.max_pose_delta_seed_t =
                                std::max(
                                    result
                                        .max_pose_delta_seed_t,
                                    (shadow_frames[i]->pose.p -
                                     seed_poses[i].p)
                                        .norm());
                            result.max_pose_delta_seed_r_deg =
                                std::max(
                                    result
                                        .max_pose_delta_seed_r_deg,
                                    camera_rotation_delta_deg(
                                        seed_poses[i],
                                        shadow_frames[i]->pose));
                            result.max_v_delta =
                                std::max(
                                    result.max_v_delta,
                                    (shadow_frames[i]->motion.v -
                                     seed_motions[i].v)
                                        .norm());
                        }
                        return result;
                    };

                quaternion identity_correction_q;
                identity_correction_q.setIdentity();
                vector<3> identity_correction_p;
                identity_correction_p.setZero();

                const MarginalizationShadowResult
                    identity_marginalization =
                        run_marginalization_shadow(
                            identity_correction_q,
                            identity_correction_p,
                            false,
                            false);
                const MarginalizationShadowResult
                    yaw_marginalization =
                        run_marginalization_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            false,
                            false);
                const MarginalizationShadowResult
                    yaw_marginalization_recovery =
                        run_marginalization_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            false,
                            true);
                const MarginalizationShadowResult
                    yaw_rebased_marginalization =
                        run_marginalization_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            true,
                            false);
                const MarginalizationShadowResult
                    yaw_rebased_marginalization_recovery =
                        run_marginalization_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            true,
                            true);

                const auto marginalization_target_delta_seed_t =
                    [](const MarginalizationShadowResult &result) {
                        return (result.target_pose.p -
                                result.target_seed_pose.p)
                            .norm();
                    };
                const auto marginalization_target_delta_seed_r =
                    [](const MarginalizationShadowResult &result) {
                        return camera_rotation_delta_deg(
                            result.target_seed_pose,
                            result.target_pose);
                    };
                const auto marginalization_target_to_committed_t =
                    [&state](
                        const MarginalizationShadowResult &result) {
                        return (result.target_pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto marginalization_target_to_committed_r =
                    [&state](
                        const MarginalizationShadowResult &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.target_pose);
                    };

                const double yaw_final_to_identity_final_t =
                    (yaw_marginalization.target_pose.p -
                     identity_marginalization.target_pose.p)
                        .norm();
                const double yaw_final_to_identity_final_r =
                    camera_rotation_delta_deg(
                        identity_marginalization.target_pose,
                        yaw_marginalization.target_pose);

                PoseState transformed_identity_target_pose;
                transformed_identity_target_pose.q =
                    yaw_correction_q *
                    identity_marginalization.target_pose.q;
                transformed_identity_target_pose.q.normalize();
                transformed_identity_target_pose.p =
                    yaw_correction_q *
                        identity_marginalization.target_pose.p +
                    yaw_correction_p;

                MotionState transformed_identity_target_motion =
                    identity_marginalization.target_motion;
                transformed_identity_target_motion.v =
                    yaw_correction_q *
                    identity_marginalization.target_motion.v;

                const double
                    rebased_to_transformed_identity_t =
                        (yaw_rebased_marginalization
                             .target_pose.p -
                         transformed_identity_target_pose.p)
                            .norm();
                const double
                    rebased_to_transformed_identity_r =
                        camera_rotation_delta_deg(
                            transformed_identity_target_pose,
                            yaw_rebased_marginalization
                                .target_pose);
                const double
                    rebased_to_transformed_identity_v =
                        (yaw_rebased_marginalization
                             .target_motion.v -
                         transformed_identity_target_motion.v)
                            .norm();
                const double
                    rebased_to_transformed_identity_bg =
                        (yaw_rebased_marginalization
                             .target_motion.bg -
                         transformed_identity_target_motion.bg)
                            .norm();
                const double
                    rebased_to_transformed_identity_ba =
                        (yaw_rebased_marginalization
                             .target_motion.ba -
                         transformed_identity_target_motion.ba)
                            .norm();

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryMarginalizationRebaseShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f latest_refined_frame=%zu "
                    "topology=active_top_level_window "
                    "visual_factors=0 "
                    "window_frames=%zu prior_frames=%zu "
                    "target_window_index=%zu "
                    "yaw_correction_deg=%.9f "
                    "rebase_rule=left_yaw_translation_"
                    "linpoint_qpv_sqrtinfo_pv "
                    "identity_target_delta_seed_t=%.9f "
                    "identity_target_delta_seed_r_deg=%.9f "
                    "original_yaw_target_delta_seed_t=%.9f "
                    "original_yaw_target_delta_seed_r_deg=%.9f "
                    "rebased_usable=%d "
                    "rebased_prior_bound=%d "
                    "rebased_prior_rebased=%d "
                    "rebased_complete=%d "
                    "rebased_imu_edges=%zu "
                    "rebased_target_delta_seed_t=%.9f "
                    "rebased_target_delta_seed_r_deg=%.9f "
                    "rebased_max_delta_seed_t=%.9f "
                    "rebased_max_delta_seed_r_deg=%.9f "
                    "rebased_max_v_delta=%.9f "
                    "rebased_target_to_committed_t=%.9f "
                    "rebased_target_to_committed_r_deg=%.9f "
                    "rebased_to_transformed_identity_t=%.9f "
                    "rebased_to_transformed_identity_r_deg=%.9f "
                    "rebased_to_transformed_identity_v=%.9f "
                    "rebased_to_transformed_identity_bg=%.9f "
                    "rebased_to_transformed_identity_ba=%.9f "
                    "rebased_recovery_usable=%d "
                    "rebased_recovery_prior_bound=%d "
                    "rebased_recovery_prior_rebased=%d "
                    "rebased_recovery_complete=%d "
                    "rebased_recovery_imu_edges=%zu "
                    "rebased_recovery_factors=%zu "
                    "rebased_recovery_rmse_px=%.9f "
                    "rebased_recovery_target_delta_seed_t=%.9f "
                    "rebased_recovery_target_delta_seed_r_deg=%.9f "
                    "rebased_recovery_max_delta_seed_t=%.9f "
                    "rebased_recovery_max_delta_seed_r_deg=%.9f "
                    "rebased_recovery_max_v_delta=%.9f "
                    "rebased_recovery_target_to_committed_t=%.9f "
                    "rebased_recovery_target_to_committed_r_deg=%.9f "
                    "infovec_transform=unchanged "
                    "marginalization_source=live_clone_rebased "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    yaw_rebased_marginalization.window_frames,
                    yaw_rebased_marginalization.prior_frames,
                    yaw_rebased_marginalization.target_index,
                    std::abs(yaw_correction_rad) *
                        180.0 / M_PI,
                    marginalization_target_delta_seed_t(
                        identity_marginalization),
                    marginalization_target_delta_seed_r(
                        identity_marginalization),
                    marginalization_target_delta_seed_t(
                        yaw_marginalization),
                    marginalization_target_delta_seed_r(
                        yaw_marginalization),
                    yaw_rebased_marginalization.usable ? 1 : 0,
                    yaw_rebased_marginalization.prior_bound
                        ? 1
                        : 0,
                    yaw_rebased_marginalization.prior_rebased
                        ? 1
                        : 0,
                    yaw_rebased_marginalization.complete
                        ? 1
                        : 0,
                    yaw_rebased_marginalization.imu_edges,
                    marginalization_target_delta_seed_t(
                        yaw_rebased_marginalization),
                    marginalization_target_delta_seed_r(
                        yaw_rebased_marginalization),
                    yaw_rebased_marginalization
                        .max_pose_delta_seed_t,
                    yaw_rebased_marginalization
                        .max_pose_delta_seed_r_deg,
                    yaw_rebased_marginalization.max_v_delta,
                    marginalization_target_to_committed_t(
                        yaw_rebased_marginalization),
                    marginalization_target_to_committed_r(
                        yaw_rebased_marginalization),
                    rebased_to_transformed_identity_t,
                    rebased_to_transformed_identity_r,
                    rebased_to_transformed_identity_v,
                    rebased_to_transformed_identity_bg,
                    rebased_to_transformed_identity_ba,
                    yaw_rebased_marginalization_recovery
                            .usable
                        ? 1
                        : 0,
                    yaw_rebased_marginalization_recovery
                            .prior_bound
                        ? 1
                        : 0,
                    yaw_rebased_marginalization_recovery
                            .prior_rebased
                        ? 1
                        : 0,
                    yaw_rebased_marginalization_recovery
                            .complete
                        ? 1
                        : 0,
                    yaw_rebased_marginalization_recovery
                        .imu_edges,
                    yaw_rebased_marginalization_recovery
                        .recovery_factors,
                    yaw_rebased_marginalization_recovery
                        .recovery_rmse_px,
                    marginalization_target_delta_seed_t(
                        yaw_rebased_marginalization_recovery),
                    marginalization_target_delta_seed_r(
                        yaw_rebased_marginalization_recovery),
                    yaw_rebased_marginalization_recovery
                        .max_pose_delta_seed_t,
                    yaw_rebased_marginalization_recovery
                        .max_pose_delta_seed_r_deg,
                    yaw_rebased_marginalization_recovery
                        .max_v_delta,
                    marginalization_target_to_committed_t(
                        yaw_rebased_marginalization_recovery),
                    marginalization_target_to_committed_r(
                        yaw_rebased_marginalization_recovery));

                // 0110j: reproduce the ordinary full-window visual
                // topology on clone-owned frame states and inverse-depth
                // scalars. This adds the same valid/static keyframe-anchored
                // track states, ordinary Cauchy(1.0) reprojection factors,
                // complete top-level IMU chain, and marginalization prior as
                // refine_window(), while leaving all live Frame/Track state
                // untouched.
                struct FullWindowVisualShadowResult {
                    bool usable = false;
                    bool prior_bound = false;
                    bool prior_rebased = false;
                    bool complete = false;
                    size_t window_frames = 0;
                    size_t prior_frames = 0;
                    size_t target_index =
                        static_cast<size_t>(-1);
                    size_t imu_edges = 0;
                    size_t visual_track_states = 0;
                    size_t visual_factors = 0;
                    size_t visual_fixed_inv_depth = 0;
                    size_t visual_depth_priors = 0;
                    size_t visual_mapping_failures = 0;
                    size_t recovery_factors = 0;
                    PoseState target_seed_pose;
                    MotionState target_seed_motion;
                    PoseState target_pose;
                    MotionState target_motion;
                    double recovery_rmse_px =
                        std::numeric_limits<double>::quiet_NaN();
                    double max_pose_delta_seed_t = 0.0;
                    double max_pose_delta_seed_r_deg = 0.0;
                    double max_v_delta = 0.0;
                    double max_bg_delta = 0.0;
                    double max_ba_delta = 0.0;
                    double max_inv_depth_delta = 0.0;
                    std::vector<double> final_inverse_depths;
                };

                const auto run_full_window_visual_shadow =
                    [this, target_top_index, &state,
                     &recovery_rmse_px](
                        const quaternion &correction_q,
                        const vector<3> &correction_p,
                        bool rebase_prior,
                        bool include_recovery) {
                        FullWindowVisualShadowResult result;
                        result.window_frames =
                            map->frame_num();
                        result.target_index =
                            target_top_index;

                        if (!config->has_imu() ||
                            !map->marginalization_factor ||
                            result.window_frames < 2 ||
                            target_top_index >=
                                result.window_frames) {
                            return result;
                        }

                        std::vector<std::unique_ptr<Frame>>
                            shadow_owners;
                        std::vector<Frame *> shadow_frames;
                        std::vector<PoseState> seed_poses;
                        std::vector<MotionState> seed_motions;
                        std::unordered_map<Frame *, Frame *>
                            shadow_by_source;
                        shadow_owners.reserve(
                            result.window_frames);
                        shadow_frames.reserve(
                            result.window_frames);
                        seed_poses.reserve(
                            result.window_frames);
                        seed_motions.reserve(
                            result.window_frames);
                        shadow_by_source.reserve(
                            result.window_frames);

                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            Frame *source =
                                map->get_frame(i);
                            if (!source || !source->image)
                                return result;

                            std::unique_ptr<Frame> owner =
                                source->clone();
                            Frame *shadow = owner.get();

                            shadow->pose.q =
                                correction_q *
                                shadow->pose.q;
                            shadow->pose.q.normalize();
                            shadow->pose.p =
                                correction_q *
                                    shadow->pose.p +
                                correction_p;
                            shadow->motion.v =
                                correction_q *
                                shadow->motion.v;
                            shadow->tag(FT_FIX_POSE) = false;
                            shadow->tag(FT_FIX_MOTION) = false;

                            seed_poses.emplace_back(
                                shadow->pose);
                            seed_motions.emplace_back(
                                shadow->motion);
                            shadow_by_source[source] = shadow;
                            shadow_frames.emplace_back(shadow);
                            shadow_owners.emplace_back(
                                std::move(owner));
                        }

                        Frame *shadow_target =
                            shadow_frames[
                                target_top_index];
                        result.target_seed_pose =
                            shadow_target->pose;
                        result.target_seed_motion =
                            shadow_target->motion;
                        result.target_pose =
                            result.target_seed_pose;
                        result.target_motion =
                            result.target_seed_motion;

                        auto solver = Solver::create();
                        for (Frame *shadow :
                             shadow_frames) {
                            solver->add_frame_states(
                                shadow);
                        }

                        // Match refine_window()'s track-state eligibility
                        // exactly, but place each inverse depth in independent
                        // shadow-owned storage rather than allocating Track.
                        std::unordered_set<Track *>
                            visited_tracks;
                        std::vector<Track *>
                            source_tracks;
                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            Frame *source =
                                map->get_frame(i);
                            for (size_t k = 0;
                                 k < source->keypoint_num();
                                 ++k) {
                                Track *track =
                                    source->get_track(k);
                                if (!track ||
                                    visited_tracks.count(
                                        track) > 0) {
                                    continue;
                                }
                                visited_tracks.insert(track);

                                if (!track->tag(TT_VALID) ||
                                    !track->tag(TT_STATIC) ||
                                    !track->first_frame()
                                         ->tag(FT_KEYFRAME)) {
                                    continue;
                                }
                                source_tracks.emplace_back(
                                    track);
                            }
                        }

                        std::vector<std::unique_ptr<double>>
                            shadow_inverse_depth_owners;
                        std::vector<Track *>
                            shadow_inverse_depth_sources;
                        std::unordered_map<Track *, double *>
                            shadow_inverse_depths;
                        shadow_inverse_depth_owners.reserve(
                            source_tracks.size());
                        shadow_inverse_depth_sources.reserve(
                            source_tracks.size());
                        shadow_inverse_depths.reserve(
                            source_tracks.size());

                        for (Track *track :
                             source_tracks) {
                            auto inv_depth =
                                std::make_unique<double>(
                                    track->landmark
                                        .inv_depth);
                            double *value =
                                inv_depth.get();
                            if (!solver
                                     ->add_shadow_track_state(
                                         value, track)) {
                                ++result
                                      .visual_mapping_failures;
                                continue;
                            }

                            if (track->tag(TT_FIX_INVD)) {
                                ++result
                                      .visual_fixed_inv_depth;
                            } else if (
                                track->has_depth_prior) {
                                ++result
                                      .visual_depth_priors;
                            }

                            shadow_inverse_depths[
                                track] = value;
                            shadow_inverse_depth_owners
                                .emplace_back(
                                    std::move(inv_depth));
                            shadow_inverse_depth_sources
                                .emplace_back(track);
                            ++result
                                  .visual_track_states;
                        }

                        // Match refine_window()'s ordinary reprojection-factor
                        // eligibility and use the same Cauchy(1.0) loss.
                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            Frame *source =
                                map->get_frame(i);
                            Frame *shadow =
                                shadow_frames[i];

                            for (size_t k = 0;
                                 k < source->keypoint_num();
                                 ++k) {
                                Track *track =
                                    source->get_track(k);
                                if (!track)
                                    continue;
                                if (!track->all_tagged(
                                        TT_VALID,
                                        TT_TRIANGULATED,
                                        TT_STATIC)) {
                                    continue;
                                }

                                Frame *source_reference =
                                    track->first_frame();
                                if (!source_reference
                                         ->tag(FT_KEYFRAME)) {
                                    continue;
                                }
                                if (source ==
                                    source_reference) {
                                    continue;
                                }

                                const auto
                                    inverse_depth_it =
                                        shadow_inverse_depths
                                            .find(track);
                                const auto reference_it =
                                    shadow_by_source.find(
                                        source_reference);
                                const size_t
                                    reference_keypoint_index =
                                        track
                                            ->get_keypoint_index(
                                                source_reference);

                                if (inverse_depth_it ==
                                        shadow_inverse_depths
                                            .end() ||
                                    reference_it ==
                                        shadow_by_source.end() ||
                                    reference_keypoint_index ==
                                        nil()) {
                                    ++result
                                          .visual_mapping_failures;
                                    continue;
                                }

                                if (solver
                                        ->add_shadow_reprojection(
                                            shadow, k,
                                            reference_it
                                                ->second,
                                            reference_keypoint_index,
                                            inverse_depth_it
                                                ->second)) {
                                    ++result
                                          .visual_factors;
                                } else {
                                    ++result
                                          .visual_mapping_failures;
                                }
                            }
                        }

                        const auto &prior_sources =
                            map->marginalization_factor
                                ->linearization_frames();
                        result.prior_frames =
                            prior_sources.size();

                        std::vector<Frame *>
                            shadow_prior_frames;
                        shadow_prior_frames.reserve(
                            prior_sources.size());
                        for (Frame *prior_source :
                             prior_sources) {
                            const auto it =
                                shadow_by_source.find(
                                    prior_source);
                            if (it ==
                                shadow_by_source.end()) {
                                ++result
                                      .visual_mapping_failures;
                                return result;
                            }
                            shadow_prior_frames
                                .emplace_back(it->second);
                        }

                        std::unique_ptr<MarginalizationFactor>
                            rebased_prior;
                        MarginalizationFactor *prior_factor =
                            map->marginalization_factor.get();
                        if (rebase_prior) {
                            rebased_prior =
                                map->marginalization_factor
                                    ->clone_rebased_world(
                                        shadow_prior_frames,
                                        correction_q,
                                        correction_p);
                            if (!rebased_prior)
                                return result;
                            prior_factor =
                                rebased_prior.get();
                            result.prior_rebased = true;
                        }

                        result.prior_bound =
                            solver
                                ->add_marginalization_factor_for_frames(
                                    prior_factor,
                                    shadow_prior_frames);
                        if (!result.prior_bound)
                            return result;

                        // Preintegration factors hold PreIntegrator references;
                        // reserve final storage before constructing factors.
                        std::vector<PreIntegrator>
                            shadow_preintegrations;
                        shadow_preintegrations.reserve(
                            result.window_frames - 1);
                        for (size_t i = 1;
                             i < result.window_frames;
                             ++i) {
                            Frame *source_current =
                                map->get_frame(i);
                            Frame *shadow_previous =
                                shadow_frames[i - 1];
                            Frame *shadow_current =
                                shadow_frames[i];

                            shadow_preintegrations
                                .emplace_back(
                                    source_current
                                        ->keyframe_preintegration);
                            PreIntegrator &preintegration =
                                shadow_preintegrations.back();
                            if (!preintegration.integrate(
                                    shadow_current->image->t,
                                    shadow_previous
                                        ->motion.bg,
                                    shadow_previous
                                        ->motion.ba,
                                    true, true)) {
                                return result;
                            }

                            solver->put_factor(
                                Solver::
                                    create_preintegration_error_factor(
                                        shadow_previous,
                                        shadow_current,
                                        preintegration));
                            ++result.imu_edges;
                        }

                        if (include_recovery) {
                            for (size_t i = 0;
                                 i < state
                                         .recovery_landmarks_world
                                         .size();
                                 ++i) {
                                solver
                                    ->add_learned_world_reprojection(
                                        shadow_target,
                                        state
                                            .recovery_landmarks_world[i],
                                        state
                                            .recovery_observations_pixel[i]);
                                ++result
                                      .recovery_factors;
                            }
                        }

                        result.complete =
                            result.prior_bound &&
                            result.imu_edges + 1 ==
                                result.window_frames &&
                            result.visual_mapping_failures ==
                                0;
                        result.usable = solver->solve();

                        result.target_pose =
                            shadow_target->pose;
                        result.target_motion =
                            shadow_target->motion;
                        result.recovery_rmse_px =
                            recovery_rmse_px(
                                shadow_target->pose);

                        for (size_t i = 0;
                             i < result.window_frames;
                             ++i) {
                            result.max_pose_delta_seed_t =
                                std::max(
                                    result
                                        .max_pose_delta_seed_t,
                                    (shadow_frames[i]
                                         ->pose.p -
                                     seed_poses[i].p)
                                        .norm());
                            result.max_pose_delta_seed_r_deg =
                                std::max(
                                    result
                                        .max_pose_delta_seed_r_deg,
                                    camera_rotation_delta_deg(
                                        seed_poses[i],
                                        shadow_frames[i]
                                            ->pose));
                            result.max_v_delta =
                                std::max(
                                    result.max_v_delta,
                                    (shadow_frames[i]
                                         ->motion.v -
                                     seed_motions[i].v)
                                        .norm());
                            result.max_bg_delta =
                                std::max(
                                    result.max_bg_delta,
                                    (shadow_frames[i]
                                         ->motion.bg -
                                     seed_motions[i].bg)
                                        .norm());
                            result.max_ba_delta =
                                std::max(
                                    result.max_ba_delta,
                                    (shadow_frames[i]
                                         ->motion.ba -
                                     seed_motions[i].ba)
                                        .norm());
                        }

                        result.final_inverse_depths.reserve(
                            shadow_inverse_depth_owners
                                .size());
                        for (size_t i = 0;
                             i <
                             shadow_inverse_depth_owners
                                 .size();
                             ++i) {
                            const double value =
                                *shadow_inverse_depth_owners[i];
                            result.final_inverse_depths
                                .emplace_back(value);
                            result.max_inv_depth_delta =
                                std::max(
                                    result
                                        .max_inv_depth_delta,
                                    std::abs(
                                        value -
                                        shadow_inverse_depth_sources[i]
                                            ->landmark
                                            .inv_depth));
                        }

                        return result;
                    };

                const FullWindowVisualShadowResult
                    identity_full_visual =
                        run_full_window_visual_shadow(
                            identity_correction_q,
                            identity_correction_p,
                            false,
                            false);
                const FullWindowVisualShadowResult
                    yaw_rebased_full_visual =
                        run_full_window_visual_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            true,
                            false);
                const FullWindowVisualShadowResult
                    yaw_rebased_full_visual_recovery =
                        run_full_window_visual_shadow(
                            yaw_correction_q,
                            yaw_correction_p,
                            true,
                            true);

                const auto full_visual_target_delta_seed_t =
                    [](const FullWindowVisualShadowResult
                           &result) {
                        return (result.target_pose.p -
                                result.target_seed_pose.p)
                            .norm();
                    };
                const auto full_visual_target_delta_seed_r =
                    [](const FullWindowVisualShadowResult
                           &result) {
                        return camera_rotation_delta_deg(
                            result.target_seed_pose,
                            result.target_pose);
                    };
                const auto full_visual_target_to_committed_t =
                    [&state](
                        const FullWindowVisualShadowResult
                            &result) {
                        return (result.target_pose.p -
                                state.committed_body_pose.p)
                            .norm();
                    };
                const auto full_visual_target_to_committed_r =
                    [&state](
                        const FullWindowVisualShadowResult
                            &result) {
                        return camera_rotation_delta_deg(
                            state.committed_body_pose,
                            result.target_pose);
                    };

                PoseState transformed_identity_full_pose;
                transformed_identity_full_pose.q =
                    yaw_correction_q *
                    identity_full_visual.target_pose.q;
                transformed_identity_full_pose.q.normalize();
                transformed_identity_full_pose.p =
                    yaw_correction_q *
                        identity_full_visual.target_pose.p +
                    yaw_correction_p;

                MotionState transformed_identity_full_motion =
                    identity_full_visual.target_motion;
                transformed_identity_full_motion.v =
                    yaw_correction_q *
                    identity_full_visual
                        .target_motion.v;

                const double full_equivalence_t =
                    (yaw_rebased_full_visual
                         .target_pose.p -
                     transformed_identity_full_pose.p)
                        .norm();
                const double full_equivalence_r =
                    camera_rotation_delta_deg(
                        transformed_identity_full_pose,
                        yaw_rebased_full_visual.target_pose);
                const double full_equivalence_v =
                    (yaw_rebased_full_visual
                         .target_motion.v -
                     transformed_identity_full_motion.v)
                        .norm();
                const double full_equivalence_bg =
                    (yaw_rebased_full_visual
                         .target_motion.bg -
                     transformed_identity_full_motion.bg)
                        .norm();
                const double full_equivalence_ba =
                    (yaw_rebased_full_visual
                         .target_motion.ba -
                     transformed_identity_full_motion.ba)
                        .norm();

                double full_equivalence_inv_depth = 0.0;
                bool full_inverse_depth_sizes_match =
                    identity_full_visual
                        .final_inverse_depths.size() ==
                    yaw_rebased_full_visual
                        .final_inverse_depths.size();
                if (full_inverse_depth_sizes_match) {
                    for (size_t i = 0;
                         i < identity_full_visual
                                 .final_inverse_depths
                                 .size();
                         ++i) {
                        full_equivalence_inv_depth =
                            std::max(
                                full_equivalence_inv_depth,
                                std::abs(
                                    identity_full_visual
                                        .final_inverse_depths[i] -
                                    yaw_rebased_full_visual
                                        .final_inverse_depths[i]));
                    }
                } else {
                    full_equivalence_inv_depth =
                        std::numeric_limits<double>::
                            infinity();
                }

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryFullWindowVisualShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f latest_refined_frame=%zu "
                    "topology=clone_full_refine_window "
                    "window_frames=%zu prior_frames=%zu "
                    "target_window_index=%zu "
                    "visual_track_states=%zu "
                    "visual_factors=%zu "
                    "visual_fixed_inv_depth=%zu "
                    "visual_depth_priors=%zu "
                    "visual_mapping_failures=%zu "
                    "ordinary_visual_cauchy=1.000 "
                    "yaw_correction_deg=%.9f "
                    "identity_usable=%d "
                    "identity_prior_bound=%d "
                    "identity_complete=%d "
                    "identity_imu_edges=%zu "
                    "identity_target_delta_seed_t=%.9f "
                    "identity_target_delta_seed_r_deg=%.9f "
                    "identity_max_delta_seed_t=%.9f "
                    "identity_max_delta_seed_r_deg=%.9f "
                    "identity_max_v_delta=%.9f "
                    "identity_max_bg_delta=%.9f "
                    "identity_max_ba_delta=%.9f "
                    "identity_max_inv_depth_delta=%.9f "
                    "yaw_rebased_usable=%d "
                    "yaw_rebased_prior_bound=%d "
                    "yaw_rebased_prior_rebased=%d "
                    "yaw_rebased_complete=%d "
                    "yaw_rebased_imu_edges=%zu "
                    "yaw_rebased_target_delta_seed_t=%.9f "
                    "yaw_rebased_target_delta_seed_r_deg=%.9f "
                    "yaw_rebased_max_delta_seed_t=%.9f "
                    "yaw_rebased_max_delta_seed_r_deg=%.9f "
                    "yaw_rebased_max_v_delta=%.9f "
                    "yaw_rebased_max_bg_delta=%.9f "
                    "yaw_rebased_max_ba_delta=%.9f "
                    "yaw_rebased_max_inv_depth_delta=%.9f "
                    "equivalence_t=%.9f "
                    "equivalence_r_deg=%.9f "
                    "equivalence_v=%.9f "
                    "equivalence_bg=%.9f "
                    "equivalence_ba=%.9f "
                    "equivalence_inv_depth_max_abs=%.9f "
                    "equivalence_inv_depth_sizes_match=%d "
                    "yaw_rebased_recovery_usable=%d "
                    "yaw_rebased_recovery_prior_bound=%d "
                    "yaw_rebased_recovery_prior_rebased=%d "
                    "yaw_rebased_recovery_complete=%d "
                    "yaw_rebased_recovery_imu_edges=%zu "
                    "yaw_rebased_recovery_factors=%zu "
                    "yaw_rebased_recovery_rmse_px=%.9f "
                    "yaw_rebased_recovery_target_delta_seed_t=%.9f "
                    "yaw_rebased_recovery_target_delta_seed_r_deg=%.9f "
                    "yaw_rebased_recovery_max_delta_seed_t=%.9f "
                    "yaw_rebased_recovery_max_delta_seed_r_deg=%.9f "
                    "yaw_rebased_recovery_max_v_delta=%.9f "
                    "yaw_rebased_recovery_max_bg_delta=%.9f "
                    "yaw_rebased_recovery_max_ba_delta=%.9f "
                    "yaw_rebased_recovery_max_inv_depth_delta=%.9f "
                    "yaw_rebased_recovery_target_to_committed_t=%.9f "
                    "yaw_rebased_recovery_target_to_committed_r_deg=%.9f "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    yaw_rebased_full_visual.window_frames,
                    yaw_rebased_full_visual.prior_frames,
                    yaw_rebased_full_visual.target_index,
                    yaw_rebased_full_visual
                        .visual_track_states,
                    yaw_rebased_full_visual.visual_factors,
                    yaw_rebased_full_visual
                        .visual_fixed_inv_depth,
                    yaw_rebased_full_visual
                        .visual_depth_priors,
                    yaw_rebased_full_visual
                        .visual_mapping_failures,
                    std::abs(yaw_correction_rad) *
                        180.0 / M_PI,
                    identity_full_visual.usable ? 1 : 0,
                    identity_full_visual.prior_bound ? 1 : 0,
                    identity_full_visual.complete ? 1 : 0,
                    identity_full_visual.imu_edges,
                    full_visual_target_delta_seed_t(
                        identity_full_visual),
                    full_visual_target_delta_seed_r(
                        identity_full_visual),
                    identity_full_visual
                        .max_pose_delta_seed_t,
                    identity_full_visual
                        .max_pose_delta_seed_r_deg,
                    identity_full_visual.max_v_delta,
                    identity_full_visual.max_bg_delta,
                    identity_full_visual.max_ba_delta,
                    identity_full_visual
                        .max_inv_depth_delta,
                    yaw_rebased_full_visual.usable ? 1 : 0,
                    yaw_rebased_full_visual.prior_bound
                        ? 1
                        : 0,
                    yaw_rebased_full_visual.prior_rebased
                        ? 1
                        : 0,
                    yaw_rebased_full_visual.complete
                        ? 1
                        : 0,
                    yaw_rebased_full_visual.imu_edges,
                    full_visual_target_delta_seed_t(
                        yaw_rebased_full_visual),
                    full_visual_target_delta_seed_r(
                        yaw_rebased_full_visual),
                    yaw_rebased_full_visual
                        .max_pose_delta_seed_t,
                    yaw_rebased_full_visual
                        .max_pose_delta_seed_r_deg,
                    yaw_rebased_full_visual.max_v_delta,
                    yaw_rebased_full_visual.max_bg_delta,
                    yaw_rebased_full_visual.max_ba_delta,
                    yaw_rebased_full_visual
                        .max_inv_depth_delta,
                    full_equivalence_t,
                    full_equivalence_r,
                    full_equivalence_v,
                    full_equivalence_bg,
                    full_equivalence_ba,
                    full_equivalence_inv_depth,
                    full_inverse_depth_sizes_match ? 1 : 0,
                    yaw_rebased_full_visual_recovery
                            .usable
                        ? 1
                        : 0,
                    yaw_rebased_full_visual_recovery
                            .prior_bound
                        ? 1
                        : 0,
                    yaw_rebased_full_visual_recovery
                            .prior_rebased
                        ? 1
                        : 0,
                    yaw_rebased_full_visual_recovery
                            .complete
                        ? 1
                        : 0,
                    yaw_rebased_full_visual_recovery
                        .imu_edges,
                    yaw_rebased_full_visual_recovery
                        .recovery_factors,
                    yaw_rebased_full_visual_recovery
                        .recovery_rmse_px,
                    full_visual_target_delta_seed_t(
                        yaw_rebased_full_visual_recovery),
                    full_visual_target_delta_seed_r(
                        yaw_rebased_full_visual_recovery),
                    yaw_rebased_full_visual_recovery
                        .max_pose_delta_seed_t,
                    yaw_rebased_full_visual_recovery
                        .max_pose_delta_seed_r_deg,
                    yaw_rebased_full_visual_recovery
                        .max_v_delta,
                    yaw_rebased_full_visual_recovery
                        .max_bg_delta,
                    yaw_rebased_full_visual_recovery
                        .max_ba_delta,
                    yaw_rebased_full_visual_recovery
                        .max_inv_depth_delta,
                    full_visual_target_to_committed_t(
                        yaw_rebased_full_visual_recovery),
                    full_visual_target_to_committed_r(
                        yaw_rebased_full_visual_recovery));

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryMarginalizationShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f latest_refined_frame=%zu "
                    "topology=active_top_level_window "
                    "visual_factors=0 "
                    "window_frames=%zu prior_frames=%zu "
                    "target_window_index=%zu "
                    "yaw_correction_deg=%.9f "
                    "identity_usable=%d "
                    "identity_prior_bound=%d "
                    "identity_complete=%d "
                    "identity_imu_edges=%zu "
                    "identity_target_delta_seed_t=%.9f "
                    "identity_target_delta_seed_r_deg=%.9f "
                    "identity_max_delta_seed_t=%.9f "
                    "identity_max_delta_seed_r_deg=%.9f "
                    "identity_max_v_delta=%.9f "
                    "yaw_no_marg_target_delta_seed_t=%.9f "
                    "yaw_no_marg_target_delta_seed_r_deg=%.9f "
                    "yaw_usable=%d yaw_prior_bound=%d "
                    "yaw_complete=%d yaw_imu_edges=%zu "
                    "yaw_target_delta_seed_t=%.9f "
                    "yaw_target_delta_seed_r_deg=%.9f "
                    "yaw_max_delta_seed_t=%.9f "
                    "yaw_max_delta_seed_r_deg=%.9f "
                    "yaw_max_v_delta=%.9f "
                    "yaw_target_to_committed_t=%.9f "
                    "yaw_target_to_committed_r_deg=%.9f "
                    "yaw_final_to_identity_final_t=%.9f "
                    "yaw_final_to_identity_final_r_deg=%.9f "
                    "yaw_recovery_usable=%d "
                    "yaw_recovery_prior_bound=%d "
                    "yaw_recovery_complete=%d "
                    "yaw_recovery_imu_edges=%zu "
                    "yaw_recovery_factors=%zu "
                    "yaw_recovery_rmse_px=%.9f "
                    "yaw_recovery_target_delta_seed_t=%.9f "
                    "yaw_recovery_target_delta_seed_r_deg=%.9f "
                    "yaw_recovery_max_delta_seed_t=%.9f "
                    "yaw_recovery_max_delta_seed_r_deg=%.9f "
                    "yaw_recovery_max_v_delta=%.9f "
                    "yaw_recovery_target_to_committed_t=%.9f "
                    "yaw_recovery_target_to_committed_r_deg=%.9f "
                    "marginalization_source=live_read_only "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    identity_marginalization.window_frames,
                    identity_marginalization.prior_frames,
                    identity_marginalization.target_index,
                    std::abs(yaw_correction_rad) *
                        180.0 / M_PI,
                    identity_marginalization.usable ? 1 : 0,
                    identity_marginalization.prior_bound ? 1 : 0,
                    identity_marginalization.complete ? 1 : 0,
                    identity_marginalization.imu_edges,
                    marginalization_target_delta_seed_t(
                        identity_marginalization),
                    marginalization_target_delta_seed_r(
                        identity_marginalization),
                    identity_marginalization
                        .max_pose_delta_seed_t,
                    identity_marginalization
                        .max_pose_delta_seed_r_deg,
                    identity_marginalization.max_v_delta,
                    window_target_delta_seed_t(
                        yaw_window_imu),
                    window_target_delta_seed_r(
                        yaw_window_imu),
                    yaw_marginalization.usable ? 1 : 0,
                    yaw_marginalization.prior_bound ? 1 : 0,
                    yaw_marginalization.complete ? 1 : 0,
                    yaw_marginalization.imu_edges,
                    marginalization_target_delta_seed_t(
                        yaw_marginalization),
                    marginalization_target_delta_seed_r(
                        yaw_marginalization),
                    yaw_marginalization
                        .max_pose_delta_seed_t,
                    yaw_marginalization
                        .max_pose_delta_seed_r_deg,
                    yaw_marginalization.max_v_delta,
                    marginalization_target_to_committed_t(
                        yaw_marginalization),
                    marginalization_target_to_committed_r(
                        yaw_marginalization),
                    yaw_final_to_identity_final_t,
                    yaw_final_to_identity_final_r,
                    yaw_marginalization_recovery.usable ? 1 : 0,
                    yaw_marginalization_recovery.prior_bound
                        ? 1
                        : 0,
                    yaw_marginalization_recovery.complete
                        ? 1
                        : 0,
                    yaw_marginalization_recovery.imu_edges,
                    yaw_marginalization_recovery
                        .recovery_factors,
                    yaw_marginalization_recovery
                        .recovery_rmse_px,
                    marginalization_target_delta_seed_t(
                        yaw_marginalization_recovery),
                    marginalization_target_delta_seed_r(
                        yaw_marginalization_recovery),
                    yaw_marginalization_recovery
                        .max_pose_delta_seed_t,
                    yaw_marginalization_recovery
                        .max_pose_delta_seed_r_deg,
                    yaw_marginalization_recovery.max_v_delta,
                    marginalization_target_to_committed_t(
                        yaw_marginalization_recovery),
                    marginalization_target_to_committed_r(
                        yaw_marginalization_recovery));

                std::fprintf(
                    stderr,
                    "[PlaceRecoveryFactorIntegrationShadow] "
                    "event_id=%zu commit_frame=%zu "
                    "commit_t=%.9f "
                    "latest_refined_frame=%zu "
                    "topology=localize_single_frame "
                    "marginalization_prior=0 "
                    "conventional_visual_factors=%zu "
                    "self_anchor_skipped=%zu "
                    "imu_prior=%d "
                    "recovery_factors=%zu "
                    "baseline_usable=%d "
                    "augmented_usable=%d "
                    "current_recovery_rmse_px=%.9f "
                    "baseline_recovery_rmse_px=%.9f "
                    "augmented_recovery_rmse_px=%.9f "
                    "baseline_delta_current_t=%.9f "
                    "baseline_delta_current_r_deg=%.9f "
                    "augmented_delta_current_t=%.9f "
                    "augmented_delta_current_r_deg=%.9f "
                    "augmented_delta_baseline_t=%.9f "
                    "augmented_delta_baseline_r_deg=%.9f "
                    "augmented_to_committed_t=%.9f "
                    "augmented_to_committed_r_deg=%.9f "
                    "augmented_to_precommit_t=%.9f "
                    "augmented_to_precommit_r_deg=%.9f "
                    "baseline_v_delta=%.9f "
                    "baseline_bg_delta=%.9f "
                    "baseline_ba_delta=%.9f "
                    "augmented_v_delta=%.9f "
                    "augmented_bg_delta=%.9f "
                    "augmented_ba_delta=%.9f "
                    "state_mutation=0\n",
                    state.event_id,
                    state.frame_id,
                    state.timestamp,
                    latest_refined_frame_id,
                    augmented.conventional_visual_factors,
                    augmented.self_anchor_skipped,
                    augmented.imu_prior ? 1 : 0,
                    augmented.recovery_factors,
                    baseline.usable ? 1 : 0,
                    augmented.usable ? 1 : 0,
                    current_recovery_rmse_px,
                    baseline.recovery_rmse_px,
                    augmented.recovery_rmse_px,
                    baseline_delta_current_t,
                    baseline_delta_current_r,
                    augmented_delta_current_t,
                    augmented_delta_current_r,
                    augmented_delta_baseline_t,
                    augmented_delta_baseline_r,
                    augmented_to_committed_t,
                    augmented_to_committed_r,
                    augmented_to_precommit_t,
                    augmented_to_precommit_r,
                    baseline_v_delta,
                    baseline_bg_delta,
                    baseline_ba_delta,
                    augmented_v_delta,
                    augmented_bg_delta,
                    augmented_ba_delta);
            }
        }

        const double current_to_committed_t =
            (target->pose.p - state.committed_body_pose.p).norm();
        const double current_to_committed_r =
            camera_rotation_delta_deg(
                state.committed_body_pose, target->pose);
        const double current_to_precommit_t =
            (target->pose.p - state.precommit_body_pose.p).norm();
        const double current_to_precommit_r =
            camera_rotation_delta_deg(
                state.precommit_body_pose, target->pose);

        const double v_delta =
            (target->motion.v - state.committed_motion.v).norm();
        const double bg_delta =
            (target->motion.bg - state.committed_motion.bg).norm();
        const double ba_delta =
            (target->motion.ba - state.committed_motion.ba).norm();

        const size_t frame_gap =
            latest_refined_frame_id >= state.frame_id
                ? latest_refined_frame_id - state.frame_id
                : state.frame_id - latest_refined_frame_id;
        ++state.samples_emitted;

        std::fprintf(
            stderr,
            "[PlaceRecoveryCommitReconcile] "
            "event_id=%zu commit_frame=%zu "
            "commit_t=%.9f sample=%zu "
            "latest_refined_frame=%zu frame_gap=%zu "
            "target_active=1 target_is_top_level=%d "
            "current_to_committed_t=%.9f "
            "current_to_committed_r_deg=%.9f "
            "current_to_precommit_t=%.9f "
            "current_to_precommit_r_deg=%.9f "
            "current_v_delta=%.9f "
            "current_bg_delta=%.9f "
            "current_ba_delta=%.9f "
            "target_body_p=%.9f,%.9f,%.9f "
            "target_body_q=%.9f,%.9f,%.9f,%.9f "
            "retired=0 state_mutation=0\n",
            state.event_id,
            state.frame_id,
            state.timestamp,
            state.samples_emitted,
            latest_refined_frame_id,
            frame_gap,
            target_is_top_level ? 1 : 0,
            current_to_committed_t,
            current_to_committed_r,
            current_to_precommit_t,
            current_to_precommit_r,
            v_delta,
            bg_delta,
            ba_delta,
            target->pose.p.x(),
            target->pose.p.y(),
            target->pose.p.z(),
            target->pose.q.x(),
            target->pose.q.y(),
            target->pose.q.z(),
            target->pose.q.w());

        active.emplace_back(std::move(state));
    }

    active_place_recovery_commit_reconciliations_ =
        std::move(active);
}

void SlidingWindowTracker::extract_place_descriptors() {
    if (!place_descriptor_shadow_enabled())
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
        place_retrieval_shadow_enabled();
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
                orb_pnp_shadow_enabled() &&
                !candidates.empty();
            if (retrieval_verification_pending) {
                PendingPlaceRetrieval pending;
                pending.frame_id = frame->id();
                pending.candidates = candidates;
                pending_place_retrieval_candidates_.emplace_back(
                    std::move(pending));
            }

            bool local_ready = true;
            if (local_descriptor_shadow_enabled()) {
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
        diagnose_place_recovery_commit_reconciliation();
        extract_place_descriptors();
        archive_optimized_keyframes();
        diagnose_pending_retrieved_place_candidates();
        slide_window();
    } else {
        refine_subwindow();
    }

    synchronize_feature_tracking_landmarks();

    // Shadow-only keyframe instrumentation for place-recognition development.
    // This does not mutate SLAM state. It reports each newly observed
    // top-level keyframe together with the active local keyframe set after
    // keyframe management, optimization, and window sliding have completed.
    if (const char *value = std::getenv("XRSLAM_PLACE_KEYFRAME_SHADOW");
        value && std::string(value) == "1") {
        static std::unordered_set<size_t> logged_frame_ids;

        std::string active_frame_ids;
        for (size_t i = 0; i < map->frame_num(); ++i) {
            Frame *frame = map->get_frame(i);
            if (!frame || !frame->tag(FT_KEYFRAME))
                continue;

            if (!active_frame_ids.empty())
                active_frame_ids += ';';
            active_frame_ids += std::to_string(frame->id());
        }

        for (size_t i = 0; i < map->frame_num(); ++i) {
            Frame *frame = map->get_frame(i);
            if (!frame || !frame->tag(FT_KEYFRAME) || !frame->image)
                continue;
            if (!logged_frame_ids.emplace(frame->id()).second)
                continue;

            const PoseState camera_pose =
                frame->get_pose(frame->camera);
            std::fprintf(
                stderr,
                "[PlaceKeyframeShadow] frame_id=%zu t=%.9f "
                "p=%.9f,%.9f,%.9f q=%.9f,%.9f,%.9f,%.9f "
                "active=%s\n",
                frame->id(),
                frame->image->t,
                camera_pose.p.x(),
                camera_pose.p.y(),
                camera_pose.p.z(),
                camera_pose.q.x(),
                camera_pose.q.y(),
                camera_pose.q.z(),
                camera_pose.q.w(),
                active_frame_ids.c_str());
        }
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

void SlidingWindowTracker::synchronize_feature_tracking_landmarks() {
    if (!feature_tracking_map)
        return;

    size_t source_landmarks = 0;
    size_t synchronized_landmarks = 0;
    size_t missing_feature_track = 0;
    size_t synchronized_poses = 0;

    synchronized(feature_tracking_map) {
        // Preserve optimized camera poses for SWT keyframes AND subframes.
        // Recovery support anchors are indexed by high-rate Frame ID, so only
        // caching top-level keyframes would miss many reference-10/reference-20
        // support frames.
        const auto cache_source_pose =
            [&](Frame *source_frame) {
                if (!source_frame)
                    return;

                const size_t feature_frame_index =
                    feature_tracking_map->frame_index_by_id(
                        source_frame->id());
                if (feature_frame_index == nil())
                    return;

                set_recovery_camera_pose_world(
                    source_frame->id(),
                    source_frame->get_pose(source_frame->camera));
                ++synchronized_poses;
            };

        for (size_t frame_index = 0;
             frame_index < map->frame_num(); ++frame_index) {
            Frame *source_frame = map->get_frame(frame_index);
            cache_source_pose(source_frame);
            if (!source_frame)
                continue;

            for (const auto &subframe : source_frame->subframes)
                cache_source_pose(subframe.get());
        }

        // Learned recovery maps are intentionally not materialized here.
        // 0060 builds only recovery-relevant anchors on demand in
        // Frame::recover_keypoints(), using optimized +/-10 support poses
        // already captured in recovery_pose_cache.

        // The cache is a snapshot of the current optimized window. Never let
        // marginalized/rejected landmarks survive indefinitely in the
        // high-rate feature map.
        for (size_t track_index = 0;
             track_index < feature_tracking_map->track_num();
             ++track_index) {
            if (Track *feature_track =
                    feature_tracking_map->get_track(track_index)) {
                feature_track->recovery_landmark_world.reset();
            }
        }

        for (size_t track_index = 0;
             track_index < map->track_num(); ++track_index) {
            Track *source_track = map->get_track(track_index);
            if (!source_track ||
                source_track->tag(TT_TRASH) ||
                !source_track->all_tagged(
                    TT_VALID, TT_TRIANGULATED, TT_STATIC)) {
                continue;
            }

            ++source_landmarks;
            const vector<3> landmark =
                source_track->get_landmark_point();

            Track *feature_track = nullptr;

            // Mirrored SWT frames preserve frame ids and keypoint indices.
            // Find the newest observation that still exists in the high-rate
            // feature-tracking map, then recover the persistent FT Track
            // through that exact observation.
            for (auto it =
                     source_track->keypoint_map().rbegin();
                 it != source_track->keypoint_map().rend();
                 ++it) {
                Frame *source_frame = it->first;
                const size_t keypoint_index = it->second;

                const size_t feature_frame_index =
                    feature_tracking_map->frame_index_by_id(
                        source_frame->id());
                if (feature_frame_index == nil())
                    continue;

                Frame *feature_frame =
                    feature_tracking_map->get_frame(
                        feature_frame_index);
                if (!feature_frame ||
                    keypoint_index >=
                        feature_frame->keypoint_num()) {
                    continue;
                }

                Track *candidate =
                    feature_frame->get_track(keypoint_index);
                if (!candidate ||
                    candidate->tag(TT_TRASH)) {
                    continue;
                }

                feature_track = candidate;
                break;
            }

            if (!feature_track) {
                ++missing_feature_track;
                continue;
            }

            // Exact recovery-only world point. Do not mutate inverse depth
            // or estimator tags in the feature-tracking map.
            feature_track->recovery_landmark_world = landmark;
            ++synchronized_landmarks;
        }
    }

    static size_t sync_diag_count = 0;
    ++sync_diag_count;
    if (sync_diag_count == 1 ||
        sync_diag_count % 100 == 0 ||
        (source_landmarks > 0 &&
         synchronized_landmarks == 0)) {
        std::fprintf(
            stderr,
            "[LandmarkSync] calls=%zu poses=%zu source=%zu "
            "synchronized=%zu missing=%zu\n",
            sync_diag_count, synchronized_poses,
            source_landmarks, synchronized_landmarks,
            missing_feature_track);
    }
}


void SlidingWindowTracker::localize_newframe() {
    Frame *frame_i = map->get_frame(map->frame_num() - 2);
    if (!frame_i->subframes.empty()) {
        frame_i = frame_i->subframes.back().get();
    }
    Frame *frame_j = map->get_frame(map->frame_num() - 1);

    const std::optional<RecoveryLearnedObservationSet>
        learned_observations =
            get_recovery_learned_observations(frame_j->id());
    const bool run_recovery_shadow =
        learned_recovery_shadow_enabled() &&
        learned_observations.has_value() &&
        learned_observations->pnp_pose_valid &&
        learned_observations->landmarks_world.size() >= 5 &&
        learned_observations->landmarks_world.size() ==
            learned_observations->observations_pixel.size();

    // Ordinary XRSLAM localization remains authoritative.
    auto solver = Solver::create();
    solver->add_frame_states(frame_j);

    if (config->has_imu()) {
        solver->put_factor(Solver::create_preintegration_prior_factor(
            frame_i, frame_j, frame_j->preintegration));
    }

    size_t conventional_priors = 0;
    for (size_t k = 0; k < frame_j->keypoint_num(); ++k) {
        if (Track *track = frame_j->get_track(k)) {
            if (track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC)) {
                solver->put_factor(
                    Solver::create_reprojection_prior_factor(frame_j, track));
                ++conventional_priors;
            }
        }
    }

    const bool authoritative_usable = solver->solve();

    if (run_recovery_shadow) {
        const PoseState authoritative_pose = frame_j->pose;
        const MotionState authoritative_motion = frame_j->motion;
        const PoseState authoritative_camera_pose =
            frame_j->get_pose(frame_j->camera);
        const double authoritative_learned_rmse =
            recovery_observation_rmse_px(
                frame_j, *learned_observations);

        // Convert the learned PnP camera pose into XRSLAM's body pose.
        PoseState pnp_body_seed = authoritative_pose;
        pnp_body_seed.q =
            learned_observations->pnp_q_wc *
            frame_j->camera.q_cs.conjugate();
        pnp_body_seed.q.normalize();
        pnp_body_seed.p =
            learned_observations->pnp_p_wc -
            pnp_body_seed.q * frame_j->camera.p_cs;

        frame_j->pose = pnp_body_seed;
        frame_j->motion = authoritative_motion;

        const PoseState pnp_seed_camera_pose =
            frame_j->get_pose(frame_j->camera);
        const quaternion pnp_seed_q_wc =
            pnp_seed_camera_pose.q;
        const vector<3> pnp_seed_p_wc =
            pnp_seed_camera_pose.p;
        const double pnp_seed_rmse =
            recovery_observation_rmse_px(
                frame_j, *learned_observations);

        // Dedicated recovery localizer:
        //   - start inside the learned/PnP basin,
        //   - retain the IMU prior when available,
        //   - use learned fixed-world observations,
        //   - deliberately omit conventional visual priors because they are
        //     the measurements considered unreliable during relocalization.
        auto recovery_solver = Solver::create();
        recovery_solver->add_frame_states(frame_j);

        // Do not apply the one-step preintegration prior on the relocalization
        // frame. That prior encodes continuity from the branch considered
        // broken and was shown in 0066 to erase the learned global-pose seed.
        // IMU remains available as a disagreement/acceptance signal and normal
        // preintegration resumes after a recovery is actually committed.
        size_t learned_factor_count = 0;
        for (size_t i = 0;
             i < learned_observations->landmarks_world.size(); ++i) {
            recovery_solver->add_learned_world_reprojection(
                frame_j,
                learned_observations->landmarks_world[i],
                learned_observations->observations_pixel[i]);
            ++learned_factor_count;
        }

        const bool recovery_usable = recovery_solver->solve();

        const PoseState recovery_camera_pose =
            frame_j->get_pose(frame_j->camera);
        const double recovery_learned_rmse =
            recovery_observation_rmse_px(
                frame_j, *learned_observations);

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

        const double recovery_delta_pnp_t =
            (recovery_camera_pose.p -
             pnp_seed_camera_pose.p)
                .norm();
        const double recovery_delta_pnp_r =
            camera_rotation_delta_deg(
                pnp_seed_camera_pose,
                recovery_camera_pose);

        const double recovery_delta_v =
            (frame_j->motion.v -
             authoritative_motion.v)
                .norm();

        const quaternion &q_candidate =
            recovery_camera_pose.q;
        const vector<3> &p_candidate =
            recovery_camera_pose.p;

        // The candidate is diagnostic only. Restore the normal localizer
        // result before any later frontend/keyframe/marginalization logic.
        frame_j->pose = authoritative_pose;
        frame_j->motion = authoritative_motion;

        std::fprintf(
            stderr,
            "[LearnedRecovery] shadow current=%zu reference=%zu "
            "learned=%zu conventional_authoritative=%zu "
            "pnp_inliers=%zu pnp_ratio=%.3f "
            "authoritative_usable=%d recovery_usable=%d has_imu=%d "
            "authoritative_rmse_px=%.3f pnp_seed_rmse_px=%.3f "
            "recovery_rmse_px=%.3f "
            "seed_delta_authoritative_t=%.4f "
            "seed_delta_authoritative_r_deg=%.3f "
            "recovery_delta_authoritative_t=%.4f "
            "recovery_delta_authoritative_r_deg=%.3f "
            "recovery_delta_pnp_t=%.4f "
            "recovery_delta_pnp_r_deg=%.3f "
            "recovery_delta_v=%.4f "
            "pnp_seed_p=(%.9f,%.9f,%.9f) "
            "pnp_seed_q=(%.9f,%.9f,%.9f,%.9f) "
            "candidate_p=(%.9f,%.9f,%.9f) "
            "candidate_q=(%.9f,%.9f,%.9f,%.9f) "
            "learned_cauchy=2.448 conventional_recovery=0 "
            "imu_factor_recovery=0 pnp_seed=1 "
            "authoritative_restored=1\n",
            frame_j->id(),
            learned_observations->reference_frame_id,
            learned_factor_count,
            conventional_priors,
            learned_observations->pnp_inliers,
            learned_observations->pnp_ratio,
            authoritative_usable ? 1 : 0,
            recovery_usable ? 1 : 0,
            config->has_imu() ? 1 : 0,
            authoritative_learned_rmse,
            pnp_seed_rmse,
            recovery_learned_rmse,
            seed_delta_authoritative_t,
            seed_delta_authoritative_r,
            recovery_delta_authoritative_t,
            recovery_delta_authoritative_r,
            recovery_delta_pnp_t,
            recovery_delta_pnp_r,
            recovery_delta_v,
            pnp_seed_p_wc.x(), pnp_seed_p_wc.y(), pnp_seed_p_wc.z(),
            pnp_seed_q_wc.x(), pnp_seed_q_wc.y(),
            pnp_seed_q_wc.z(), pnp_seed_q_wc.w(),
            p_candidate.x(), p_candidate.y(), p_candidate.z(),
            q_candidate.x(), q_candidate.y(),
            q_candidate.z(), q_candidate.w());

        erase_recovery_learned_observations(frame_j->id());
    }
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
