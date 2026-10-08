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
        [frame](
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
