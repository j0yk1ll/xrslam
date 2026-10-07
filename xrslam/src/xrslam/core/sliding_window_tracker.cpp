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

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>

namespace xrslam {

namespace {

bool learned_recovery_shadow_enabled() {
    const char *value = std::getenv(
        "XRSLAM_LEARNED_RECOVERY_SHADOW");
    return value && std::string(value) == "1";
}

bool place_descriptor_shadow_enabled() {
    const char *value =
        std::getenv("XRSLAM_PLACE_DESCRIPTOR_SHADOW");
    return value && std::string(value) == "1";
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
    const char *value = std::getenv("XRSLAM_KEYFRAME_ARCHIVE_SHADOW");
    if (!value || std::string(value) != "1")
        return;

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
        const bool inserted =
            keyframe_archive_.upsert(std::move(archived));

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

            if (!place_keyframes_.add(place_keyframe)) {
                std::fprintf(
                    stderr,
                    "[PlaceDescriptorShadow] frame_id=%zu t=%.9f "
                    "reject=duplicate_key\n",
                    frame->id(), frame->image->t);
                continue;
            }

            frame->image->retain_place_recognition_source(false);
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
