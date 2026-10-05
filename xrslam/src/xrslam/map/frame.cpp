#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <unordered_map>

#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <xrslam/core/recovery_learned_map.h>
#include <xrslam/estimation/ceres/quaternion_parameterization.h>
#include <xrslam/core/recovery_pose_cache.h>
#include <xrslam/estimation/reprojection_factor.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/inspection.h>
#include <xrslam/local_feature_backend.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/map/track.h>
#include <xrslam/utility/poisson_disk_filter.h>

namespace xrslam {

namespace {

struct LearnedFixedWorldReprojectionCost {
    LearnedFixedWorldReprojectionCost(
        const vector<3> &landmark_world,
        const vector<2> &observation_pixel,
        const matrix<3> &K,
        const ExtrinsicParams &camera)
        : landmark_world(landmark_world),
          observation_pixel(observation_pixel),
          K(K), camera(camera) {}

    template <typename T>
    bool operator()(const T *const q_wb_data,
                    const T *const p_wb_data,
                    T *residuals) const {
        Eigen::Map<const Eigen::Quaternion<T>> q_wb(q_wb_data);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_wb(p_wb_data);

        Eigen::Quaternion<T> q_bc =
            camera.q_cs.template cast<T>();
        Eigen::Matrix<T, 3, 1> p_bc =
            camera.p_cs.template cast<T>();
        Eigen::Quaternion<T> q_wc = q_wb * q_bc;
        Eigen::Matrix<T, 3, 1> p_wc =
            p_wb + q_wb * p_bc;

        const Eigen::Matrix<T, 3, 1> point_world =
            landmark_world.template cast<T>();
        const Eigen::Matrix<T, 3, 1> point_camera =
            q_wc.conjugate() * (point_world - p_wc);

        const T inv_z = T(1.0) / point_camera.z();
        const T u =
            T(K(0, 0)) * point_camera.x() * inv_z +
            T(K(0, 2));
        const T v =
            T(K(1, 1)) * point_camera.y() * inv_z +
            T(K(1, 2));

        residuals[0] = u - T(observation_pixel.x());
        residuals[1] = v - T(observation_pixel.y());
        return true;
    }

    vector<3> landmark_world;
    vector<2> observation_pixel;
    matrix<3> K;
    ExtrinsicParams camera;
};

double pose_rotation_delta_deg(
    const quaternion &a, const quaternion &b) {
    const matrix<3> R_delta =
        a.conjugate().matrix() * b.matrix();
    const double cosine =
        std::max(
            -1.0,
            std::min(
                1.0,
                (R_delta.trace() - 1.0) * 0.5));
    return std::acos(cosine) * 180.0 / M_PI;
}

double learned_reprojection_rmse_px(
    const PoseState &body_pose,
    const ExtrinsicParams &camera,
    const matrix<3> &K,
    const std::vector<vector<3>> &landmarks,
    const std::vector<vector<2>> &observations) {
    if (landmarks.empty() ||
        landmarks.size() != observations.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const quaternion q_wc =
        body_pose.q * camera.q_cs;
    const vector<3> p_wc =
        body_pose.p + body_pose.q * camera.p_cs;

    double squared_error_sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const vector<3> point_camera =
            q_wc.conjugate() *
            (landmarks[i] - p_wc);
        if (!point_camera.allFinite() ||
            point_camera.z() <= 1.0e-6) {
            continue;
        }

        const vector<2> projected =
            apply_k(point_camera, K);
        const double error =
            (projected - observations[i]).norm();
        squared_error_sum += error * error;
        ++count;
    }

    if (count == 0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(
        squared_error_sum / static_cast<double>(count));
}

} // namespace

struct Frame::construct_by_frame_t {};

Frame::Frame() : map(nullptr) {}

Frame::Frame(const Frame &frame, const construct_by_frame_t &construct_by_frame)
    : Tagged(frame), Identifiable(frame), map(nullptr) {}

Frame::~Frame() = default;

std::unique_ptr<Frame> Frame::clone() const {
    std::unique_ptr<Frame> frame =
        std::make_unique<Frame>(*this, construct_by_frame_t());
    frame->K = K;
    frame->sqrt_inv_cov = sqrt_inv_cov;
    frame->image = image;
    frame->use_depth = use_depth;
    frame->pose = pose;
    frame->motion = motion;
    frame->camera = camera;
    frame->imu = imu;
    frame->preintegration = preintegration;
    frame->bearings = bearings;
    frame->tracks = std::vector<Track *>(bearings.size(), nullptr);
    frame->reprojection_error_factors = std::vector<std::unique_ptr<ReprojectionErrorFactor>>(bearings.size());
    frame->map = nullptr;
    return frame;
}

void Frame::append_keypoint(const vector<3> &keypoint) {
    bearings.emplace_back(keypoint);
    tracks.emplace_back(nullptr);
    reprojection_error_factors.emplace_back(nullptr);
}

Track *Frame::get_track(size_t keypoint_index, Map *allocation_map) {
    if (!allocation_map) {
        allocation_map = map;
    }
    if (tracks[keypoint_index] == nullptr) {
        Track *track = allocation_map->create_track();
        track->add_keypoint(this, keypoint_index);
        if (use_depth && image && image->has_depth()) {
            const vector<3> &bearing = bearings[keypoint_index];
            const vector<2> pixel = apply_k(bearing, K);
            const double z = image->depth(pixel);
            const double depth_confidence = image->depth_confidence(pixel);
            if (std::isfinite(z) && z > 0.0 && bearing.z() > 1.0e-6 &&
                std::isfinite(depth_confidence) && depth_confidence > 0.0) {
                // Landmark inverse depth is inverse range along the unit
                // bearing, while RGB-D convention is optical-axis z depth.
                track->landmark.inv_depth = bearing.z() / z;
                track->has_depth_prior = true;
                track->depth_prior_inv_depth = track->landmark.inv_depth;
                track->depth_prior_source = image->depth_source();
                track->depth_prior_confidence =
                    std::max(0.0, std::min(1.0, depth_confidence));
                track->tag(TT_VALID) = true;
                track->tag(TT_TRIANGULATED) = true;
                track->tag(TT_STATIC) = true;
                track->tag(TT_FIX_INVD) = false;
            }
        }
    }
    return tracks[keypoint_index];
}

void Frame::detect_keypoints(
    Config *config, bool recovery_anchor) {
    std::vector<vector<2>> pkeypoints(bearings.size());
    for (size_t i = 0; i < bearings.size(); ++i) {
        pkeypoints[i] = apply_k(bearings[i], K);
    }

    auto backend = local_feature_backend();
    if (recovery_anchor) {
        backend->detect_recovery_anchor_keypoints(
            image.get(), pkeypoints,
            config->feature_tracker_max_keypoint_detection(),
            config->feature_tracker_min_keypoint_distance());
    } else {
        backend->detect_keypoints(
            image.get(), pkeypoints,
            config->feature_tracker_max_keypoint_detection(),
            config->feature_tracker_min_keypoint_distance());
    }

    size_t old_keypoint_num = bearings.size();
    bearings.resize(pkeypoints.size());
    tracks.resize(pkeypoints.size(), nullptr);
    reprojection_error_factors.resize(pkeypoints.size());
    for (size_t i = old_keypoint_num; i < pkeypoints.size(); ++i) {
        bearings[i] = remove_k(pkeypoints[i], K);
    }
}

void Frame::track_keypoints(Frame *next_frame, Config *config) {
    std::vector<vector<2>> curr_keypoints(bearings.size());
    std::vector<vector<2>> next_keypoints;

    for (size_t i = 0; i < bearings.size(); ++i) {
        curr_keypoints[i] = apply_k(bearings[i], K);
    }

    if (config->has_imu() && config->feature_tracker_predict_keypoints()) {
        quaternion delta_key_q =
            (camera.q_cs.conjugate() * imu.q_cs *
             next_frame->preintegration.delta.q *
             next_frame->imu.q_cs.conjugate() * next_frame->camera.q_cs)
                .conjugate();
        next_keypoints.resize(curr_keypoints.size());
        for (size_t i = 0; i < bearings.size(); ++i) {
            next_keypoints[i] =
                apply_k(delta_key_q * bearings[i], next_frame->K);
        }
    }

    std::vector<char> status, mask;
    auto backend = local_feature_backend();
    backend->track_keypoints(image.get(), next_frame->image.get(),
                             curr_keypoints, next_keypoints, status);

    std::vector<vector<2>> curr_keypoints_h, next_keypoints_h;
    std::vector<vector<3>> next_bearings;
    for (size_t i = 0; i < curr_keypoints.size(); ++i) {
        curr_keypoints_h.push_back(bearings[i].hnormalized());
        vector<3> next_keypoint = remove_k(next_keypoints[i], next_frame->K);
        next_keypoints_h.push_back(next_keypoint.hnormalized());
        next_bearings.push_back(next_keypoint);
    }

    matrix<3> E =
        find_essential_matrix(curr_keypoints_h, next_keypoints_h, mask, 1.0);
    for (size_t i = 0; i < status.size(); ++i) {
        if (!mask[i]) {
            status[i] = 0;
        }
    }
    matrix<3> R = find_rotation_matrix(bearings, next_bearings, mask,
                                       (M_PI / 180.0) *
                                           config->rotation_ransac_threshold());

    std::vector<double> angles;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            double angle = acos((R * bearings[i]).dot(next_bearings[i]));
            angles.emplace_back(angle * 180 / M_PI);
        }
    }
    std::sort(angles.begin(), angles.end());
    double misalignment =
        angles.size() > 0 ? angles[angles.size() * 7 / 10] : 0;
    inspect_debug(feature_tracker_angle_misalignment, angle_misalignment) {
        angle_misalignment = misalignment;
    }
    if (misalignment < config->rotation_misalignment_threshold()) {
        next_frame->tag(FT_NO_TRANSLATION) = true;
    }

    // filter keypoints based on track length
    std::vector<std::pair<size_t, size_t>> keypoint_index_track_length;
    keypoint_index_track_length.reserve(curr_keypoints.size());
    for (size_t i = 0; i < curr_keypoints.size(); ++i) {
        if (status[i] == 0)
            continue;
        Track *track = get_track(i);
        if (track == nullptr)
            continue;
        keypoint_index_track_length.emplace_back(i, track->keypoint_num());
    }

    std::sort(keypoint_index_track_length.begin(),
              keypoint_index_track_length.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    PoissonDiskFilter<2> filter(
        config->feature_tracker_min_keypoint_distance());
    for (auto &[keypoint_index, track_length] : keypoint_index_track_length) {
        vector<2> pt = next_keypoints[keypoint_index];
        Track *track = this->get_track(keypoint_index);
        if (filter.permit_point(pt) &&
            (!track || (track && !track->tag(TT_TRASH)))) {
            filter.preset_point(pt);
        } else {
            status[keypoint_index] = 0;
        }
    }

    for (size_t curr_keypoint_index = 0;
         curr_keypoint_index < curr_keypoints.size(); ++curr_keypoint_index) {
        if (status[curr_keypoint_index]) {
            size_t next_keypoint_index = next_frame->keypoint_num();
            next_frame->append_keypoint(next_bearings[curr_keypoint_index]);
            get_track(curr_keypoint_index, nullptr)
                ->add_keypoint(next_frame, next_keypoint_index);
        }
    }
}

void Frame::cache_local_features() {
    local_feature_backend()->cache_keyframe_features(image.get());
}

bool Frame::has_cached_local_features() {
    return local_feature_backend()->has_cached_features(image.get());
}

size_t Frame::recover_keypoints(Frame *reference_frame, Config *config) {
    if (!reference_frame || reference_frame == this)
        return 0;

    std::fprintf(stderr,
                 "[LighterGlueDiag] attempt current=%zu reference=%zu "
                 "current_kpts=%zu reference_kpts=%zu\n",
                 id(), reference_frame->id(), keypoint_num(),
                 reference_frame->keypoint_num());

    // Build the learned reference feature list directly from live XRSLAM
    // tracks. The backend samples a descriptor at each of these exact pixel
    // coordinates, so no learned-keypoint -> KLT-keypoint snapping is needed.
    std::vector<vector<2>> reference_track_points;
    std::vector<size_t> reference_frame_indices;
    reference_track_points.reserve(reference_frame->keypoint_num());
    reference_frame_indices.reserve(reference_frame->keypoint_num());
    for (size_t i = 0; i < reference_frame->keypoint_num(); ++i) {
        Track *track = reference_frame->get_track(i);
        if (!track || track->tag(TT_TRASH) || track->has_keypoint(this))
            continue;

        reference_track_points.emplace_back(
            apply_k(reference_frame->get_keypoint(i),
                    reference_frame->K));
        reference_frame_indices.emplace_back(i);
    }

    if (reference_track_points.size() < 5) {
        std::fprintf(stderr,
                     "[LighterGlueDiag] reject stage=reference_tracks "
                     "tracks=%zu\n",
                     reference_track_points.size());
        return 0;
    }

    auto backend = local_feature_backend();

    // Wide-baseline recovery is certified in XFeat's native feature space.
    // Once native XFeat + LighterGlue passes geometry, use its verified local
    // displacement field only to initialize XRSLAM's existing forward/backward
    // KLT at the exact persistent Track pixels. This keeps learned features as
    // the recovery/pose source while KLT remains the Track-identity verifier.
    std::vector<LocalFeatureMatch> native_matches;
    const bool native_matched =
        backend->match_keyframes(
            reference_frame->image.get(), image.get(), native_matches);

    // 0060 lazy learned-map diagnostic. Do not build every stride-10 anchor
    // in the sliding-window thread. Only an anchor that is actually selected
    // for recovery pays for learned-landmark construction.
    //
    // The anchor is triangulated from whichever optimized neighbor is
    // currently available: reference-10 and/or reference+10. The resulting
    // native XFeat feature-index -> world-XYZ table is persisted and reused
    // by subsequent recovery attempts. Learned and KLT identities remain
    // completely separate.
    std::optional<RecoveryLearnedAnchor> learned_anchor =
        get_recovery_learned_anchor(reference_frame->id());
    if (!learned_anchor.has_value())
        learned_anchor = RecoveryLearnedAnchor{};

    const auto reference_pose_opt =
        get_recovery_camera_pose_world(reference_frame->id());

    auto merge_learned_support =
        [&](size_t support_slot,
            size_t support_id,
            const char *support_name,
            bool retry_if_unavailable) {
            bool &attempted =
                support_slot == 0
                    ? learned_anchor->past_support_attempted
                    : learned_anchor->future_support_attempted;
            if (attempted)
                return;

            if (!reference_frame->map ||
                !reference_pose_opt.has_value()) {
                if (!retry_if_unavailable)
                    attempted = true;
                return;
            }

            const size_t support_index =
                reference_frame->map->frame_index_by_id(support_id);
            if (support_index == nil()) {
                if (!retry_if_unavailable)
                    attempted = true;
                return;
            }

            Frame *support_frame =
                reference_frame->map->get_frame(support_index);
            const auto support_pose_opt =
                get_recovery_camera_pose_world(support_id);
            if (!support_frame || !support_frame->image ||
                support_frame->image->width() <= 0 ||
                support_frame->image->height() <= 0 ||
                !support_pose_opt.has_value()) {
                if (!retry_if_unavailable)
                    attempted = true;
                return;
            }

            attempted = true;

            std::vector<LocalFeatureMatch> support_matches;
            const bool support_matched =
                backend->match_keyframes(
                    reference_frame->image.get(),
                    support_frame->image.get(),
                    support_matches);

            const size_t landmarks_before =
                learned_anchor->landmarks.size();
            size_t landmarks_added = 0;
            size_t landmarks_replaced = 0;

            if (support_matched) {
                learned_anchor->support_raw[support_slot] =
                    support_matches.size();

                const PoseState reference_pose =
                    *reference_pose_opt;
                const PoseState support_pose =
                    *support_pose_opt;

                const auto make_projection =
                    [](const PoseState &camera_pose) {
                        matrix<3, 4> P;
                        const matrix<3> R =
                            camera_pose.q.conjugate().matrix();
                        const vector<3> t =
                            -(R * camera_pose.p);
                        P << R, t;
                        return P;
                    };

                const matrix<3, 4> reference_projection =
                    make_projection(reference_pose);
                const matrix<3, 4> support_projection =
                    make_projection(support_pose);

                const matrix<3> R_10 =
                    support_pose.q.conjugate().matrix() *
                    reference_pose.q.matrix();
                const vector<3> t_10 =
                    support_pose.q.conjugate() *
                    (reference_pose.p - support_pose.p);

                matrix<3> t_cross;
                t_cross <<
                    0.0, -t_10.z(), t_10.y(),
                    t_10.z(), 0.0, -t_10.x(),
                    -t_10.y(), t_10.x(), 0.0;
                const matrix<3> known_E =
                    t_cross * R_10;

                const double support_focal =
                    0.25 *
                    (reference_frame->K(0, 0) +
                     reference_frame->K(1, 1) +
                     support_frame->K(0, 0) +
                     support_frame->K(1, 1));
                const double epipolar_threshold =
                    1.0 / std::max(1.0, support_focal);
                const double epipolar_threshold_sq =
                    epipolar_threshold *
                    epipolar_threshold;

                for (const auto &match : support_matches) {
                    if (match.reference_index ==
                        static_cast<size_t>(-1)) {
                        continue;
                    }

                    const vector<3> reference_bearing =
                        remove_k(match.reference_point,
                                 reference_frame->K);
                    const vector<3> support_bearing =
                        remove_k(match.current_point,
                                 support_frame->K);

                    const vector<2> reference_normalized =
                        reference_bearing.hnormalized();
                    const vector<2> support_normalized =
                        support_bearing.hnormalized();

                    const vector<3> x0{
                        reference_normalized.x(),
                        reference_normalized.y(), 1.0};
                    const vector<3> x1{
                        support_normalized.x(),
                        support_normalized.y(), 1.0};

                    const vector<3> Ex0 = known_E * x0;
                    const vector<3> Etx1 =
                        known_E.transpose() * x1;
                    const double numerator = x1.dot(Ex0);
                    const double denominator =
                        Ex0.x() * Ex0.x() +
                        Ex0.y() * Ex0.y() +
                        Etx1.x() * Etx1.x() +
                        Etx1.y() * Etx1.y();
                    if (denominator <= 1.0e-16)
                        continue;

                    const double sampson_sq =
                        numerator * numerator / denominator;
                    if (sampson_sq >
                        epipolar_threshold_sq) {
                        continue;
                    }
                    ++learned_anchor
                          ->support_epipolar[support_slot];

                    const vector<4> hlandmark =
                        triangulate_point(
                            reference_projection,
                            support_projection,
                            reference_bearing,
                            support_bearing);
                    if (std::abs(hlandmark.w()) <= 1.0e-12)
                        continue;

                    const vector<3> landmark =
                        hlandmark.hnormalized();
                    if (!landmark.allFinite())
                        continue;

                    const vector<3> reference_camera_point =
                        reference_pose.q.conjugate() *
                        (landmark - reference_pose.p);
                    const vector<3> support_camera_point =
                        support_pose.q.conjugate() *
                        (landmark - support_pose.p);
                    if (reference_camera_point.z() <= 1.0e-3 ||
                        support_camera_point.z() <= 1.0e-3) {
                        continue;
                    }

                    const double reference_distance =
                        (landmark - reference_pose.p).norm();
                    const double support_distance =
                        (landmark - support_pose.p).norm();
                    if (std::max(reference_distance,
                                 support_distance) > 50.0) {
                        continue;
                    }

                    const double reference_error =
                        (apply_k(reference_camera_point,
                                 reference_frame->K) -
                         match.reference_point)
                            .norm();
                    const double support_error =
                        (apply_k(support_camera_point,
                                 support_frame->K) -
                         match.current_point)
                            .norm();
                    const double max_reprojection =
                        std::max(reference_error,
                                 support_error);
                    if (max_reprojection > 2.0)
                        continue;

                    vector<3> reference_ray =
                        landmark - reference_pose.p;
                    vector<3> support_ray =
                        landmark - support_pose.p;
                    const double reference_ray_norm =
                        reference_ray.norm();
                    const double support_ray_norm =
                        support_ray.norm();
                    if (reference_ray_norm <= 1.0e-12 ||
                        support_ray_norm <= 1.0e-12) {
                        continue;
                    }
                    reference_ray /= reference_ray_norm;
                    support_ray /= support_ray_norm;

                    const double cosine =
                        std::max(
                            -1.0,
                            std::min(
                                1.0,
                                reference_ray.dot(support_ray)));
                    const double angle_deg =
                        std::acos(cosine) *
                        180.0 / M_PI;
                    if (angle_deg < 1.0)
                        continue;

                    ++learned_anchor
                          ->support_triangulated[support_slot];

                    bool replace = false;
                    const auto existing =
                        learned_anchor->landmarks.find(
                            match.reference_index);
                    if (existing ==
                        learned_anchor->landmarks.end()) {
                        replace = true;
                    } else {
                        const auto old_angle_it =
                            learned_anchor
                                ->landmark_angle_deg.find(
                                    match.reference_index);
                        const auto old_reprojection_it =
                            learned_anchor
                                ->landmark_reprojection_px.find(
                                    match.reference_index);

                        const double old_angle =
                            old_angle_it ==
                                    learned_anchor
                                        ->landmark_angle_deg.end()
                                ? 0.0
                                : old_angle_it->second;
                        const double old_reprojection =
                            old_reprojection_it ==
                                    learned_anchor
                                        ->landmark_reprojection_px.end()
                                ? std::numeric_limits<double>::infinity()
                                : old_reprojection_it->second;

                        replace =
                            angle_deg > old_angle + 1.0e-6 ||
                            (std::abs(angle_deg - old_angle) <=
                                 1.0e-6 &&
                             max_reprojection <
                                 old_reprojection);
                    }

                    if (!replace)
                        continue;

                    if (existing ==
                        learned_anchor->landmarks.end()) {
                        ++landmarks_added;
                    } else {
                        ++landmarks_replaced;
                    }

                    learned_anchor->landmarks[
                        match.reference_index] = landmark;
                    learned_anchor->landmark_angle_deg[
                        match.reference_index] = angle_deg;
                    learned_anchor->landmark_reprojection_px[
                        match.reference_index] =
                        max_reprojection;
                }
            }

            set_recovery_learned_anchor(
                reference_frame->id(), *learned_anchor);

            std::fprintf(
                stderr,
                "[LearnedMap] lazy anchor=%zu support=%s "
                "support_id=%zu matched=%d raw=%zu "
                "epipolar=%zu triangulated=%zu "
                "before=%zu after=%zu added=%zu replaced=%zu\n",
                reference_frame->id(), support_name,
                support_id, support_matched ? 1 : 0,
                learned_anchor->support_raw[support_slot],
                learned_anchor->support_epipolar[support_slot],
                learned_anchor
                    ->support_triangulated[support_slot],
                landmarks_before,
                learned_anchor->landmarks.size(),
                landmarks_added,
                landmarks_replaced);
        };

    // A past support can only disappear as time advances, so if it is not
    // available on the first recovery attempt there is no value in retrying.
    if (reference_frame->id() >= 10) {
        merge_learned_support(
            0, reference_frame->id() - 10,
            "past10", false);
    } else {
        learned_anchor->past_support_attempted = true;
    }

    // A future support may not have been optimized yet on the first recovery
    // frame (for example current=310, reference=300). Retry until it becomes
    // available, then evaluate it exactly once.
    merge_learned_support(
        1, reference_frame->id() + 10,
        "future10", true);

    // Persist the empty attempt state too, so repeated calls do not redo a
    // dead past-support lookup while still allowing future+10 to arrive.
    set_recovery_learned_anchor(
        reference_frame->id(), *learned_anchor);

    bool learned_pnp_pose_valid = false;
    matrix<3> learned_pnp_R_wc = matrix<3>::Identity();
    vector<3> learned_pnp_p_wc = vector<3>::Zero();
    size_t learned_pnp_inliers_for_track_audit = 0;
    std::vector<vector<3>> learned_pnp_inlier_landmarks;
    std::vector<vector<2>> learned_pnp_inlier_observations;

    if (native_matched && !native_matches.empty() &&
        learned_anchor.has_value()) {
        std::vector<size_t> source_feature_indices;
        source_feature_indices.reserve(
            learned_anchor->landmarks.size());
        for (const auto &entry : learned_anchor->landmarks)
            source_feature_indices.emplace_back(entry.first);
        std::sort(source_feature_indices.begin(),
                  source_feature_indices.end());
        if (source_feature_indices.size() > 128)
            source_feature_indices.resize(128);

        std::vector<cv::Point3f> pnp_points_3d;
        std::vector<cv::Point2f> pnp_points_2d;
        pnp_points_3d.reserve(source_feature_indices.size());
        pnp_points_2d.reserve(source_feature_indices.size());

        for (const auto &match : native_matches) {
            if (!std::binary_search(
                    source_feature_indices.begin(),
                    source_feature_indices.end(),
                    match.reference_index)) {
                continue;
            }

            const auto landmark_it =
                learned_anchor->landmarks.find(
                    match.reference_index);
            if (landmark_it ==
                learned_anchor->landmarks.end()) {
                continue;
            }

            const vector<3> &landmark =
                landmark_it->second;
            pnp_points_3d.emplace_back(
                static_cast<float>(landmark.x()),
                static_cast<float>(landmark.y()),
                static_cast<float>(landmark.z()));
            pnp_points_2d.emplace_back(
                static_cast<float>(match.current_point.x()),
                static_cast<float>(match.current_point.y()));
        }

        bool learned_pnp_solved = false;
        size_t learned_pnp_inliers = 0;
        double learned_pnp_ratio = 0.0;
        double pose_delta_translation =
            std::numeric_limits<double>::quiet_NaN();
        double pose_delta_rotation_deg =
            std::numeric_limits<double>::quiet_NaN();

        if (pnp_points_3d.size() >= 6) {
            cv::Mat camera_matrix =
                (cv::Mat_<double>(3, 3) <<
                    K(0, 0), 0.0, K(0, 2),
                    0.0, K(1, 1), K(1, 2),
                    0.0, 0.0, 1.0);
            cv::Mat rvec, tvec;
            cv::Mat pnp_inlier_indices;

            learned_pnp_solved = cv::solvePnPRansac(
                pnp_points_3d, pnp_points_2d,
                camera_matrix, cv::noArray(),
                rvec, tvec, false, 300, 2.448f,
                0.99, pnp_inlier_indices,
                cv::SOLVEPNP_EPNP);

            if (learned_pnp_solved) {
                learned_pnp_inliers =
                    static_cast<size_t>(
                        pnp_inlier_indices.total());
                learned_pnp_ratio =
                    static_cast<double>(
                        learned_pnp_inliers) /
                    pnp_points_3d.size();

                cv::Mat R_cw_cv;
                cv::Rodrigues(rvec, R_cw_cv);
                matrix<3> R_wc;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        R_wc(row, col) =
                            R_cw_cv.at<double>(col, row);
                    }
                }

                vector<3> t_cw;
                t_cw <<
                    tvec.at<double>(0, 0),
                    tvec.at<double>(1, 0),
                    tvec.at<double>(2, 0);
                const vector<3> p_wc =
                    -(R_wc * t_cw);

                learned_pnp_pose_valid = true;
                learned_pnp_R_wc = R_wc;
                learned_pnp_p_wc = p_wc;
                learned_pnp_inliers_for_track_audit =
                    learned_pnp_inliers;

                learned_pnp_inlier_landmarks.reserve(
                    learned_pnp_inliers);
                learned_pnp_inlier_observations.reserve(
                    learned_pnp_inliers);
                for (int inlier_row = 0;
                     inlier_row < pnp_inlier_indices.rows;
                     ++inlier_row) {
                    const int correspondence_index =
                        pnp_inlier_indices.at<int>(
                            inlier_row, 0);
                    if (correspondence_index < 0 ||
                        static_cast<size_t>(
                            correspondence_index) >=
                            pnp_points_3d.size() ||
                        static_cast<size_t>(
                            correspondence_index) >=
                            pnp_points_2d.size()) {
                        continue;
                    }

                    const cv::Point3f &point_3d =
                        pnp_points_3d[
                            correspondence_index];
                    const cv::Point2f &point_2d =
                        pnp_points_2d[
                            correspondence_index];
                    learned_pnp_inlier_landmarks.emplace_back(
                        point_3d.x, point_3d.y, point_3d.z);
                    learned_pnp_inlier_observations.emplace_back(
                        point_2d.x, point_2d.y);
                }

                if (!learned_pnp_inlier_landmarks.empty() &&
                    learned_pnp_inlier_landmarks.size() ==
                        learned_pnp_inlier_observations.size()) {
                    RecoveryLearnedObservationSet observation_set;
                    observation_set.reference_frame_id =
                        reference_frame->id();
                    observation_set.landmarks_world =
                        learned_pnp_inlier_landmarks;
                    observation_set.observations_pixel =
                        learned_pnp_inlier_observations;
                    observation_set.pnp_inliers =
                        learned_pnp_inlier_landmarks.size();
                    observation_set.pnp_ratio =
                        learned_pnp_ratio;
                    observation_set.pnp_pose_valid = true;
                    observation_set.pnp_q_wc =
                        quaternion(learned_pnp_R_wc);
                    observation_set.pnp_q_wc.normalize();
                    observation_set.pnp_p_wc =
                        learned_pnp_p_wc;
                    set_recovery_learned_observations(
                        id(), observation_set);
                }

                const PoseState predicted_pose =
                    get_pose(camera);
                pose_delta_translation =
                    (p_wc - predicted_pose.p).norm();

                const matrix<3> R_delta =
                    R_wc.transpose() *
                    predicted_pose.q.matrix();
                const double trace_cosine =
                    std::max(
                        -1.0,
                        std::min(
                            1.0,
                            (R_delta.trace() - 1.0) * 0.5));
                pose_delta_rotation_deg =
                    std::acos(trace_cosine) *
                    180.0 / M_PI;
            }
        }

        std::fprintf(
            stderr,
            "[LighterGlueDiag] learned_pnp "
            "current=%zu reference=%zu "
            "landmarks=%zu source=%zu "
            "lg_2d3d=%zu solved=%d "
            "pnp_inliers=%zu ratio=%.3f "
            "pose_delta_t=%.4f pose_delta_r_deg=%.3f "
            "past10=%zu/%zu/%zu "
            "future10=%zu/%zu/%zu cache=1 "
            "source_mode=full_match_then_top128_filter\n",
            id(), reference_frame->id(),
            learned_anchor->landmarks.size(),
            source_feature_indices.size(),
            pnp_points_3d.size(),
            learned_pnp_solved ? 1 : 0,
            learned_pnp_inliers,
            learned_pnp_ratio,
            pose_delta_translation,
            pose_delta_rotation_deg,
            learned_anchor->support_raw[0],
            learned_anchor->support_epipolar[0],
            learned_anchor->support_triangulated[0],
            learned_anchor->support_raw[1],
            learned_anchor->support_epipolar[1],
            learned_anchor->support_triangulated[1]);
    } else {
        std::fprintf(
            stderr,
            "[LighterGlueDiag] learned_pnp "
            "current=%zu reference=%zu "
            "landmarks=%zu source=0 lg_2d3d=0 solved=0 "
            "pnp_inliers=0 ratio=0.000 "
            "pose_delta_t=nan pose_delta_r_deg=nan "
            "cache=%d native_matched=%d native_matches=%zu\n",
            id(), reference_frame->id(),
            learned_anchor.has_value()
                ? learned_anchor->landmarks.size()
                : 0,
            learned_anchor.has_value() ? 1 : 0,
            native_matched ? 1 : 0,
            native_matches.size());
    }

    // 0063 diagnostic: validate both the fixed-world factor convention and
    // the local optimization basin before these factors are allowed into the
    // real XRSLAM solver.
    //
    // Solve the exact same learned factors twice on copies of the body pose:
    //   (1) from XRSLAM/VIO,
    //   (2) from the independently estimated learned-PnP pose.
    //
    // If the factor model is consistent, the raw PnP pose must itself have a
    // low reprojection RMSE under this factor. If both starts converge to the
    // same solution/cost basin, the factor is suitable for later in-solver
    // integration. No real Frame/Track/solver state is modified here.
    if (learned_pnp_pose_valid &&
        learned_pnp_inlier_landmarks.size() >= 5 &&
        learned_pnp_inlier_landmarks.size() ==
            learned_pnp_inlier_observations.size()) {
        const PoseState vio_body_pose = pose;
        const PoseState vio_camera_pose = get_pose(camera);

        quaternion learned_pnp_q_wc(learned_pnp_R_wc);
        learned_pnp_q_wc.normalize();

        PoseState pnp_body_pose;
        pnp_body_pose.q =
            learned_pnp_q_wc * camera.q_cs.conjugate();
        pnp_body_pose.q.normalize();
        pnp_body_pose.p =
            learned_pnp_p_wc -
            pnp_body_pose.q * camera.p_cs;

        const double vio_initial_rmse =
            learned_reprojection_rmse_px(
                vio_body_pose, camera, K,
                learned_pnp_inlier_landmarks,
                learned_pnp_inlier_observations);
        const double pnp_initial_rmse =
            learned_reprojection_rmse_px(
                pnp_body_pose, camera, K,
                learned_pnp_inlier_landmarks,
                learned_pnp_inlier_observations);

        struct LearnedPoseSolveResult {
            PoseState body_pose;
            PoseState camera_pose;
            bool usable = false;
            double final_rmse =
                std::numeric_limits<double>::quiet_NaN();
            double final_cost =
                std::numeric_limits<double>::quiet_NaN();
        };

        const auto solve_learned_pose =
            [&](const PoseState &initial_body_pose) {
                LearnedPoseSolveResult result;
                result.body_pose = initial_body_pose;

                ceres::Problem::Options problem_options;
                problem_options.cost_function_ownership =
                    ceres::TAKE_OWNERSHIP;
                problem_options.loss_function_ownership =
                    ceres::DO_NOT_TAKE_OWNERSHIP;
                problem_options.local_parameterization_ownership =
                    ceres::DO_NOT_TAKE_OWNERSHIP;
                ceres::Problem problem(problem_options);

                QuaternionParameterization
                    quaternion_parameterization;
                ceres::CauchyLoss learned_loss(2.448);

                problem.AddParameterBlock(
                    result.body_pose.q.coeffs().data(), 4,
                    &quaternion_parameterization);
                problem.AddParameterBlock(
                    result.body_pose.p.data(), 3);

                for (size_t i = 0;
                     i < learned_pnp_inlier_landmarks.size();
                     ++i) {
                    auto *cost =
                        new ceres::AutoDiffCostFunction<
                            LearnedFixedWorldReprojectionCost,
                            2, 4, 3>(
                            new LearnedFixedWorldReprojectionCost(
                                learned_pnp_inlier_landmarks[i],
                                learned_pnp_inlier_observations[i],
                                K, camera));
                    problem.AddResidualBlock(
                        cost, &learned_loss,
                        result.body_pose.q.coeffs().data(),
                        result.body_pose.p.data());
                }

                ceres::Solver::Options solver_options;
                solver_options.linear_solver_type =
                    ceres::DENSE_QR;
                solver_options.trust_region_strategy_type =
                    ceres::DOGLEG;
                solver_options.max_num_iterations = 15;
                solver_options.max_solver_time_in_seconds =
                    0.005;
                solver_options.num_threads = 1;
                solver_options
                    .minimizer_progress_to_stdout = false;
                solver_options
                    .update_state_every_iteration = true;

                ceres::Solver::Summary solver_summary;
                ceres::Solve(
                    solver_options, &problem,
                    &solver_summary);

                result.usable =
                    solver_summary.IsSolutionUsable();
                result.final_cost =
                    solver_summary.final_cost;
                result.final_rmse =
                    learned_reprojection_rmse_px(
                        result.body_pose, camera, K,
                        learned_pnp_inlier_landmarks,
                        learned_pnp_inlier_observations);

                result.camera_pose.q =
                    result.body_pose.q * camera.q_cs;
                result.camera_pose.p =
                    result.body_pose.p +
                    result.body_pose.q * camera.p_cs;
                return result;
            };

        const LearnedPoseSolveResult vio_solve =
            solve_learned_pose(vio_body_pose);
        const LearnedPoseSolveResult pnp_solve =
            solve_learned_pose(pnp_body_pose);

        const double vio_solve_delta_vio_t =
            (vio_solve.camera_pose.p -
             vio_camera_pose.p)
                .norm();
        const double vio_solve_delta_vio_r =
            pose_rotation_delta_deg(
                vio_camera_pose.q,
                vio_solve.camera_pose.q);
        const double vio_solve_delta_pnp_t =
            (vio_solve.camera_pose.p -
             learned_pnp_p_wc)
                .norm();
        const double vio_solve_delta_pnp_r =
            pose_rotation_delta_deg(
                learned_pnp_q_wc,
                vio_solve.camera_pose.q);

        const double pnp_solve_delta_vio_t =
            (pnp_solve.camera_pose.p -
             vio_camera_pose.p)
                .norm();
        const double pnp_solve_delta_vio_r =
            pose_rotation_delta_deg(
                vio_camera_pose.q,
                pnp_solve.camera_pose.q);
        const double pnp_solve_delta_pnp_t =
            (pnp_solve.camera_pose.p -
             learned_pnp_p_wc)
                .norm();
        const double pnp_solve_delta_pnp_r =
            pose_rotation_delta_deg(
                learned_pnp_q_wc,
                pnp_solve.camera_pose.q);

        const double basin_delta_t =
            (vio_solve.camera_pose.p -
             pnp_solve.camera_pose.p)
                .norm();
        const double basin_delta_r =
            pose_rotation_delta_deg(
                vio_solve.camera_pose.q,
                pnp_solve.camera_pose.q);

        std::fprintf(
            stderr,
            "[LighterGlueDiag] learned_factor_basin "
            "current=%zu reference=%zu factors=%zu "
            "pnp_rmse_px=%.3f "
            "vio_usable=%d vio_initial_rmse_px=%.3f "
            "vio_final_rmse_px=%.3f vio_final_cost=%.6f "
            "vio_delta_vio_t=%.4f vio_delta_vio_r_deg=%.3f "
            "vio_delta_pnp_t=%.4f vio_delta_pnp_r_deg=%.3f "
            "pnp_usable=%d pnp_final_rmse_px=%.3f "
            "pnp_final_cost=%.6f "
            "pnp_delta_vio_t=%.4f pnp_delta_vio_r_deg=%.3f "
            "pnp_delta_pnp_t=%.4f pnp_delta_pnp_r_deg=%.3f "
            "basin_delta_t=%.4f basin_delta_r_deg=%.3f "
            "state_mutation=0\n",
            id(), reference_frame->id(),
            learned_pnp_inlier_landmarks.size(),
            pnp_initial_rmse,
            vio_solve.usable ? 1 : 0,
            vio_initial_rmse,
            vio_solve.final_rmse,
            vio_solve.final_cost,
            vio_solve_delta_vio_t,
            vio_solve_delta_vio_r,
            vio_solve_delta_pnp_t,
            vio_solve_delta_pnp_r,
            pnp_solve.usable ? 1 : 0,
            pnp_solve.final_rmse,
            pnp_solve.final_cost,
            pnp_solve_delta_vio_t,
            pnp_solve_delta_vio_r,
            pnp_solve_delta_pnp_t,
            pnp_solve_delta_pnp_r,
            basin_delta_t,
            basin_delta_r);
    }

    // 0061 diagnostic: use the learned PnP pose only as a geometric
    // initial-flow hypothesis for ordinary persistent XRSLAM tracks that the
    // normal high-rate frontend failed to carry into this frame.
    //
    // The source is the immediately preceding feature-map frame, not the
    // learned reference anchor. Therefore this tests the intended handoff:
    // learned wide-baseline recovery establishes a coarse current pose, then
    // short-baseline strict bidirectional KLT restores normal Track identity.
    //
    // No Track is reconnected here and no estimator pose is changed.
    if (learned_pnp_pose_valid && reference_frame->map &&
        reference_frame->map->frame_num() > 0) {
        Frame *track_source_frame =
            reference_frame->map->get_frame(
                reference_frame->map->frame_num() - 1);

        size_t source_tracks = 0;
        size_t already_current = 0;
        size_t exact_world = 0;
        size_t projected_visible = 0;
        size_t within_lk_motion_limit = 0;

        std::vector<size_t> seed_source_indices;
        std::vector<vector<2>> seed_source_points;
        std::vector<vector<2>> seed_current_points;
        std::vector<vector<2>> pnp_projected_points;
        std::vector<double> pnp_vio_projection_delta;

        if (track_source_frame && track_source_frame != this &&
            track_source_frame->image &&
            track_source_frame->image->width() > 0 &&
            track_source_frame->image->height() > 0) {
            const PoseState predicted_camera = get_pose(camera);

            seed_source_indices.reserve(
                track_source_frame->keypoint_num());
            seed_source_points.reserve(
                track_source_frame->keypoint_num());
            seed_current_points.reserve(
                track_source_frame->keypoint_num());
            pnp_projected_points.reserve(
                track_source_frame->keypoint_num());
            pnp_vio_projection_delta.reserve(
                track_source_frame->keypoint_num());

            for (size_t source_index = 0;
                 source_index < track_source_frame->keypoint_num();
                 ++source_index) {
                Track *track =
                    track_source_frame->get_track(source_index);
                if (!track || track->tag(TT_TRASH))
                    continue;

                ++source_tracks;
                if (track->has_keypoint(this)) {
                    ++already_current;
                    continue;
                }
                if (!track->recovery_landmark_world.has_value())
                    continue;
                ++exact_world;

                const vector<3> landmark =
                    *track->recovery_landmark_world;
                const vector<3> pnp_camera_point =
                    learned_pnp_R_wc.transpose() *
                    (landmark - learned_pnp_p_wc);
                if (!pnp_camera_point.allFinite() ||
                    pnp_camera_point.z() <= 1.0e-3 ||
                    pnp_camera_point.z() > 50.0) {
                    continue;
                }

                const vector<2> projected_current =
                    apply_k(pnp_camera_point, K);
                if (projected_current.x() < 20.0 ||
                    projected_current.y() < 20.0 ||
                    projected_current.x() >= image->width() - 20.0 ||
                    projected_current.y() >= image->height() - 20.0) {
                    continue;
                }

                const vector<2> source_point =
                    apply_k(
                        track_source_frame->get_keypoint(source_index),
                        track_source_frame->K);
                if (source_point.x() < 20.0 ||
                    source_point.y() < 20.0 ||
                    source_point.x() >=
                        track_source_frame->image->width() - 20.0 ||
                    source_point.y() >=
                        track_source_frame->image->height() - 20.0) {
                    continue;
                }

                ++projected_visible;
                const double source_to_projection =
                    (projected_current - source_point).norm();
                if (source_to_projection <=
                    track_source_frame->image->height() / 4.0) {
                    ++within_lk_motion_limit;
                }

                // Log how different the learned-PnP seed is from XRSLAM's
                // already-existing current pose prediction. This is
                // diagnostic only; the KLT below always receives the learned
                // PnP projection.
                const vector<3> vio_camera_point =
                    predicted_camera.q.conjugate() *
                    (landmark - predicted_camera.p);
                if (vio_camera_point.z() > 1.0e-3 &&
                    vio_camera_point.z() <= 50.0) {
                    const vector<2> vio_projection =
                        apply_k(vio_camera_point, K);
                    if (vio_projection.allFinite()) {
                        pnp_vio_projection_delta.emplace_back(
                            (projected_current -
                             vio_projection)
                                .norm());
                    }
                }

                seed_source_indices.emplace_back(source_index);
                seed_source_points.emplace_back(source_point);
                seed_current_points.emplace_back(projected_current);
                pnp_projected_points.emplace_back(projected_current);
            }
        }

        std::vector<char> projected_klt_status;
        if (!seed_source_points.empty()) {
            backend->track_keypoints(
                track_source_frame->image.get(),
                image.get(),
                seed_source_points,
                seed_current_points,
                projected_klt_status);
        }

        size_t klt_ok = 0;
        size_t reprojection_2px = 0;
        size_t reprojection_4px = 0;
        size_t reprojection_8px = 0;
        std::vector<double> klt_from_pnp_shift;
        klt_from_pnp_shift.reserve(seed_source_points.size());

        for (size_t i = 0;
             i < seed_source_points.size() &&
             i < projected_klt_status.size() &&
             i < seed_current_points.size() &&
             i < pnp_projected_points.size();
             ++i) {
            if (!projected_klt_status[i])
                continue;

            ++klt_ok;
            const double reprojection_error =
                (seed_current_points[i] -
                 pnp_projected_points[i])
                    .norm();
            klt_from_pnp_shift.emplace_back(reprojection_error);

            if (reprojection_error <= 2.0)
                ++reprojection_2px;
            if (reprojection_error <= 4.0)
                ++reprojection_4px;
            if (reprojection_error <= 8.0)
                ++reprojection_8px;
        }

        const auto median_of =
            [](std::vector<double> values) {
                if (values.empty())
                    return std::numeric_limits<double>::quiet_NaN();
                std::sort(values.begin(), values.end());
                const size_t middle = values.size() / 2;
                if (values.size() % 2 != 0)
                    return values[middle];
                return 0.5 *
                       (values[middle - 1] + values[middle]);
            };

        const double median_klt_shift =
            median_of(klt_from_pnp_shift);
        const double median_pnp_vio_projection_delta =
            median_of(pnp_vio_projection_delta);

        std::fprintf(
            stderr,
            "[LighterGlueDiag] pnp_track_reacquire "
            "current=%zu source=%zu reference=%zu "
            "learned_pnp_inliers=%zu "
            "source_tracks=%zu already_current=%zu "
            "exact_world=%zu visible=%zu seeds=%zu "
            "within_lk_motion=%zu klt_ok=%zu "
            "reproj2=%zu reproj4=%zu reproj8=%zu "
            "median_klt_shift_px=%.3f "
            "median_pnp_vio_delta_px=%.3f "
            "state_mutation=0\n",
            id(),
            track_source_frame ? track_source_frame->id() : nil(),
            reference_frame->id(),
            learned_pnp_inliers_for_track_audit,
            source_tracks,
            already_current,
            exact_world,
            projected_visible,
            seed_source_points.size(),
            within_lk_motion_limit,
            klt_ok,
            reprojection_2px,
            reprojection_4px,
            reprojection_8px,
            median_klt_shift,
            median_pnp_vio_projection_delta);
    }

    if (native_matched && native_matches.size() >= 5) {
        std::vector<vector<2>> native_reference_points;
        std::vector<vector<2>> native_current_points;
        std::vector<size_t> native_valid_indices;
        native_reference_points.reserve(native_matches.size());
        native_current_points.reserve(native_matches.size());
        native_valid_indices.reserve(native_matches.size());

        for (size_t match_index = 0;
             match_index < native_matches.size(); ++match_index) {
            const auto &match = native_matches[match_index];
            if (match.reference_point.x() < 20.0 ||
                match.reference_point.y() < 20.0 ||
                match.reference_point.x() >=
                    reference_frame->image->width() - 20.0 ||
                match.reference_point.y() >=
                    reference_frame->image->height() - 20.0 ||
                match.current_point.x() < 20.0 ||
                match.current_point.y() < 20.0 ||
                match.current_point.x() >= image->width() - 20.0 ||
                match.current_point.y() >= image->height() - 20.0) {
                continue;
            }

            native_reference_points.emplace_back(
                remove_k(match.reference_point,
                         reference_frame->K)
                    .hnormalized());
            native_current_points.emplace_back(
                remove_k(match.current_point, K).hnormalized());
            native_valid_indices.emplace_back(match_index);
        }

        if (native_reference_points.size() >= 5) {
            std::vector<char> native_inlier_mask;
            const double native_focal =
                0.25 *
                (reference_frame->K(0, 0) +
                 reference_frame->K(1, 1) +
                 K(0, 0) + K(1, 1));
            const matrix<3> native_E = find_essential_matrix(
                native_reference_points, native_current_points,
                native_inlier_mask,
                1.0 / std::max(1.0, native_focal),
                0.999, 1000, config->random());

            size_t native_inliers = 0;
            for (char inlier : native_inlier_mask)
                native_inliers += inlier != 0;

            const double native_ratio =
                native_reference_points.empty()
                    ? 0.0
                    : static_cast<double>(native_inliers) /
                          native_reference_points.size();
            const size_t min_native_inliers =
                std::max<size_t>(
                    20, config->initializer_min_matches() / 2);

            std::fprintf(
                stderr,
                "[LighterGlueDiag] native_xfeat matches=%zu "
                "border_valid=%zu ransac_inliers=%zu ratio=%.3f "
                "required_inliers=%zu required_ratio=0.500\n",
                native_matches.size(),
                native_reference_points.size(),
                native_inliers, native_ratio,
                min_native_inliers);

            if (native_inliers >= min_native_inliers &&
                native_ratio >= 0.5) {
                std::vector<size_t> verified_native_indices;
                verified_native_indices.reserve(native_inliers);
                for (size_t i = 0;
                     i < native_inlier_mask.size() &&
                     i < native_valid_indices.size(); ++i) {
                    if (native_inlier_mask[i])
                        verified_native_indices.emplace_back(
                            native_valid_indices[i]);
                }

                // Diagnostic: use the CURRENT-frame metric pose prediction to
                // project synchronized 3D landmarks, then associate those
                // projections only to native XFeat correspondences that
                // already passed the 2D-2D Essential gate. PnP remains
                // diagnostic-only in 0050.
                const size_t reference_keypoint_count =
                    reference_frame->keypoint_num();

                size_t eligible_landmark_tracks = 0;
                std::vector<char> eligible_track(
                    reference_keypoint_count, 0);
                for (size_t reference_index = 0;
                     reference_index < reference_keypoint_count;
                     ++reference_index) {
                    Track *track =
                        reference_frame->get_track(reference_index);
                    if (!track || track->tag(TT_TRASH) ||
                        track->has_keypoint(this) ||
                        !track->recovery_landmark_world.has_value()) {
                        continue;
                    }
                    eligible_track[reference_index] = 1;
                    ++eligible_landmark_tracks;
                }

                size_t projected_visible = 0;
                size_t projected_within_2px = 0;
                size_t projected_within_4px = 0;
                size_t projected_within_8px = 0;

                std::vector<size_t> best_track_for_native(
                    native_matches.size(), nil());
                std::vector<double> best_distance_sq_for_native(
                    native_matches.size(), 1.0e30);

                if (config->has_imu()) {
                    const PoseState predicted_camera =
                        get_pose(camera);

                    for (size_t reference_index = 0;
                         reference_index <
                             reference_keypoint_count;
                         ++reference_index) {
                        if (!eligible_track[reference_index])
                            continue;

                        Track *track =
                            reference_frame->get_track(
                                reference_index);
                        if (!track)
                            continue;

                        const vector<3> landmark =
                            *track->recovery_landmark_world;
                        const vector<3> camera_point =
                            predicted_camera.q.conjugate() *
                            (landmark - predicted_camera.p);
                        if (camera_point.z() <= 1.0e-3 ||
                            camera_point.z() > 50.0) {
                            continue;
                        }

                        const vector<2> projected_point =
                            apply_k(camera_point, K);
                        if (projected_point.x() < 20.0 ||
                            projected_point.y() < 20.0 ||
                            projected_point.x() >=
                                image->width() - 20.0 ||
                            projected_point.y() >=
                                image->height() - 20.0) {
                            continue;
                        }
                        ++projected_visible;

                        size_t nearest_native_index = nil();
                        double nearest_distance_sq = 1.0e30;
                        for (size_t native_index :
                             verified_native_indices) {
                            const double distance_sq =
                                (native_matches[native_index]
                                     .current_point -
                                 projected_point)
                                    .squaredNorm();
                            if (distance_sq <
                                nearest_distance_sq) {
                                nearest_distance_sq =
                                    distance_sq;
                                nearest_native_index =
                                    native_index;
                            }
                        }

                        if (nearest_native_index == nil())
                            continue;

                        if (nearest_distance_sq <= 4.0)
                            ++projected_within_2px;
                        if (nearest_distance_sq <= 16.0)
                            ++projected_within_4px;
                        if (nearest_distance_sq <= 64.0)
                            ++projected_within_8px;

                        if (nearest_distance_sq <= 64.0 &&
                            nearest_distance_sq <
                                best_distance_sq_for_native[
                                    nearest_native_index]) {
                            best_distance_sq_for_native[
                                nearest_native_index] =
                                nearest_distance_sq;
                            best_track_for_native[
                                nearest_native_index] =
                                reference_index;
                        }
                    }
                }

                std::vector<cv::Point3f> pnp_points_3d;
                std::vector<cv::Point2f> pnp_points_2d;
                pnp_points_3d.reserve(
                    verified_native_indices.size());
                pnp_points_2d.reserve(
                    verified_native_indices.size());

                for (size_t native_index :
                     verified_native_indices) {
                    if (native_index >=
                            best_track_for_native.size() ||
                        best_track_for_native[native_index] ==
                            nil()) {
                        continue;
                    }

                    Track *track =
                        reference_frame->get_track(
                            best_track_for_native[native_index]);
                    if (!track)
                        continue;

                    const vector<3> landmark =
                        *track->recovery_landmark_world;
                    const vector<2> normalized =
                        remove_k(
                            native_matches[native_index]
                                .current_point,
                            K)
                            .hnormalized();

                    pnp_points_3d.emplace_back(
                        static_cast<float>(landmark.x()),
                        static_cast<float>(landmark.y()),
                        static_cast<float>(landmark.z()));
                    pnp_points_2d.emplace_back(
                        static_cast<float>(normalized.x()),
                        static_cast<float>(normalized.y()));
                }

                bool pnp_solved = false;
                size_t pnp_inliers = 0;
                double pnp_ratio = 0.0;

                if (pnp_points_3d.size() >= 6) {
                    cv::Mat pnp_camera =
                        cv::Mat::eye(3, 3, CV_64F);
                    cv::Mat pnp_rvec, pnp_tvec;
                    cv::Mat pnp_inlier_indices;

                    const double pnp_threshold =
                        1.0 /
                        std::max(
                            1.0,
                            0.5 * (K(0, 0) +
                                   K(1, 1)));

                    pnp_solved = cv::solvePnPRansac(
                        pnp_points_3d,
                        pnp_points_2d,
                        pnp_camera,
                        cv::noArray(),
                        pnp_rvec,
                        pnp_tvec,
                        false,
                        1000,
                        static_cast<float>(
                            pnp_threshold),
                        0.999,
                        pnp_inlier_indices,
                        cv::SOLVEPNP_EPNP);

                    if (pnp_solved) {
                        pnp_inliers =
                            static_cast<size_t>(
                                pnp_inlier_indices.total());
                        pnp_ratio =
                            pnp_points_3d.empty()
                                ? 0.0
                                : static_cast<double>(
                                      pnp_inliers) /
                                      pnp_points_3d.size();
                    }
                }

                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] projected_pnp "
                    "native_inliers=%zu eligible_landmarks=%zu "
                    "visible=%zu near2=%zu near4=%zu near8=%zu "
                    "unique8=%zu solved=%d "
                    "pnp_inliers=%zu ratio=%.3f imu=%d\n",
                    native_inliers,
                    eligible_landmark_tracks,
                    projected_visible,
                    projected_within_2px,
                    projected_within_4px,
                    projected_within_8px,
                    pnp_points_3d.size(),
                    pnp_solved ? 1 : 0,
                    pnp_inliers,
                    pnp_ratio,
                    config->has_imu() ? 1 : 0);

                // Diagnostic: revisit the original same-image bridge now
                // that both reference/current learned keypoints come from the
                // coordinate-exact official XFeat export. The old snap
                // experiment was contaminated by the third-party sparse model
                // whose keypoint geometry differed from official XFeat by
                // many pixels.
                //
                // We do NOT reconnect anything here. For each radius, map a
                // verified native reference feature to the nearest existing
                // persistent XRSLAM reference Track, enforce one native point
                // per Track, then evaluate the substituted XRSLAM reference
                // pixel under the already-estimated native Essential matrix.
                // Reusing native_E avoids extra RANSAC calls and therefore
                // avoids perturbing config->random() / the trajectory.
                constexpr double snap_radii[] = {1.0, 2.0, 3.0, 5.0, 8.0};
                for (double snap_radius : snap_radii) {
                    const double snap_radius_sq =
                        snap_radius * snap_radius;
                    std::vector<size_t> best_native_for_track(
                        reference_track_points.size(), nil());
                    std::vector<double> best_distance_sq_for_track(
                        reference_track_points.size(), 1.0e30);
                    size_t native_with_neighbor = 0;

                    for (size_t native_index : verified_native_indices) {
                        const vector<2> &native_reference_point =
                            native_matches[native_index].reference_point;

                        size_t nearest_track_index = nil();
                        double nearest_distance_sq = snap_radius_sq;

                        for (size_t track_index = 0;
                             track_index < reference_track_points.size();
                             ++track_index) {
                            const double distance_sq =
                                (reference_track_points[track_index] -
                                 native_reference_point)
                                    .squaredNorm();
                            if (distance_sq <= nearest_distance_sq) {
                                nearest_distance_sq = distance_sq;
                                nearest_track_index = track_index;
                            }
                        }

                        if (nearest_track_index == nil())
                            continue;
                        ++native_with_neighbor;

                        if (nearest_distance_sq <
                            best_distance_sq_for_track[nearest_track_index]) {
                            best_distance_sq_for_track[nearest_track_index] =
                                nearest_distance_sq;
                            best_native_for_track[nearest_track_index] =
                                native_index;
                        }
                    }

                    size_t unique_tracks = 0;
                    size_t sampson_inliers = 0;
                    double distance_sum = 0.0;
                    double distance_max = 0.0;
                    const double normalized_threshold =
                        1.0 / std::max(1.0, native_focal);
                    const double normalized_threshold_sq =
                        normalized_threshold * normalized_threshold;

                    for (size_t track_index = 0;
                         track_index < best_native_for_track.size();
                         ++track_index) {
                        const size_t native_index =
                            best_native_for_track[track_index];
                        if (native_index == nil())
                            continue;

                        ++unique_tracks;
                        const double snap_distance =
                            std::sqrt(
                                best_distance_sq_for_track[track_index]);
                        distance_sum += snap_distance;
                        distance_max =
                            std::max(distance_max, snap_distance);

                        const vector<2> reference_normalized =
                            remove_k(reference_track_points[track_index],
                                     reference_frame->K)
                                .hnormalized();
                        const vector<2> current_normalized =
                            remove_k(native_matches[native_index].current_point,
                                     K)
                                .hnormalized();

                        vector<3> x0{reference_normalized.x(),
                                     reference_normalized.y(), 1.0};
                        vector<3> x1{current_normalized.x(),
                                     current_normalized.y(), 1.0};

                        const vector<3> Ex0 = native_E * x0;
                        const vector<3> Etx1 = native_E.transpose() * x1;
                        const double numerator = x1.dot(Ex0);
                        const double denominator =
                            Ex0.x() * Ex0.x() + Ex0.y() * Ex0.y() +
                            Etx1.x() * Etx1.x() + Etx1.y() * Etx1.y();

                        if (denominator <= 1.0e-16)
                            continue;

                        const double sampson_sq =
                            numerator * numerator / denominator;
                        if (sampson_sq <= normalized_threshold_sq)
                            ++sampson_inliers;
                    }

                    const double snap_ratio =
                        unique_tracks == 0
                            ? 0.0
                            : static_cast<double>(sampson_inliers) /
                                  unique_tracks;
                    const double mean_snap =
                        unique_tracks == 0
                            ? 0.0
                            : distance_sum / unique_tracks;
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] anchor_snap "
                        "radius=%.1f native_inliers=%zu "
                        "neighbor=%zu unique_tracks=%zu "
                        "sampson_inliers=%zu ratio=%.3f "
                        "mean_snap=%.3f max_snap=%.3f\n",
                        snap_radius, native_inliers, native_with_neighbor,
                        unique_tracks, sampson_inliers, snap_ratio, mean_snap,
                        distance_max);
                }

                const char *diagnostic_only =
                    std::getenv("XRSLAM_RECOVERY_DIAGNOSTIC_ONLY");
                if (diagnostic_only && diagnostic_only[0] == '1' &&
                    diagnostic_only[1] == '\0') {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] recovery diagnostic-only; "
                        "skip state-changing reacquisition\n");
                    return 0;
                }

                std::vector<size_t> seed_reference_indices;
                std::vector<vector<2>> seed_reference_points;
                std::vector<vector<2>> seed_current_points;
                seed_reference_indices.reserve(
                    reference_frame->keypoint_num());
                seed_reference_points.reserve(
                    reference_frame->keypoint_num());
                seed_current_points.reserve(
                    reference_frame->keypoint_num());

                // Interpolate a local flow seed from the three nearest
                // geometrically verified native XFeat correspondences. There
                // is no acceptance radius here: the subsequent bidirectional
                // KLT and essential-matrix checks decide validity.
                for (size_t reference_index = 0;
                     reference_index < reference_frame->keypoint_num();
                     ++reference_index) {
                    Track *track =
                        reference_frame->get_track(reference_index);
                    if (!track || track->tag(TT_TRASH) ||
                        track->has_keypoint(this)) {
                        continue;
                    }

                    const vector<2> reference_point =
                        apply_k(reference_frame->get_keypoint(
                                    reference_index),
                                reference_frame->K);

                    size_t nearest[3] = {nil(), nil(), nil()};
                    double nearest_distance_sq[3] = {
                        1.0e30, 1.0e30, 1.0e30};

                    for (size_t native_index :
                         verified_native_indices) {
                        const double distance_sq =
                            (native_matches[native_index]
                                 .reference_point -
                             reference_point)
                                .squaredNorm();

                        for (size_t slot = 0; slot < 3; ++slot) {
                            if (distance_sq >=
                                nearest_distance_sq[slot]) {
                                continue;
                            }
                            for (size_t shift = 2;
                                 shift > slot; --shift) {
                                nearest_distance_sq[shift] =
                                    nearest_distance_sq[shift - 1];
                                nearest[shift] =
                                    nearest[shift - 1];
                            }
                            nearest_distance_sq[slot] = distance_sq;
                            nearest[slot] = native_index;
                            break;
                        }
                    }

                    vector<2> predicted_flow = vector<2>::Zero();
                    double weight_sum = 0.0;
                    for (size_t slot = 0; slot < 3; ++slot) {
                        if (nearest[slot] == nil())
                            continue;
                        const auto &native_match =
                            native_matches[nearest[slot]];
                        const double weight =
                            1.0 /
                            (std::sqrt(nearest_distance_sq[slot]) +
                             1.0);
                        predicted_flow +=
                            weight *
                            (native_match.current_point -
                             native_match.reference_point);
                        weight_sum += weight;
                    }
                    if (weight_sum <= 0.0)
                        continue;

                    const vector<2> predicted_current =
                        reference_point +
                        predicted_flow / weight_sum;
                    if (predicted_current.x() < 20.0 ||
                        predicted_current.y() < 20.0 ||
                        predicted_current.x() >=
                            image->width() - 20.0 ||
                        predicted_current.y() >=
                            image->height() - 20.0) {
                        continue;
                    }

                    seed_reference_indices.emplace_back(
                        reference_index);
                    seed_reference_points.emplace_back(
                        reference_point);
                    seed_current_points.emplace_back(
                        predicted_current);
                }

                std::vector<char> klt_status;
                if (!seed_reference_points.empty()) {
                    backend->track_keypoints(
                        reference_frame->image.get(), image.get(),
                        seed_reference_points, seed_current_points,
                        klt_status);
                }

                struct ReacquiredCandidate {
                    size_t reference_index;
                    vector<2> current_point;
                    vector<2> reference_normalized;
                    vector<2> current_normalized;
                };

                std::vector<ReacquiredCandidate> reacquired;
                reacquired.reserve(seed_reference_points.size());
                for (size_t i = 0;
                     i < seed_reference_points.size() &&
                     i < klt_status.size(); ++i) {
                    if (!klt_status[i])
                        continue;
                    if (seed_current_points[i].x() < 20.0 ||
                        seed_current_points[i].y() < 20.0 ||
                        seed_current_points[i].x() >=
                            image->width() - 20.0 ||
                        seed_current_points[i].y() >=
                            image->height() - 20.0) {
                        continue;
                    }

                    reacquired.push_back(
                        {seed_reference_indices[i],
                         seed_current_points[i],
                         remove_k(seed_reference_points[i],
                                  reference_frame->K)
                             .hnormalized(),
                         remove_k(seed_current_points[i], K)
                             .hnormalized()});
                }

                if (reacquired.size() < 5) {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] native_reacquire "
                        "native_inliers=%zu seeds=%zu klt_ok=%zu "
                        "recovered=0 reject=klt\n",
                        native_inliers,
                        seed_reference_points.size(),
                        reacquired.size());
                    return 0;
                }

                std::vector<vector<2>> reacquired_reference;
                std::vector<vector<2>> reacquired_current;
                reacquired_reference.reserve(reacquired.size());
                reacquired_current.reserve(reacquired.size());
                for (const auto &candidate : reacquired) {
                    reacquired_reference.emplace_back(
                        candidate.reference_normalized);
                    reacquired_current.emplace_back(
                        candidate.current_normalized);
                }

                std::vector<char> reacquired_inlier_mask;
                find_essential_matrix(
                    reacquired_reference, reacquired_current,
                    reacquired_inlier_mask,
                    1.0 / std::max(1.0, native_focal),
                    0.999, 1000, config->random());

                size_t reacquired_inliers = 0;
                for (char inlier : reacquired_inlier_mask)
                    reacquired_inliers += inlier != 0;

                if (reacquired_inliers < 5) {
                    std::fprintf(
                        stderr,
                        "[LighterGlueDiag] native_reacquire "
                        "native_inliers=%zu seeds=%zu klt_ok=%zu "
                        "geometric=%zu recovered=0 reject=geometry\n",
                        native_inliers,
                        seed_reference_points.size(),
                        reacquired.size(), reacquired_inliers);
                    return 0;
                }

                std::vector<size_t> accepted;
                accepted.reserve(reacquired_inliers);
                for (size_t i = 0;
                     i < reacquired.size() &&
                     i < reacquired_inlier_mask.size(); ++i) {
                    if (reacquired_inlier_mask[i])
                        accepted.emplace_back(i);
                }

                std::stable_sort(
                    accepted.begin(), accepted.end(),
                    [&](size_t a, size_t b) {
                        Track *track_a = reference_frame->get_track(
                            reacquired[a].reference_index);
                        Track *track_b = reference_frame->get_track(
                            reacquired[b].reference_index);
                        const size_t length_a =
                            track_a ? track_a->keypoint_num() : 0;
                        const size_t length_b =
                            track_b ? track_b->keypoint_num() : 0;
                        return length_a > length_b;
                    });

                PoissonDiskFilter<2> filter(
                    config->feature_tracker_min_keypoint_distance());
                for (const auto &bearing : bearings)
                    filter.preset_point(apply_k(bearing, K));

                std::vector<char> used_reference(
                    reference_frame->keypoint_num(), 0);
                size_t recovered = 0;
                for (size_t candidate_index : accepted) {
                    const auto &candidate =
                        reacquired[candidate_index];
                    if (candidate.reference_index >=
                            used_reference.size() ||
                        used_reference[candidate.reference_index]) {
                        continue;
                    }

                    Track *track = reference_frame->get_track(
                        candidate.reference_index);
                    if (!track || track->tag(TT_TRASH) ||
                        track->has_keypoint(this)) {
                        continue;
                    }
                    if (!filter.insert_point(
                            candidate.current_point)) {
                        continue;
                    }

                    const size_t current_index = keypoint_num();
                    append_keypoint(
                        remove_k(candidate.current_point, K));
                    track->add_keypoint(this, current_index);
                    used_reference[candidate.reference_index] = 1;
                    ++recovered;
                }

                std::fprintf(
                    stderr,
                    "[LighterGlueDiag] native_reacquire "
                    "native_inliers=%zu seeds=%zu klt_ok=%zu "
                    "geometric=%zu recovered=%zu\n",
                    native_inliers,
                    seed_reference_points.size(),
                    reacquired.size(), reacquired_inliers,
                    recovered);

                if (recovered > 0) {
                    log_info(
                        "XFeat+LighterGlue/KLT recovery: "
                        "%zu native inliers, %zu KLT inliers, "
                        "%zu tracks reconnected",
                        native_inliers, reacquired_inliers,
                        recovered);
                }
                return recovered;
            }
        } else {
            std::fprintf(
                stderr,
                "[LighterGlueDiag] native_xfeat matches=%zu "
                "border_valid=%zu reject=border\n",
                native_matches.size(),
                native_reference_points.size());
        }
    } else {
        std::fprintf(
            stderr,
            "[LighterGlueDiag] native_xfeat matched=%d matches=%zu\n",
            native_matched ? 1 : 0, native_matches.size());
    }
    std::vector<LocalFeatureMatch> matches;
    const bool matched =
        backend->match_track_keypoints(
            reference_frame->image.get(), reference_track_points,
            image.get(), matches);
    std::fprintf(
        stderr,
        "[LighterGlueDiag] matcher matched=%d raw_matches=%zu "
        "reference_tracks=%zu\n",
        matched ? 1 : 0, matches.size(),
        reference_track_points.size());
    if (!matched || matches.size() < 5) {
        std::fprintf(stderr,
                     "[LighterGlueDiag] reject stage=matcher "
                     "raw_matches=%zu\n",
                     matches.size());
        return 0;
    }

    struct Candidate {
        size_t reference_index;
        vector<2> current_point;
        vector<2> reference_normalized;
        vector<2> current_normalized;
        double confidence;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(matches.size());
    for (const auto &match : matches) {
        if (match.reference_index >= reference_frame_indices.size())
            continue;
        if (match.current_point.x() < 20.0 ||
            match.current_point.y() < 20.0 ||
            match.current_point.x() >= image->width() - 20.0 ||
            match.current_point.y() >= image->height() - 20.0) {
            continue;
        }

        const size_t reference_index =
            reference_frame_indices[match.reference_index];
        Track *track = reference_frame->get_track(reference_index);
        if (!track || track->tag(TT_TRASH) || track->has_keypoint(this))
            continue;

        const vector<3> reference_bearing =
            reference_frame->get_keypoint(reference_index);
        const vector<3> current_bearing =
            remove_k(match.current_point, K);

        candidates.push_back(
            {reference_index,
             match.current_point,
             reference_bearing.hnormalized(),
             current_bearing.hnormalized(),
             match.confidence});
    }

    std::fprintf(stderr,
                 "[LighterGlueDiag] direct_candidates=%zu/%zu\n",
                 candidates.size(), matches.size());

    if (candidates.size() < 5) {
        std::fprintf(stderr,
                     "[LighterGlueDiag] reject stage=direct_candidates "
                     "candidates=%zu\n",
                     candidates.size());
        return 0;
    }

    std::vector<vector<2>> reference_points;
    std::vector<vector<2>> current_points;
    reference_points.reserve(candidates.size());
    current_points.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        reference_points.emplace_back(candidate.reference_normalized);
        current_points.emplace_back(candidate.current_normalized);
    }

    std::vector<char> inlier_mask;
    const double focal =
        0.25 * (reference_frame->K(0, 0) + reference_frame->K(1, 1) +
                K(0, 0) + K(1, 1));
    find_essential_matrix(reference_points, current_points, inlier_mask,
                          1.0 / std::max(1.0, focal), 0.999, 1000,
                          config->random());

    size_t inlier_count = 0;
    for (char inlier : inlier_mask)
        inlier_count += inlier != 0;

    const size_t min_inliers =
        std::max<size_t>(20, config->initializer_min_matches() / 2);
    const double inlier_ratio =
        candidates.empty()
            ? 0.0
            : static_cast<double>(inlier_count) / candidates.size();

    std::fprintf(stderr,
                 "[LighterGlueDiag] ransac inliers=%zu/%zu ratio=%.3f "
                 "required_inliers=%zu required_ratio=0.500\n",
                 inlier_count, candidates.size(), inlier_ratio,
                 min_inliers);

    if (inlier_count < min_inliers || inlier_ratio < 0.5) {
        std::fprintf(stderr,
                     "[LighterGlueDiag] reject stage=ransac\n");
        return 0;
    }

    std::vector<size_t> accepted;
    accepted.reserve(inlier_count);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i < inlier_mask.size() && inlier_mask[i])
            accepted.emplace_back(i);
    }

    std::stable_sort(
        accepted.begin(), accepted.end(),
        [&](size_t a, size_t b) {
            Track *track_a =
                reference_frame->get_track(candidates[a].reference_index);
            Track *track_b =
                reference_frame->get_track(candidates[b].reference_index);
            const size_t length_a = track_a ? track_a->keypoint_num() : 0;
            const size_t length_b = track_b ? track_b->keypoint_num() : 0;
            if (length_a != length_b)
                return length_a > length_b;
            return candidates[a].confidence > candidates[b].confidence;
        });

    PoissonDiskFilter<2> filter(
        config->feature_tracker_min_keypoint_distance());
    for (const auto &bearing : bearings)
        filter.preset_point(apply_k(bearing, K));

    std::vector<char> used_reference(reference_frame->keypoint_num(), 0);
    size_t recovered = 0;
    for (size_t candidate_index : accepted) {
        const Candidate &candidate = candidates[candidate_index];
        if (candidate.reference_index >= used_reference.size() ||
            used_reference[candidate.reference_index]) {
            continue;
        }

        Track *track =
            reference_frame->get_track(candidate.reference_index);
        if (!track || track->tag(TT_TRASH) || track->has_keypoint(this))
            continue;
        if (!filter.insert_point(candidate.current_point))
            continue;

        const size_t current_index = keypoint_num();
        append_keypoint(remove_k(candidate.current_point, K));
        track->add_keypoint(this, current_index);
        used_reference[candidate.reference_index] = 1;
        ++recovered;
    }

    std::fprintf(stderr,
                 "[LighterGlueDiag] result recovered=%zu "
                 "ransac_inliers=%zu candidates=%zu\n",
                 recovered, inlier_count, candidates.size());

    if (recovered > 0) {
        log_info(
            "LighterGlue recovery: %zu/%zu geometric inliers, "
            "%zu tracks reconnected",
            inlier_count, candidates.size(), recovered);
    }
    return recovered;
}

PoseState Frame::get_pose(const ExtrinsicParams &sensor) const {
    PoseState result;
    result.q = pose.q * sensor.q_cs;
    result.p = pose.p + pose.q * sensor.p_cs;
    return result;
}

void Frame::set_pose(const ExtrinsicParams &sensor, const PoseState &pose) {
    this->pose.q = pose.q * sensor.q_cs.conjugate();
    this->pose.p = pose.p - this->pose.q * sensor.p_cs;
}

std::unique_lock<std::mutex> Frame::lock() const {
    if (map) {
        return map->lock();
    } else {
        return {};
    }
}

} // namespace xrslam
