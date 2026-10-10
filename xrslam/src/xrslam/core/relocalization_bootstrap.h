#ifndef XRSLAM_RELOCALIZATION_BOOTSTRAP_H
#define XRSLAM_RELOCALIZATION_BOOTSTRAP_H

#include <xrslam/core/relocalization_consensus.h>
#include <xrslam/core/sliding_window_tracker.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/map/track.h>

#include <cmath>
#include <memory>
#include <unordered_set>

namespace xrslam {

inline PoseState relocalized_body_pose(
    const PoseState &camera_pose, const ExtrinsicParams &camera) {
    PoseState body;
    body.q = (camera_pose.q * camera.q_cs.conjugate()).normalized();
    body.p = camera_pose.p - body.q * camera.p_cs;
    return body;
}

// Construct a NEW, isolated SLAM map in the historical session's local
// coordinates. No archived frame/track pointers are inserted in the new map.
// The first observed frame is fixed as the gauge anchor. Following frames
// must obtain temporal visual tracks through the ordinary KLT frontend.
inline std::unique_ptr<SlidingWindowTracker>
make_relocalization_bootstrap(
    Map *feature_map, const RelocalizationConfirmedSeed &seed,
    const MotionState &previous_motion,
    const std::shared_ptr<Config> &config,
    size_t &seeded_landmarks) {
    seeded_landmarks = 0;
    const auto &a = seed.first;
    const auto &b = seed.second;
    if (!feature_map || !config || seed.dt_s <= 0.0 ||
        a.session_id == 0 || a.session_id != b.session_id ||
        a.current_frame_id >= b.current_frame_id ||
        a.landmarks.size() < 8 || b.landmarks.size() < 8)
        return nullptr;

    const size_t index = feature_map->frame_index_by_id(a.current_frame_id);
    if (index == nil()) return nullptr;
    Frame *source = feature_map->get_frame(index);
    if (!source || !source->image || source->id() != a.current_frame_id)
        return nullptr;
    const PoseState first_body = relocalized_body_pose(a.camera_pose, source->camera);
    const PoseState second_body = relocalized_body_pose(b.camera_pose, source->camera);
    if (!first_body.p.allFinite() || !second_body.p.allFinite() ||
        !first_body.q.coeffs().allFinite() || !second_body.q.coeffs().allFinite())
        return nullptr;

    std::unique_ptr<Map> map = std::make_unique<Map>();
    map->attach_frame(source->clone());
    Frame *anchor = map->get_frame(0);
    anchor->pose = first_body;
    anchor->motion = previous_motion;
    anchor->motion.v = (second_body.p - first_body.p) / seed.dt_s;
    if (!anchor->motion.v.allFinite() || !anchor->motion.bg.allFinite() ||
        !anchor->motion.ba.allFinite()) return nullptr;
    anchor->tag(FT_KEYFRAME) = true;
    anchor->tag(FT_NO_TRANSLATION) = false;
    anchor->tag(FT_FIX_POSE) = true;
    anchor->tag(FT_FIX_MOTION) = true;

    std::unordered_set<size_t> seen_keypoints;
    for (const auto &landmark : a.landmarks) {
        if (landmark.current_keypoint_index >= anchor->keypoint_num() ||
            !landmark.world_point.allFinite() ||
            !seen_keypoints.insert(landmark.current_keypoint_index).second)
            continue;
        const PoseState cam = anchor->get_pose(anchor->camera);
        const vector<3> offset = landmark.world_point - cam.p;
        const vector<3> point_camera = cam.q.conjugate() * offset;
        const vector<3> bearing = anchor->get_keypoint(
            landmark.current_keypoint_index);
        if (!point_camera.allFinite() || point_camera.z() <= 0.1 ||
            !std::isfinite(offset.norm()) || offset.norm() > 50.0 ||
            offset.norm() < 0.1 ||
            (point_camera.normalized() - bearing).norm() > 0.03)
            continue;
        Track *track = anchor->get_track(
            landmark.current_keypoint_index, map.get());
        track->set_landmark_point(landmark.world_point);
        track->tag(TT_VALID) = true;
        track->tag(TT_TRIANGULATED) = true;
        track->tag(TT_STATIC) = true;
        track->tag(TT_FIX_INVD) = true;
        ++seeded_landmarks;
    }
    if (seeded_landmarks < 8) return nullptr;
    return std::make_unique<SlidingWindowTracker>(std::move(map), config);
}

inline size_t count_relocalization_mapped_tracks(const Frame *frame) {
    if (!frame) return 0;
    size_t result = 0;
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Track *track = frame->get_track(i);
        if (track && track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC))
            ++result;
    }
    return result;
}

inline double relocalization_pose_disagreement_deg(
    const quaternion &a, const quaternion &b) {
    const quaternion delta = a.conjugate() * b;
    return 2.0 * std::acos(std::min(1.0, std::max(0.0,
        std::abs(delta.normalized().w())))) *
        (180.0 / 3.14159265358979323846);
}

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_BOOTSTRAP_H
