#ifndef XRSLAM_RELOCALIZATION_TRACKING_HEALTH_H
#define XRSLAM_RELOCALIZATION_TRACKING_HEALTH_H

#include <xrslam/geometry/stereo.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/track.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace xrslam {

// Passive tracking-health observation. This code never changes feature tags,
// map contents, IMU state, estimator poses, or lifecycle state. Measurements
// are from the CURRENT camera frame, not keyframe totals or archived tracks.
struct RelocalizationTrackingHealth {
    size_t frame_id = 0;
    double timestamp = 0.0;
    size_t detected = 0;
    size_t associated = 0;
    size_t mapped = 0;
    size_t reprojectable = 0;
    size_t inliers_3px = 0;
    size_t inliers_6px = 0;
    double median_reprojection_px =
        std::numeric_limits<double>::quiet_NaN();
    bool pose_finite = false;
};

inline RelocalizationTrackingHealth measure_relocalization_tracking_health(
    Frame *frame) {
    RelocalizationTrackingHealth result;
    if (!frame) return result;
    result.frame_id = frame->id();
    result.timestamp = frame->image ? frame->image->t : 0.0;
    result.detected = frame->keypoint_num();
    result.pose_finite = frame->pose.p.allFinite() &&
        frame->pose.q.coeffs().allFinite() &&
        frame->pose.q.squaredNorm() > 1.0e-12;
    if (!result.pose_finite) return result;

    const PoseState camera = frame->get_pose(frame->camera);
    if (!camera.p.allFinite() || !camera.q.coeffs().allFinite()) {
        result.pose_finite = false;
        return result;
    }

    std::vector<double> reprojection_errors;
    reprojection_errors.reserve(frame->keypoint_num());
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Track *track = frame->get_track(i);
        if (!track) continue;
        ++result.associated;
        if (!track->all_tagged(TT_VALID, TT_TRIANGULATED, TT_STATIC) ||
            !std::isfinite(track->landmark.inv_depth) ||
            track->landmark.inv_depth <= 1.0e-8)
            continue;
        ++result.mapped;

        const vector<3> point_world = track->get_landmark_point();
        if (!point_world.allFinite()) continue;
        const vector<3> point_camera =
            camera.q.conjugate() * (point_world - camera.p);
        if (!point_camera.allFinite() || point_camera.z() <= 1.0e-6)
            continue;
        const vector<2> projected = apply_k(point_camera, frame->K);
        const vector<2> observed =
            apply_k(frame->get_keypoint(i), frame->K);
        if (!projected.allFinite() || !observed.allFinite()) continue;
        const double error = (projected - observed).norm();
        if (!std::isfinite(error)) continue;
        reprojection_errors.emplace_back(error);
        ++result.reprojectable;
        if (error <= 3.0) ++result.inliers_3px;
        if (error <= 6.0) ++result.inliers_6px;
    }
    if (!reprojection_errors.empty()) {
        const size_t middle = reprojection_errors.size() / 2;
        std::nth_element(reprojection_errors.begin(),
                         reprojection_errors.begin() + middle,
                         reprojection_errors.end());
        result.median_reprojection_px = reprojection_errors[middle];
    }
    return result;
}

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_TRACKING_HEALTH_H
