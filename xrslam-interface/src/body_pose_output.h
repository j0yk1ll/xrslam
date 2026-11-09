#ifndef XRSLAM_INTERFACE_BODY_POSE_OUTPUT_H
#define XRSLAM_INTERFACE_BODY_POSE_OUTPUT_H

#include "XRSLAM.h"
#include <xrslam/xrslam.h>

namespace xrslam {

// The supplied pose is already the configured output-frame pose from
// Detail::predict_pose (including output.q_bo / output.p_bo).
// For Leroy, those output extrinsics are identity, so it is T_world_body.
// Do not apply IMU-to-body extrinsics a second time.
inline void CopyBodyPoseToResult(const Pose &body_pose,
                                 double timestamp,
                                 XRSLAMPose *result) {
    result->timestamp = timestamp;
    for (int i = 0; i < 3; ++i)
        result->translation[i] = body_pose.p(i);
    result->quaternion[0] = body_pose.q.x();
    result->quaternion[1] = body_pose.q.y();
    result->quaternion[2] = body_pose.q.z();
    result->quaternion[3] = body_pose.q.w();
}

}  // namespace xrslam

#endif  // XRSLAM_INTERFACE_BODY_POSE_OUTPUT_H
