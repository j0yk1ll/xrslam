#include <cmath>

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include "body_pose_output.h"

namespace {

TEST(BodyPoseOutput, PreservesNonidentityBodyPoseAndTimestamp) {
    xrslam::Pose body;
    body.q = Eigen::Quaterniond(
        Eigen::AngleAxisd(0.43, Eigen::Vector3d(0.3, -0.4, 0.7).normalized()));
    body.p = xrslam::vector<3>(1.2, -2.3, 0.8);
    XRSLAMPose output{};
    xrslam::CopyBodyPoseToResult(body, 42.25, &output);

    EXPECT_DOUBLE_EQ(output.timestamp, 42.25);
    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output.translation[i], body.p(i));
    EXPECT_NEAR(output.quaternion[0], body.q.x(), 1e-15);
    EXPECT_NEAR(output.quaternion[1], body.q.y(), 1e-15);
    EXPECT_NEAR(output.quaternion[2], body.q.z(), 1e-15);
    EXPECT_NEAR(output.quaternion[3], body.q.w(), 1e-15);

    // A nonzero/nonidentity IMU extrinsic must NOT leak into BODY_POSE.
    const Eigen::Quaterniond q_bi(
        Eigen::AngleAxisd(-0.38, Eigen::Vector3d::UnitY()));
    const xrslam::vector<3> p_bi(0.095, -0.095, 0.02);
    const xrslam::vector<3> incorrectly_shifted = body.p + body.q * p_bi;
    const Eigen::Quaterniond incorrectly_rotated = body.q * q_bi;
    EXPECT_GT((incorrectly_shifted - body.p).norm(), 0.1);
    EXPECT_GT(incorrectly_rotated.angularDistance(body.q), 0.3);
    EXPECT_NEAR((xrslam::vector<3>(output.translation[0],
                                  output.translation[1],
                                  output.translation[2]) - body.p).norm(),
                0.0, 1e-14);
}

TEST(BodyPoseOutput, CameraCompositionRemainsIndependent) {
    xrslam::Pose body;
    body.q = Eigen::Quaterniond(
        Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()));
    body.p = xrslam::vector<3>(0.4, 1.1, -0.2);
    const Eigen::Quaterniond q_bc(
        Eigen::AngleAxisd(0.52, Eigen::Vector3d::UnitY()));
    const xrslam::vector<3> p_bc(0.18, 0.04, 0.12);

    XRSLAMPose output{};
    xrslam::CopyBodyPoseToResult(body, 3.0, &output);
    const Eigen::Quaterniond out_q(output.quaternion[3],
                                    output.quaternion[0],
                                    output.quaternion[1],
                                    output.quaternion[2]);
    const xrslam::vector<3> out_p(output.translation[0],
                                  output.translation[1],
                                  output.translation[2]);
    EXPECT_NEAR(out_q.angularDistance(body.q), 0.0, 1e-14);
    const xrslam::vector<3> expected_camera_p = body.p + body.q * p_bc;
    const xrslam::vector<3> camera_p = out_p + out_q * p_bc;
    EXPECT_NEAR((camera_p - expected_camera_p).norm(), 0.0, 1e-14);
    EXPECT_NEAR((out_q * q_bc).angularDistance(body.q * q_bc), 0.0, 1e-14);
}

}  // namespace
