# xrslam_ros2 — Jazzy bootstrap adapter

Standalone **ROS 2** / `ament_cmake` package. It intentionally does not use
`xrslam-ros` (ROS 1), RTAB-Map, Nav2 libraries, wheel encoders, or occupancy
mapping. Only the XRSLAM C API is linked; do not add this package to the
XRSLAM superbuild.

**Status: source implementation only. Not yet compiled on ROS 2 Jazzy or tested
with Leroy. Motors must remain disabled until calibration, pose conventions,
timestamp alignment and loss/recovery behavior pass hardware validation.**

## Prerequisites

1. Build XRSLAM independently, including its existing C API (`libxrslam.so`).
2. Source a ROS 2 Jazzy environment with `ament_cmake`, `rclcpp`,
   `tf2_ros`, the common ROS messages, and `yaml-cpp`.
3. Obtain a **robot-calibrated** SLAM YAML and sensor/noise template YAML.
   The EuRoC example sensor file is NOT a substitute for Leroy's noise or
   temporal calibration. The URDF/robot_state_publisher owns extrinsics.
4. Use `ros2 topic list -t`, `ros2 topic info -v /imu/data_raw`, and
   `ros2 topic info -v <candidate-image-topic>` to identify the actual
   rectified image and matching CameraInfo topics. Both must use the optical
   frame ID that is in the TF tree.

Example workspace (adjust paths):

```sh
source /opt/ros/jazzy/setup.bash
mkdir -p ~/xrslam_ws/src
ln -s /home/kevin/Code/xrslam/xrslam_ros2 ~/xrslam_ws/src/xrslam_ros2
cd ~/xrslam_ws
colcon build --packages-select xrslam_ros2 --cmake-args \
  -DXRSLAM_ROOT=/home/kevin/Code/xrslam \
  -DXRSLAM_LIBRARY=/home/kevin/Code/xrslam/lib/libxrslam.so
source install/setup.bash
export LD_LIBRARY_PATH=/home/kevin/Code/xrslam/lib:"${LD_LIBRARY_PATH}"
```

Choose parameters explicitly; no placeholder transforms or guessed time offsets:

```sh
ros2 run xrslam_ros2 xrslam_ros2_node --ros-args \
  -p slam_yaml:=/absolute/path/leroy_slam.yaml \
  -p device_template_yaml:=/absolute/path/leroy_sensor_calibrated.yaml \
  -p camera_imu_offset_sec:=0.0 \
  -p image_topic:=/actual/rectified/image/topic \
  -p camera_info_topic:=/matching/camera_info/topic \
  -p camera_optical_frame:=camera_optical_frame \
  -p imu_frame:=imu_link
```

The `0.0` offset above is only an example: use it **only when measured and
verified**. `camera_imu_offset_sec` has no default. XRSLAM initializes only
after valid CameraInfo, `base_link <- imu_link` and
`base_link <- camera_optical_frame` TF lookups. For rectified monocular input,
the **P matrix** from CameraInfo is used and XRSLAM distortion is disabled;
stereo/projection offsets are rejected. Input uses header timestamps, gyro and
accelerometer at the same stamp, and a bounded camera queue gated on IMU data.

ROS outputs: `/xrslam/tracking_status` (String),
`/xrslam/localization_valid` (latched Bool),
`/xrslam/pose` (PoseStamped), `/odom` (Odometry) and
`odom -> base_link` (dynamic TF), *only after safety gates pass*.

By default `pose_convention_validated=false`: the node **does not publish
odometry** before the nonidentity-transform pose-direction test is performed.
Only after verifying on the robot that `XRSLAM_RESULT_BODY_POSE` represents
the robot body in the local VIO frame should that parameter be set to true.
Velocity is finite-differenced from consecutive body poses, **not directly
measured**. Covariance is deliberately conservative/uninformative (variance
floors of 100 in SI units), not estimator-derived; calibrate uncertainty and
velocity before motion control. There must be exactly one publisher of
`odom -> base_link`.

On tracking failure, clock/IMU discontinuity or a pose jump, odometry output is
**latched off** until an externally verified session alignment or a supervised
full node restart; this first adapter intentionally does not claim automatic
continuity-preserving recovery. No `map -> odom` is published.

## Acceptance work still required

- Jazzy `colcon build` and replay test with synchronized camera/IMU
- Nonidentity quaternion and translation fixture verifying body pose and TF
  directions (`q_bc`, `p_bc`, `q_bi`, `p_bi`)
- Hardware-measured timestamp offsets, time-domain/clock reset handling
- Validate Euler/quaternion and ROS `odom` child-frame twist conventions
- Verify tracking-loss latch and continuity over synthetic session restart
- Check TF authority and no competing `odom -> base_link` publisher
- EuRoC healthy/dropout regression and semantic/dense reconstruction parity

This package does not implement persistent-map save/load, semantic-keyframe
search, global `map -> odom`, Nav2 occupancy-grid generation, or navigation.
