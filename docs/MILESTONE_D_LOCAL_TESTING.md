# Milestone D: verified persistent map localization (draft)

This change introduces a **separate** XRSLAM map-frame alignment status, read using `XRSLAMGetVerifiedMapAlignment`, and a gated ROS 2 publisher for `map -> odom`. It must not be conflated with `XRSLAM_RESULT_GLOBAL_BODY_POSE`, which can fall back to local poses.

## Map identity and startup

Set `XRSLAM_REFERENCE_MAP_PATH` and `XRSLAM_MAP_ID` to enable persistent map storage. If the reference map is **absent**, setting `XRSLAM_MAP_CREATE=1` explicitly defines its first successfully initialized VIO session as the origin of a *new map*. This is coordinate-frame establishment, not relocalization. To localize in a saved map, leave `XRSLAM_MAP_CREATE` unset, use the same `XRSLAM_MAP_ID`, and point to the saved archive.

Saved references now use file format v2, which stores an optional provenance-backed map alignment per session. Version 1 archives are readable but **do not** acquire map alignment by guessing identity.

For a saved map, a subsequent VIO session only becomes globally valid after geometric PnP agreement with a reference session with an established map alignment, plus the 20-step bootstrap follow-up (at least 15 good steps). This is an initial gate, **not** proof of adequate accuracy for driving. If no session carries alignment provenance, the map status stays invalid.

## ROS adapter

Pass `-p enable_map_tf:=true -p map_id:=<XRSLAM_MAP_ID>` and optionally `-p map_frame:=map`. The adapter continues publishing its local `odom -> base_link`. It publishes `map -> odom` only if map identity, session epoch and timestamp gates hold. Additional status topics are `/xrslam/map_localization_valid` and `/xrslam/map_localization_status`. A false validity signal must inhibit motion; TF consumers must not interpret a stale transform as a valid localization.

**Caution:** The first ROS adapter already latches off local odometry after a tracking-loss/session-origin jump. It therefore does not yet provide unattended recovery and resumption across sessions. This draft also does not generate a Nav2 occupancy grid or call Nav2.

## Local validation checklist (motors disconnected)

1. Compile and run the existing EuRoC healthy/dropout regression with semantics disabled. Verify baseline trajectory parity and one recovery.
2. Run `colcon build --packages-select xrslam_ros2` with the built XRSLAM `libxrslam.so`. Verify the symbol with `nm -D lib/libxrslam.so | grep XRSLAMGetVerifiedMapAlignment`.
3. Inspect live topics and static URDF TF. Verify optical-camera and IMU extrinsics with *nonidentity* transforms. Test the rectified CameraInfo projection and calibrated IMU-camera timing.
4. With `enable_map_tf=false`, verify that no `map -> odom` is published.
5. With a **new, empty** map file, explicit `XRSLAM_MAP_CREATE=1`, and the robot stationary, verify valid becomes true only after VIO initialization and that the new map is saved on clean shutdown.
6. Restart with the *same* `XRSLAM_MAP_ID`, `XRSLAM_MAP_CREATE` unset. Expect `map_localization_valid=false` until historical PnP relocalization and post-bootstrap verification succeed. Wrong map identity, old-format provenance, bad checksum, and missing reference descriptors must not trigger a valid map transform.
7. Inject a dropout/recovery, verify false validity during loss and **exactly one** publisher each for `odom -> base_link` and `map -> odom`. Inspect frame direction against a nonzero yaw/translation fixture. Do not enable motors.

## Known gaps to resolve before marking complete

- Full Jazzy `colcon` and XRSLAM native builds and replay tests have **not been run** here.
- Validate that the current estimator's asynchronous pose timestamp and frame-origin behavior meet the ROS TF contract.
- The 20-frame follow-up and inherited reference alignment require replay and cross-session validation. The ROS adapter intentionally remains fail-closed after local tracking loss.
- Verify persistence of semantic masks separately (Milestone B/C remains incomplete), and add a ROS 2 semantic query interface.
- Keep Nav2 out of the XRSLAM build and maintain a separate occupancy-grid provider.
