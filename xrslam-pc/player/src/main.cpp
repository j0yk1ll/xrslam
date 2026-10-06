#include <argparse.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <dataset_reader.h>
#include <trajectory_writer.h>
#include <opencv2/opencv.hpp>

#include "XRSLAM.h"
#include "opencv_painter.h"
#include "visualizer.h"

std::shared_ptr<Viewer> viewer = nullptr;
cv::Mat feature_tracker_cvimage;
bool headless = false;

void GetShowElements() {
    if (headless) {
        return;
    }

    XRSLAMIntrinsics intrinsics;
    XRSLAMGetResult(XRSLAM_INFO_INTRINSICS, &intrinsics);
    Eigen::Vector4f intrinsics_v(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy);

    XRSLAMPose pose_c;
    XRSLAMGetResult(XRSLAM_RESULT_CAMERA_POSE, &pose_c);
    Eigen::Matrix4f pose_c_m = Eigen::Matrix4f::Identity();
    pose_c_m.block<3, 3>(0, 0) = Eigen::Quaternionf(
        pose_c.quaternion[3], 
        pose_c.quaternion[0],
        pose_c.quaternion[1], 
        pose_c.quaternion[2]).toRotationMatrix();
    pose_c_m.block<3, 1>(0, 3) = Eigen::Vector3f(
        pose_c.translation[0], 
        pose_c.translation[1],
        pose_c.translation[2]);

    XRSLAMLandmarks landmarks;
    XRSLAMGetResult(XRSLAM_RESULT_LANDMARKS, &landmarks);
    std::vector<Eigen::Vector3f> points;
    points.reserve(landmarks.num_landmarks);
    for (int i = 0; i < landmarks.num_landmarks; ++i) {
        XRSLAMLandmark &landmark = landmarks.landmarks[i];
        points.emplace_back(landmark.x, landmark.y, landmark.z);
    }

    VisData::Frame frame = VisData::Frame(cv::Mat(), intrinsics_v, pose_c_m);

    viewer->data()->update_frames({frame});
    viewer->data()->update_points(points);
    viewer->data()->update_poses(pose_c_m);
    viewer->data()->update_image(feature_tracker_cvimage);

}

typedef std::tuple<double, XRSLAMAcceleration, XRSLAMGyroscope> IMUData;

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("XRSLAM Player");
    program.add_argument("-sc", "--slamconfig")
        .help("SLAM configuration YAML file.")
        .nargs(1);
    program.add_argument("-dc", "--deviceconfig")
        .help("Device configuration YAML file.")
        .nargs(1);
    program.add_argument("-lc", "--license").help("License file.").nargs(1);
    program.add_argument("--csv").help("Save CSV-format trajectory.").nargs(1);
    program.add_argument("--tum").help("Save TUM-format trajectory.").nargs(1);
    program.add_argument("-p", "--play")
        .help("Start playing immediately.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--headless")
        .help("Disable LiteViz and exit when dataset processing completes.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("input").help("input file");
    program.parse_args(argc, argv);
    std::string data_path = program.get<std::string>("input");
    std::string slam_config_path = program.get<std::string>("-sc");
    std::string device_config_path = program.get<std::string>("-dc");
    std::string license_path = program.get<std::string>("-lc");
    std::string csv_output = program.get<std::string>("--csv");
    std::string tum_output = program.get<std::string>("--tum");
    headless = program.get<bool>("--headless");
    bool isRunning = program.get<bool>("-p") || headless;

    // create slam with configuration files
    void *yaml_config = nullptr;
    int create_succ =
        XRSLAMCreate(slam_config_path.c_str(), device_config_path.c_str(),
                     license_path.c_str(), "XRSLAM PC", &yaml_config);
    std::cout << "create SLAM success: " << create_succ << std::endl;

    std::vector<std::unique_ptr<TrajectoryWriter>> outputs;
    if (csv_output.length() > 0) {
        outputs.emplace_back(std::make_unique<CsvTrajectoryWriter>(csv_output));
    }
    if (tum_output.length() > 0) {
        outputs.emplace_back(std::make_unique<TumTrajectoryWriter>(tum_output));
    }

    std::unique_ptr<DatasetReader> reader =
        DatasetReader::create_reader(data_path, yaml_config);
    if (!reader) {
        fprintf(stderr, "Cannot open \"%s\"\n", data_path.c_str());
        return EXIT_FAILURE;
    }
    const bool requires_imu = XRSLAMRequiresIMU(yaml_config) != 0;

    bool has_gyroscope = false, has_accelerometer = false;
    outputs.emplace_back(std::make_unique<ConsoleTrajectoryWriter>());
    DatasetReader::NextDataType next_type;
    size_t processed_camera_frames = 0;

    if (!headless) {
        viewer = std::make_shared<Viewer>("XRSLAM PC", 1280, 720);
        viewer->start();
        if (isRunning) {
            {
                std::lock_guard<std::mutex> lock(viewer->_notifier->mtx);
                viewer->_notifier->ready = true;
            }
            viewer->_notifier->cv.notify_one();
        }
    }

    std::unique_ptr<xrslam::InspectPainter> feature_tracker_painter;
    if (!headless) {
        feature_tracker_painter =
            std::make_unique<OpenCvPainter>(feature_tracker_cvimage);
        inspect_debug(feature_tracker_painter, painter) {
            painter = feature_tracker_painter.get();
        }
    }

    const auto dataset_processing_start = std::chrono::steady_clock::now();

    while ((next_type = reader->next()) != DatasetReader::END) {

        switch (next_type) {
        case DatasetReader::AGAIN:
            continue;
        case DatasetReader::GYROSCOPE: {
            has_gyroscope = true;
            auto [t, gyro] = reader->read_gyroscope();
            XRSLAMPushSensorData(XRSLAM_SENSOR_GYROSCOPE, &gyro);
        } break;
        case DatasetReader::ACCELEROMETER: {
            has_accelerometer = true;
            auto [t, acc] = reader->read_accelerometer();
            XRSLAMPushSensorData(XRSLAM_SENSOR_ACCELERATION, &acc);
        } break;
        case DatasetReader::CAMERA: {

            if (!isRunning) {
                auto &notifier = viewer->_notifier;
                std::unique_lock<std::mutex> lock(viewer->_notifier->mtx);
                notifier->cv.wait(lock, [&notifier] { return notifier->ready;});
            }

            auto [t, img] = reader->read_image();
            ++processed_camera_frames;
            auto [depth_t, depth] = reader->read_depth();
            if (!depth.empty()) {
                XRSLAMDepthImage depth_image{};
                depth_image.confidence = nullptr;
                depth_image.timeStamp = depth_t;
                depth_image.width = depth.cols;
                depth_image.height = depth.rows;
                depth_image.stride = (int)depth.step[0];
                depth_image.source = reader->depth_source();

                if (depth.type() == CV_32FC1) {
                    depth_image.data_f32 = depth.ptr<float>();
                    depth_image.format = XRSLAM_DEPTH_FLOAT32;
                    depth_image.scale = 1.0;
                } else if (depth.type() == CV_16UC1) {
                    depth_image.data = depth.ptr<uint16_t>();
                    depth_image.format = XRSLAM_DEPTH_UINT16;
                    depth_image.scale = reader->depth_scale();
                } else {
                    break;
                }

                XRSLAMPushSensorData(XRSLAM_SENSOR_DEPTH_CAMERA, &depth_image);
            }

            XRSLAMImage image;
            image.camera_id = 0;
            image.timeStamp = t;
            image.ext = nullptr;
            image.data = img.data;
            image.channel = img.channels();
            image.stride = img.step[0];
            XRSLAMPushSensorData(XRSLAM_SENSOR_CAMERA, &image);
            if (!requires_imu || (has_accelerometer && has_gyroscope)) {
                XRSLAMRunOneFrame();
                XRSLAMState state;
                XRSLAMGetResult(XRSLAM_RESULT_STATE, &state);
                if (state == XRSLAM_STATE_TRACKING_SUCCESS) {
                    
                    GetShowElements();

                    XRSLAMPose pose_b;
                    XRSLAMGetResult(XRSLAM_RESULT_BODY_POSE, &pose_b);
                    if (pose_b.timestamp > 0) {
                        for (auto &output : outputs) {
                            output->write_pose(pose_b.timestamp, pose_b);
                        }
                    }
                }
            }
        } break;
        default: {
        } break;
        }
    }

    const auto dataset_processing_end = std::chrono::steady_clock::now();
    const double dataset_processing_seconds =
        std::chrono::duration<double>(dataset_processing_end -
                                      dataset_processing_start)
            .count();
    std::cerr << "[XRSLAM Player] dataset processing complete: "
              << processed_camera_frames << " camera frames in "
              << dataset_processing_seconds << " s"
              << " ("
              << (processed_camera_frames > 0
                      ? 1000.0 * dataset_processing_seconds /
                            processed_camera_frames
                      : 0.0)
              << " ms/camera frame)" << std::endl;

    while (viewer) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    XRSLAMDestroy();

    return EXIT_SUCCESS;
}
