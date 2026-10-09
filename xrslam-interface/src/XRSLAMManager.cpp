#include "XRSLAMManager.h"

#include <xrslam/core/place_graph_4dof_shadow.h>
#include <xrslam/place_recognition.h>
#include <xrslam/extra/orb_local_descriptor_extractor.h>

#if defined(XRSLAM_HAS_EIGENPLACES)
#include <xrslam/extra/eigenplaces_descriptor_extractor.h>
#endif

#if defined(XRSLAM_HAS_USEARCH)
#include <xrslam/extra/usearch_place_database.h>
#endif

#include <cstdlib>
#include <cmath>
#include <stdexcept>

#define XRSLAM_VERSION "0.1.0"

namespace xrslam {
XRSLAMManager &XRSLAMManager::Instance() {
    static XRSLAMManager SLAMManagerInstance;
    return SLAMManagerInstance;
}

XRSLAMManager::XRSLAMManager() {}
XRSLAMManager::~XRSLAMManager() {}

static const unsigned char logo_ascii[] = {
    0x0A, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0x20,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x95, 0x97, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x96,
    0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96,
    0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x0A, 0xE2, 0x95, 0x9A,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90,
    0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x97, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0x20, 0xE2, 0x95, 0x9A, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2,
    0x95, 0x9D, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94,
    0xE2, 0x95, 0x9D, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88,
    0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91,
    0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x91, 0x0A, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x94, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0x20, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0x20, 0x20,
    0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x90, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x95,
    0x9A, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95,
    0x9D, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x94, 0xE2, 0x95, 0x9D, 0x20,
    0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x96, 0x88, 0xE2, 0x95, 0x97, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2,
    0x95, 0x91, 0x20, 0x20, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95,
    0x91, 0xE2, 0x96, 0x88, 0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x20, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0x20, 0xE2, 0x96, 0x88,
    0xE2, 0x96, 0x88, 0xE2, 0x95, 0x91, 0x0A, 0xE2, 0x95, 0x9A, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90,
    0xE2, 0x95, 0x9D, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D,
    0x20, 0x20, 0xE2, 0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2,
    0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2,
    0x95, 0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0xE2, 0x95,
    0x9A, 0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D, 0xE2, 0x95, 0x9A, 0xE2, 0x95,
    0x90, 0xE2, 0x95, 0x9D, 0x20, 0x20, 0x20, 0x20, 0x20, 0xE2, 0x95, 0x9A,
    0xE2, 0x95, 0x90, 0xE2, 0x95, 0x9D};

void XRSLAMManager::Init(std::shared_ptr<Config> config) {
    std::shared_ptr<PlaceDescriptorExtractor>
        place_descriptor_extractor;
    std::shared_ptr<PlaceDatabase> place_database;
    std::shared_ptr<LocalDescriptorExtractor>
        local_descriptor_extractor;

    const char *descriptor_shadow =
        std::getenv("XRSLAM_PLACE_DESCRIPTOR_SHADOW");
    const char *retrieval_shadow =
        std::getenv("XRSLAM_PLACE_RETRIEVAL_SHADOW");
    const bool retrieval_enabled =
        retrieval_shadow &&
        std::string(retrieval_shadow) == "1";
    const bool descriptor_enabled =
        (descriptor_shadow &&
         std::string(descriptor_shadow) == "1") ||
        retrieval_enabled;

    if (descriptor_enabled) {
#if defined(XRSLAM_HAS_EIGENPLACES)
        const char *model_path =
            std::getenv("XRSLAM_EIGENPLACES_MODEL");
        if (!model_path || model_path[0] == '\0') {
            throw std::runtime_error(
                "place descriptor/retrieval shadow requires "
                "XRSLAM_EIGENPLACES_MODEL");
        }

        place_descriptor_extractor =
            std::make_shared<
                xrslam::extra::EigenPlacesDescriptorExtractor>(
                model_path, false);
#else
        throw std::runtime_error(
            "place descriptor/retrieval shadow requires a build "
            "configured with XRSLAM_ENABLE_EIGENPLACES=ON");
#endif
    }

    if (retrieval_enabled) {
#if defined(XRSLAM_HAS_USEARCH)
        place_database =
            xrslam::extra::make_usearch_place_database();
#else
        throw std::runtime_error(
            "XRSLAM_PLACE_RETRIEVAL_SHADOW=1 requires a build "
            "configured with XRSLAM_ENABLE_USEARCH=ON");
#endif
    }

    const char *local_descriptor_shadow =
        std::getenv("XRSLAM_LOCAL_DESCRIPTOR_SHADOW");
    const char *orb_association_shadow =
        std::getenv("XRSLAM_ORB_ASSOCIATION_SHADOW");
    const char *orb_pnp_shadow =
        std::getenv("XRSLAM_ORB_PNP_SHADOW");
    if ((local_descriptor_shadow &&
         std::string(local_descriptor_shadow) == "1") ||
        (orb_association_shadow &&
         std::string(orb_association_shadow) == "1") ||
        (orb_pnp_shadow &&
         std::string(orb_pnp_shadow) == "1")) {
        local_descriptor_extractor =
            std::make_shared<
                xrslam::extra::OrbLocalDescriptorExtractor>();
    }

    detail_ = std::make_unique<XRSLAM::Detail>(config);
    detail_->set_place_descriptor_extractor(
        std::move(place_descriptor_extractor));
    detail_->set_place_database(
        std::move(place_database));
    detail_->set_local_descriptor_extractor(
        std::move(local_descriptor_extractor));
    config_ = config;
    log_message(XRSLAM_LOG_INFO, (char *)logo_ascii, XRSLAM_VERSION_STRING);
    config_->log_config();
    std::cout << "-----------------Create XRSLAM v" << XRSLAM_VERSION
              << " successfully-----------" << std::endl;
}

void XRSLAMManager::Destroy() {
    std::cout << "-----------------Destroy XRSLAM v" << XRSLAM_VERSION
              << " successfully-----------" << std::endl;
}

int XRSLAMManager::CheckLicense(const char *license_path,
                                const char *product_name) {
    return 1;
}

void XRSLAMManager::PushImage(XRSLAMImage *image) {
    // left img
    if (image->camera_id == 0) {
        std::shared_ptr<xrslam::extra::OpenCvImage> opencv_image =
            std::make_shared<xrslam::extra::OpenCvImage>();
        int cols = config_->camera_resolution()[0];
        int rows = config_->camera_resolution()[1];
        opencv_image->t = image->timeStamp;

        cv::Mat img;
        if(image->channel == 1){
            img = cv::Mat(rows, cols, CV_8UC1, image->data, image->stride);
            opencv_image->image = img.clone();
        }
        else if(image->channel == 3){
            img = cv::Mat(rows, cols, CV_8UC3, image->data, image->stride);
            cv::cvtColor(img, opencv_image->image, cv::COLOR_BGR2GRAY);
        }
        else if(image->channel == 4){
            img = cv::Mat(rows, cols, CV_8UC4, image->data, image->stride);
            cv::cvtColor(img, opencv_image->image, cv::COLOR_BGRA2GRAY);
        }
        else{
            std::cerr << "Image channel is not supported!" << std::endl;
            exit(-1);
        }
            
        // Preserve the canonical pre-CLAHE camera image for place recognition.
        // The OpenCvImage source initially shares this allocation, so enabling
        // global/local descriptors does not add another full-frame copy.
        // If correct_distortion() is used, both the frontend image and this
        // source are rectified into the same pixel geometry.
        opencv_image->raw = img.clone();
        opencv_image->set_place_recognition_source(opencv_image->raw);

        std::lock_guard<std::mutex> lck(input_mutex_);
        if (!pending_depth_.empty() &&
            std::abs(pending_depth_timestamp_ - image->timeStamp) < 1.0e-6 &&
            pending_depth_.cols == cols && pending_depth_.rows == rows) {
            opencv_image->depth_image = pending_depth_;
            opencv_image->depth_source_type = pending_depth_source_;
        }
        pending_depth_.release();
        pending_depth_timestamp_ = -1.0;
        cur_image_ = std::shared_ptr<xrslam::Image>(opencv_image);
    }
}

void XRSLAMManager::PushDepth(XRSLAMDepthImage *depth) {
    if (!depth || depth->width <= 0 || depth->height <= 0 ||
        depth->stride <= 0)
        return;

    cv::Mat meters;
    if (depth->format == XRSLAM_DEPTH_FLOAT32) {
        if (!depth->data_f32)
            return;
        cv::Mat raw(depth->height, depth->width, CV_32FC1, depth->data_f32,
                    (size_t)depth->stride);
        meters = raw.clone();
    } else {
        if (!depth->data || depth->scale <= 0.0)
            return;
        cv::Mat raw(depth->height, depth->width, CV_16UC1, depth->data,
                    (size_t)depth->stride);
        raw.convertTo(meters, CV_32FC1, depth->scale);
    }

    DepthSource source = DepthSource::SENSOR_METRIC;
    if (depth->source == XRSLAM_DEPTH_MONOCULAR_METRIC)
        source = DepthSource::MONOCULAR_METRIC;
    else if (depth->source == XRSLAM_DEPTH_MONOCULAR_RELATIVE)
        source = DepthSource::MONOCULAR_RELATIVE;

    std::lock_guard<std::mutex> lck(input_mutex_);
    pending_depth_ = meters;
    pending_depth_timestamp_ = depth->timeStamp;
    pending_depth_source_ = source;
}

void XRSLAMManager::PushAcceleration(XRSLAMAcceleration *acc) {
    detail_->track_accelerometer(acc->timestamp, acc->data[0], acc->data[1],
                                 acc->data[2]);
}

void XRSLAMManager::PushGyroscope(XRSLAMGyroscope *gyro) {
    detail_->track_gyroscope(gyro->timestamp, gyro->data[0], gyro->data[1],
                             gyro->data[2]);
}

void XRSLAMManager::RunOneFrame() {
    std::lock_guard<std::mutex> lck(input_mutex_);
    detail_->track_camera(cur_image_);
}

void XRSLAMManager::GetResultBodyPose(XRSLAMPose *pose) const {
    std::tuple<double, Pose> latest_state = detail_->get_latest_pose();
    pose->timestamp = std::get<0>(latest_state);
    Pose latest_pose = std::get<1>(latest_state);
    Pose body_pose;
    body_pose.q = latest_pose.q * config_->imu_to_body_rotation();
    body_pose.p = latest_pose.p + latest_pose.q * config_->imu_to_body_translation();
    for (int i = 0; i < 3; i++)
        pose->translation[i] = body_pose.p(i);
    pose->quaternion[0] = body_pose.q.x();
    pose->quaternion[1] = body_pose.q.y();
    pose->quaternion[2] = body_pose.q.z();
    pose->quaternion[3] = body_pose.q.w();
}

void XRSLAMManager::GetResultGlobalBodyPose(XRSLAMPose *pose) const {
    // Copy the existing body output; never mutate the estimator/local result.
    GetResultBodyPose(pose);
    PlaceGraph4DoFCorrection correction;
    if (pose->timestamp <= 0.0 ||
        !detail_->get_latest_global_correction(correction) ||
        pose->timestamp < correction.timestamp ||
        !std::isfinite(correction.yaw_deg) ||
        !correction.translation.allFinite()) {
        return;
    }

    const vector<3> local_p(
        pose->translation[0], pose->translation[1], pose->translation[2]);
    const quaternion local_q(
        pose->quaternion[3], pose->quaternion[0],
        pose->quaternion[1], pose->quaternion[2]);
    if (!local_p.allFinite() || !local_q.coeffs().allFinite() ||
        local_q.squaredNorm() < 1.0e-20) {
        return;
    }

    constexpr double radians_per_degree =
        3.14159265358979323846 / 180.0;
    const quaternion drift_q(Eigen::AngleAxisd(
        correction.yaw_deg * radians_per_degree,
        Eigen::Vector3d::UnitZ()));
    const vector<3> global_p = drift_q * local_p + correction.translation;
    const quaternion global_q = (drift_q * local_q).normalized();
    pose->translation[0] = global_p.x();
    pose->translation[1] = global_p.y();
    pose->translation[2] = global_p.z();
    pose->quaternion[0] = global_q.x();
    pose->quaternion[1] = global_q.y();
    pose->quaternion[2] = global_q.z();
    pose->quaternion[3] = global_q.w();
}

void XRSLAMManager::GetResultCameraPose(XRSLAMPose *pose) const {
    std::tuple<double, Pose> latest_state = detail_->get_latest_pose();
    pose->timestamp = std::get<0>(latest_state);
    Pose latest_pose = std::get<1>(latest_state);
    Pose camera_pose;
    camera_pose.q = latest_pose.q * config_->camera_to_body_rotation();
    camera_pose.p = latest_pose.p + latest_pose.q * config_->camera_to_body_translation();
    for (int i = 0; i < 3; i++)
        pose->translation[i] = camera_pose.p(i);
    pose->quaternion[0] = camera_pose.q.x();
    pose->quaternion[1] = camera_pose.q.y();
    pose->quaternion[2] = camera_pose.q.z();
    pose->quaternion[3] = camera_pose.q.w();
}

void XRSLAMManager::GetInfoIntrinsics(XRSLAMIntrinsics *intrinsics) const {
    intrinsics->fx = config_->camera_intrinsic()(0, 0);
    intrinsics->fy = config_->camera_intrinsic()(1, 1);
    intrinsics->cx = config_->camera_intrinsic()(0, 2);
    intrinsics->cy = config_->camera_intrinsic()(1, 2);
}


void XRSLAMManager::GetResultState(XRSLAMState *state) const {
    SysState cur_state = detail_->get_system_state();
    if (cur_state == SYS_INITIALIZING) {
        *state = XRSLAM_STATE_INITIALIZING;
    } else if (cur_state == SYS_TRACKING) {
        *state = XRSLAM_STATE_TRACKING_SUCCESS;
    } else if (cur_state == SYS_CRASH) {
        *state = XRSLAM_STATE_TRACKING_FAIL;
    } else if (cur_state == SYS_UNKNOWN) {
        *state = XRSLAM_STATE_TRACKING_FAIL;
    }
}
void XRSLAMManager::GetResultLandmarks(XRSLAMLandmarks *landmarks) const {
    inspect_debug(sliding_window_landmarks, swlandmarks) {
        auto pts = std::any_cast<std::vector<xrslam::Landmark>>(swlandmarks);
        landmarks->num_landmarks = pts.size();
        landmarks->landmarks = new XRSLAMLandmark[pts.size()];
        for (int i = 0; i < pts.size(); i++) {
            xrslam::vector<3> cur_p = pts[i].p;
            landmarks->landmarks[i].x = cur_p(0);
            landmarks->landmarks[i].y = cur_p(1);
            landmarks->landmarks[i].z = cur_p(2);
        }
    }
}

void XRSLAMManager::GetResultFeatures(XRSLAMFeatures *features) const {}

void XRSLAMManager::GetResultBias(XRSLAMIMUBias *bias) const {
    inspect_debug(sliding_window_current_bg, bg) {
        if (bg.has_value()) {
            xrslam::vector<3> gyr_bias = std::any_cast<xrslam::vector<3>>(bg);
            bias->gyr_bias.data[0] = gyr_bias(0);
            bias->gyr_bias.data[1] = gyr_bias(1);
            bias->gyr_bias.data[2] = gyr_bias(2);
        }
    }
    inspect_debug(sliding_window_current_ba, ba) {
        if (ba.has_value()) {
            xrslam::vector<3> acc_bias = std::any_cast<xrslam::vector<3>>(ba);
            bias->acc_bias.data[0] = acc_bias(0);
            bias->acc_bias.data[1] = acc_bias(1);
            bias->acc_bias.data[2] = acc_bias(2);
        }
    }
}

void XRSLAMManager::GetResultVersion(XRSLAMStringOutput *output) const {
    output->str_length = strlen(XRSLAM_VERSION);
    output->data = new char[output->str_length + 5];
    strcpy(output->data, XRSLAM_VERSION);
}

} // namespace xrslam
