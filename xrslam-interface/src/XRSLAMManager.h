#ifndef XRSLAM_SLAMMANAGER_H
#define XRSLAM_SLAMMANAGER_H

#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>

#include "xrslam/inspection.h"
#include "xrslam/core/detail.h"
#include <xrslam/utility/logger.h>
#include "xrslam/extra/opencv_image.h"
#include "xrslam/extra/yaml_config.h"
#include "XRSLAM.h"

namespace xrslam {

class Config;
class Image;

class XRSLAMManager {
  public:
    static XRSLAMManager &Instance();
    ~XRSLAMManager();

    void Init(std::shared_ptr<Config> config);
    int CheckLicense(const char *license_path, const char *product_name);

    void PushImage(XRSLAMImage *image);
    void PushDepth(XRSLAMDepthImage *depth);
    void PushAcceleration(XRSLAMAcceleration *acc);
    void PushGyroscope(XRSLAMGyroscope *gyro);

    void RunOneFrame();
    bool GetAsyncDepthStats(XRSLAMAsyncDepthStats *out);

    void Destroy();

    size_t CopySemanticMap(XRSLAMSemanticMapPoint *dst, size_t capacity,
                           size_t *total, bool stable_only) const;
    bool WaitForSemanticIdle() const;
    size_t QuerySemanticKeyframes(
        const char *class_name, double min_mean_confidence,
        double min_pixel_coverage, size_t max_results,
        XRSLAMSemanticKeyframeHit *dst, size_t capacity, size_t *total) const;
    size_t CopyDenseSemanticMap(XRSLAMDenseSemanticPoint *dst,
                                size_t capacity, size_t *total,
                                bool stable_only) const;

    void GetResultCameraPose(XRSLAMPose *pose) const;
    void GetResultBodyPose(XRSLAMPose *pose) const;
    void GetResultGlobalBodyPose(XRSLAMPose *pose) const;
    bool GetVerifiedMapAlignment(XRSLAMVerifiedMapAlignment *out) const;
    void GetResultState(XRSLAMState *state) const;
    void GetResultLandmarks(XRSLAMLandmarks *landmarks) const;
    void GetResultFeatures(XRSLAMFeatures *features) const;
    void GetResultBias(XRSLAMIMUBias *bias) const;
    void GetResultVersion(XRSLAMStringOutput *output) const;
    void GetInfoIntrinsics(XRSLAMIntrinsics *intrinsics) const;
  private:
    XRSLAMManager();

  private:
    std::shared_ptr<Config> config_;
    std::unique_ptr<XRSLAM::Detail> detail_;
    std::mutex input_mutex_;
    std::shared_ptr<xrslam::Image> cur_image_;
    cv::Mat pending_depth_;
    double pending_depth_timestamp_ = -1.0;
    DepthSource pending_depth_source_ = DepthSource::SENSOR_METRIC;
    struct DelayedDepth {
        double timestamp = 0.0;
        cv::Mat meters;
        DepthSource source = DepthSource::MONOCULAR_METRIC;
    };
    std::deque<DelayedDepth> delayed_depth_queue_;
    // Separate A3 work queue: A2 RGB association and counters are unchanged.
    std::deque<DelayedDepth> geometry_depth_queue_;
    // Last 32 per-frame medians, not raw per-feature ratios.
    std::deque<double> scale_frame_medians_;
    std::deque<double> candidate_frame_medians_;
    XRSLAMAsyncDepthStats async_depth_stats_{};
    bool async_depth_enabled_ = false;
    bool experimental_vio_depth_ = false;
    void ProcessDelayedDepthLocked();
    void ProcessOptimizedDepthLocked();
};
} // namespace xrslam
#endif
