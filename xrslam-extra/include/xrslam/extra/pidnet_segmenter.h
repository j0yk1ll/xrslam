#ifndef XRSLAM_EXTRA_PIDNET_SEGMENTER_H
#define XRSLAM_EXTRA_PIDNET_SEGMENTER_H

#include <cstdint>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace xrslam::extra {

// The label and confidence images always have the source image dimensions.
// Pixels are in the same coordinate system as the supplied undistorted image.
// label: CV_8UC1, indices [0, num_classes); confidence: CV_32FC1, [0, 1].
struct SemanticSegmentationResult {
    cv::Mat label;
    cv::Mat confidence;
    std::uint32_t num_classes = 0;
};

// Self-contained optional TorchScript model executor. It does not access or
// mutate any tracker, estimator, graph, or archive state.
//
// The traced model must have been exported with exactly canvas_width x
// canvas_height input; dynamic sizing is intentionally not assumed.
class PidNetSegmenter final {
  public:
    PidNetSegmenter(const std::string &model_path,
                    const std::string &device,
                    std::uint32_t num_classes,
                    int canvas_width = 1024,
                    int canvas_height = 512);
    ~PidNetSegmenter();

    PidNetSegmenter(const PidNetSegmenter &) = delete;
    PidNetSegmenter &operator=(const PidNetSegmenter &) = delete;

    // Input: uint8 grayscale, BGR, or BGRA. Reprojects model labels back into
    // original source image pixels with aspect-ratio-preserving letterboxing.
    SemanticSegmentationResult segment(const cv::Mat &source);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_PIDNET_SEGMENTER_H
