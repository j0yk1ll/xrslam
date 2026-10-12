#include <xrslam/extra/pidnet_segmenter.h>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace xrslam::extra {
namespace {

cv::Mat make_rgb(const cv::Mat &source) {
    if (source.empty() || source.depth() != CV_8U)
        throw std::invalid_argument("PIDNet expects nonempty uint8 camera image");
    cv::Mat rgb;
    if (source.channels() == 1)
        cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
    else if (source.channels() == 3)
        cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
    else if (source.channels() == 4)
        cv::cvtColor(source, rgb, cv::COLOR_BGRA2RGB);
    else
        throw std::invalid_argument("PIDNet supports grayscale/BGR/BGRA images");
    return rgb;
}

struct Letterbox {
    cv::Mat rgb;
    cv::Rect content;
};

Letterbox letterbox_rgb(const cv::Mat &source, int width, int height) {
    const cv::Mat rgb = make_rgb(source);
    const double scale = std::min(
        static_cast<double>(width) / rgb.cols,
        static_cast<double>(height) / rgb.rows);
    const int resized_width = std::clamp(
        static_cast<int>(std::lround(rgb.cols * scale)), 1, width);
    const int resized_height = std::clamp(
        static_cast<int>(std::lround(rgb.rows * scale)), 1, height);
    const int left = (width - resized_width) / 2;
    const int top = (height - resized_height) / 2;
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_width, resized_height),
               0.0, 0.0, cv::INTER_LINEAR);
    // Padding in the ImageNet-mean RGB color is approximately zero after
    // normalization. Class predictions in the padding are discarded.
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(124, 116, 104));
    const cv::Rect roi(left, top, resized_width, resized_height);
    resized.copyTo(canvas(roi));
    return {std::move(canvas), roi};
}

torch::Tensor normalize_rgb(const cv::Mat &canvas, const torch::Device &device) {
    // Create a privately owned tensor: the OpenCV storage is only borrowed
    // until the conversion and copy below complete.
    auto input = torch::from_blob(
        canvas.data, {canvas.rows, canvas.cols, 3},
        torch::TensorOptions().dtype(torch::kUInt8))
            .to(torch::kFloat32)
            .div_(255.0f)
            .permute({2, 0, 1})
            .unsqueeze(0)
            .contiguous();
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(torch::kCPU);
    const auto mean = torch::tensor(
        {0.485f, 0.456f, 0.406f}, options).view({1, 3, 1, 1});
    const auto std = torch::tensor(
        {0.229f, 0.224f, 0.225f}, options).view({1, 3, 1, 1});
    return input.sub(mean).div(std).to(device);
}

torch::Tensor upsample(const torch::Tensor &input, int height, int width) {
    namespace F = torch::nn::functional;
    return F::interpolate(
        input,
        F::InterpolateFuncOptions()
            .size(std::vector<std::int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(true));
}

} // namespace

struct PidNetSegmenter::Impl {
    Impl(const std::string &path, const std::string &device_name,
         std::uint32_t classes, int width, int height)
        : device(device_name == "cuda" ? torch::Device(torch::kCUDA)
                                       : torch::Device(torch::kCPU)),
          num_classes(classes), canvas_width(width), canvas_height(height) {
        if (device_name != "cpu" && device_name != "cuda")
            throw std::invalid_argument("PIDNet device must be cpu or cuda");
        if (device_name == "cuda" && !torch::cuda::is_available())
            throw std::runtime_error("PIDNet CUDA requested, but unavailable");
        if (classes == 0 || classes > 255)
            throw std::invalid_argument("PIDNet class count must be 1..255");
        if (width < 64 || height < 64 || width % 32 || height % 32)
            throw std::invalid_argument(
                "PIDNet fixed canvas width and height must be >=64 and divisible by 32");
        module = torch::jit::load(path, device);
        module.eval();
    }
    torch::Device device;
    std::uint32_t num_classes;
    int canvas_width;
    int canvas_height;
    torch::jit::script::Module module;
    std::mutex mutex;
};

PidNetSegmenter::PidNetSegmenter(const std::string &path,
                                 const std::string &device_name,
                                 std::uint32_t classes,
                                 int width, int height)
    : impl_(std::make_unique<Impl>(
          path, device_name, classes, width, height)) {}

PidNetSegmenter::~PidNetSegmenter() = default;

SemanticSegmentationResult PidNetSegmenter::segment(const cv::Mat &source) {
    if (source.empty())
        throw std::invalid_argument("PIDNet source is empty");
    // The model and its GPU execution are serialized to avoid concurrent
    // access to the same module; the future worker owns queue scheduling.
    std::lock_guard<std::mutex> lock(impl_->mutex);
    c10::InferenceMode inference_mode;
    const Letterbox box = letterbox_rgb(
        source, impl_->canvas_width, impl_->canvas_height);
    const torch::Tensor input = normalize_rgb(box.rgb, impl_->device);
    torch::Tensor logits = impl_->module.forward({input}).toTensor();
    if (logits.dim() != 4 || logits.size(0) != 1 ||
        logits.size(1) != static_cast<std::int64_t>(impl_->num_classes) ||
        logits.size(2) <= 0 || logits.size(3) <= 0 ||
        !torch::isfinite(logits).all().item<bool>()) {
        throw std::runtime_error("PIDNet returned nonfinite or invalid [1,C,H,W] logits");
    }
    // Reference PIDNet upsamples logits before argmax (align_corners=True).
    // Crop the letterbox AFTER upsampling on the model canvas, then resize
    // logits to the original source resolution to maintain pixel alignment.
    logits = upsample(logits, impl_->canvas_height, impl_->canvas_width);
    logits = logits.narrow(2, box.content.y, box.content.height)
                   .narrow(3, box.content.x, box.content.width);
    logits = upsample(logits, source.rows, source.cols);

    const torch::Tensor probabilities = torch::softmax(logits, 1);
    const auto best = probabilities.max(1);
    const torch::Tensor labels = std::get<1>(best)
                                     .to(torch::kCPU, torch::kUInt8)
                                     .contiguous();
    const torch::Tensor confidence = std::get<0>(best)
                                         .to(torch::kCPU, torch::kFloat32)
                                         .contiguous();
    SemanticSegmentationResult result;
    result.num_classes = impl_->num_classes;
    result.label = cv::Mat(source.rows, source.cols, CV_8UC1,
                           labels.data_ptr<std::uint8_t>()).clone();
    result.confidence = cv::Mat(source.rows, source.cols, CV_32FC1,
                                confidence.data_ptr<float>()).clone();
    return result;
}

} // namespace xrslam::extra
