#include <xrslam/extra/eigenplaces_descriptor_extractor.h>

#include <xrslam/extra/opencv_image.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace xrslam::extra {

namespace {

cv::Mat make_rgb_input(const OpenCvImage &image) {
    const cv::Mat &source = image.raw.empty() ? image.image : image.raw;
    if (source.empty())
        throw std::runtime_error("EigenPlaces received an empty image");

    cv::Mat rgb;
    if (source.channels() == 1) {
        cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
    } else if (source.channels() == 3) {
        cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
    } else if (source.channels() == 4) {
        cv::cvtColor(source, rgb, cv::COLOR_BGRA2RGB);
    } else {
        throw std::runtime_error(
            "EigenPlaces supports 1, 3, or 4 channel uint8 images");
    }

    if (rgb.depth() != CV_8U)
        throw std::runtime_error("EigenPlaces expects uint8 camera images");
    if (!rgb.isContinuous())
        rgb = rgb.clone();
    return rgb;
}

torch::Tensor make_eigenplaces_tensor(const OpenCvImage &image,
                                      const torch::Device &device) {
    cv::Mat rgb = make_rgb_input(image);

    auto input = torch::from_blob(
                     rgb.data, {rgb.rows, rgb.cols, 3},
                     torch::TensorOptions().dtype(torch::kUInt8))
                     .to(torch::kFloat32)
                     .div_(255.0f)
                     .permute({2, 0, 1})
                     .unsqueeze(0)
                     .contiguous();

    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto mean =
        torch::tensor({0.485f, 0.456f, 0.406f}, options).view({1, 3, 1, 1});
    const auto std =
        torch::tensor({0.229f, 0.224f, 0.225f}, options).view({1, 3, 1, 1});

    input = input.sub(mean).div(std);
    return input.to(device);
}

} // namespace

struct EigenPlacesDescriptorExtractor::Impl {
    explicit Impl(const std::string &model_path, bool use_cuda)
        : device(use_cuda ? torch::Device(torch::kCUDA)
                          : torch::Device(torch::kCPU)) {
        if (use_cuda && !torch::cuda::is_available())
            throw std::runtime_error(
                "EigenPlaces CUDA requested but torch::cuda is unavailable");

        module = torch::jit::load(model_path, device);
        module.eval();
    }

    torch::jit::script::Module module;
    torch::Device device;
    std::mutex mutex;
};

EigenPlacesDescriptorExtractor::EigenPlacesDescriptorExtractor(
    const std::string &model_path, bool use_cuda)
    : impl_(std::make_unique<Impl>(model_path, use_cuda)) {}

EigenPlacesDescriptorExtractor::~EigenPlacesDescriptorExtractor() = default;

PlaceDescriptor EigenPlacesDescriptorExtractor::extract(const Image &image) {
    const auto *opencv_image = dynamic_cast<const OpenCvImage *>(&image);
    if (!opencv_image)
        throw std::runtime_error(
            "EigenPlacesDescriptorExtractor requires OpenCvImage input");

    std::lock_guard<std::mutex> lock(impl_->mutex);
    c10::InferenceMode inference_mode;

    torch::Tensor input =
        make_eigenplaces_tensor(*opencv_image, impl_->device);
    torch::Tensor output = impl_->module.forward({input}).toTensor();

    if (output.dim() != 2 || output.size(0) != 1 ||
        output.size(1) != static_cast<long>(kDescriptorDimension)) {
        throw std::runtime_error(
            "EigenPlaces TorchScript model did not return [1, 512]");
    }

    output = output.to(torch::kCPU).contiguous().view({-1});
    const float norm = output.norm(2).item<float>();
    if (!std::isfinite(norm) || norm <= 1.0e-12f)
        throw std::runtime_error("EigenPlaces returned an invalid descriptor");

    // The reference model already ends in L2Norm. Normalize once more to make
    // the PlaceDescriptor contract robust to tiny serialization roundoff.
    output.div_(norm);

    PlaceDescriptor descriptor;
    descriptor.values.resize(kDescriptorDimension);
    std::memcpy(descriptor.values.data(), output.data_ptr<float>(),
                kDescriptorDimension * sizeof(float));
    return descriptor;
}

} // namespace xrslam::extra
