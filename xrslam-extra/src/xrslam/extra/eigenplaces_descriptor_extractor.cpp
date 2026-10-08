#include <xrslam/extra/eigenplaces_descriptor_extractor.h>

#include <xrslam/extra/opencv_image.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace xrslam::extra {

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kCacheVersion = 1;
constexpr std::array<char, 8> kCacheMagic = {
    'X', 'R', 'E', 'P', '5', '1', '2', '\0'};

// Bump this tag whenever the exact model input or output contract changes.
constexpr char kCachePipelineTag[] =
    "eigenplaces-r50-512-rgb-imagenet-l2-v1";

std::uint64_t fnv1a_append(std::uint64_t hash, const void *data, size_t size) {
    const auto *bytes =
        static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

std::uint64_t descriptor_checksum(const PlaceDescriptor &descriptor) {
    return fnv1a_append(
        kFnvOffset,
        descriptor.values.data(),
        descriptor.values.size() * sizeof(float));
}

std::uint64_t model_fingerprint(const std::string &model_path) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "EigenPlaces cache could not open model for fingerprinting");

    std::uint64_t hash = fnv1a_append(
        kFnvOffset,
        kCachePipelineTag,
        sizeof(kCachePipelineTag) - 1);

    std::array<char, 64 * 1024> buffer;
    while (input) {
        input.read(
            buffer.data(),
            static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash = fnv1a_append(
                hash, buffer.data(),
                static_cast<size_t>(count));
        }
    }
    if (!input.eof())
        throw std::runtime_error(
            "EigenPlaces cache failed while fingerprinting model");
    return hash;
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0')
           << std::setw(16) << value;
    return stream.str();
}

cv::Mat make_rgb_input(const OpenCvImage &image) {
    const cv::Mat &source =
        image.place_recognition_source();
    if (source.empty())
        throw std::runtime_error(
            "EigenPlaces place-recognition source is unavailable");

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

std::uint64_t image_fingerprint(const cv::Mat &rgb) {
    std::uint64_t hash = kFnvOffset;
    const std::uint32_t rows =
        static_cast<std::uint32_t>(rgb.rows);
    const std::uint32_t cols =
        static_cast<std::uint32_t>(rgb.cols);
    const std::uint32_t type =
        static_cast<std::uint32_t>(rgb.type());

    hash = fnv1a_append(hash, &rows, sizeof(rows));
    hash = fnv1a_append(hash, &cols, sizeof(cols));
    hash = fnv1a_append(hash, &type, sizeof(type));
    hash = fnv1a_append(
        hash, rgb.data,
        rgb.total() * rgb.elemSize());
    return hash;
}

torch::Tensor make_eigenplaces_tensor(const cv::Mat &rgb,
                                      const torch::Device &device) {
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

bool descriptor_valid(const PlaceDescriptor &descriptor) {
    if (descriptor.dimension() !=
        EigenPlacesDescriptorExtractor::kDescriptorDimension) {
        return false;
    }

    double squared_norm = 0.0;
    for (float value : descriptor.values) {
        if (!std::isfinite(value))
            return false;
        const double value_double =
            static_cast<double>(value);
        squared_norm += value_double * value_double;
    }
    return std::isfinite(squared_norm) &&
           squared_norm > 1.0e-24;
}

bool load_cached_descriptor(
    const std::filesystem::path &path,
    std::uint64_t expected_model_fingerprint,
    std::uint64_t expected_image_fingerprint,
    PlaceDescriptor &descriptor) {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            return false;

        std::array<char, 8> magic{};
        std::uint32_t version = 0;
        std::uint32_t dimension = 0;
        std::uint64_t model_hash = 0;
        std::uint64_t image_hash = 0;
        std::uint64_t checksum = 0;

        input.read(magic.data(), magic.size());
        input.read(
            reinterpret_cast<char *>(&version),
            sizeof(version));
        input.read(
            reinterpret_cast<char *>(&dimension),
            sizeof(dimension));
        input.read(
            reinterpret_cast<char *>(&model_hash),
            sizeof(model_hash));
        input.read(
            reinterpret_cast<char *>(&image_hash),
            sizeof(image_hash));
        input.read(
            reinterpret_cast<char *>(&checksum),
            sizeof(checksum));

        if (!input ||
            magic != kCacheMagic ||
            version != kCacheVersion ||
            dimension !=
                EigenPlacesDescriptorExtractor::kDescriptorDimension ||
            model_hash != expected_model_fingerprint ||
            image_hash != expected_image_fingerprint) {
            return false;
        }

        descriptor.values.resize(dimension);
        input.read(
            reinterpret_cast<char *>(
                descriptor.values.data()),
            static_cast<std::streamsize>(
                dimension * sizeof(float)));
        if (!input || input.peek() != std::ifstream::traits_type::eof()) {
            descriptor.values.clear();
            return false;
        }

        if (!descriptor_valid(descriptor) ||
            descriptor_checksum(descriptor) != checksum) {
            descriptor.values.clear();
            return false;
        }
        return true;
    } catch (...) {
        descriptor.values.clear();
        return false;
    }
}

bool store_cached_descriptor(
    const std::filesystem::path &path,
    std::uint64_t model_hash,
    std::uint64_t image_hash,
    const PlaceDescriptor &descriptor) {
    if (!descriptor_valid(descriptor))
        return false;

    try {
        std::filesystem::create_directories(
            path.parent_path());

        std::ofstream output(
            path,
            std::ios::binary | std::ios::trunc);
        if (!output)
            return false;

        const std::uint32_t dimension =
            static_cast<std::uint32_t>(
                descriptor.dimension());
        const std::uint64_t checksum =
            descriptor_checksum(descriptor);

        output.write(kCacheMagic.data(), kCacheMagic.size());
        output.write(
            reinterpret_cast<const char *>(&kCacheVersion),
            sizeof(kCacheVersion));
        output.write(
            reinterpret_cast<const char *>(&dimension),
            sizeof(dimension));
        output.write(
            reinterpret_cast<const char *>(&model_hash),
            sizeof(model_hash));
        output.write(
            reinterpret_cast<const char *>(&image_hash),
            sizeof(image_hash));
        output.write(
            reinterpret_cast<const char *>(&checksum),
            sizeof(checksum));
        output.write(
            reinterpret_cast<const char *>(
                descriptor.values.data()),
            static_cast<std::streamsize>(
                descriptor.values.size() * sizeof(float)));
        output.flush();
        return output.good();
    } catch (...) {
        return false;
    }
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

        const char *cache_dir =
            std::getenv("XRSLAM_EIGENPLACES_CACHE_DIR");
        if (cache_dir && cache_dir[0] != '\0') {
            try {
                cache_root = std::filesystem::path(cache_dir);
                model_hash = model_fingerprint(model_path);
                cache_enabled = true;
                std::fprintf(
                    stderr,
                    "[EigenPlacesCache] enabled=1 "
                    "version=%u device=%s model_hash=%s "
                    "dir=%s\n",
                    kCacheVersion,
                    use_cuda ? "cuda" : "cpu",
                    hex_u64(model_hash).c_str(),
                    cache_root.string().c_str());
            } catch (const std::exception &e) {
                std::fprintf(
                    stderr,
                    "[EigenPlacesCache] enabled=0 "
                    "reason=initialization_failed error=%s\n",
                    e.what());
            }
        }
    }

    std::filesystem::path cache_path(
        std::uint64_t image_hash) const {
        const std::string device_name =
            device.is_cuda() ? "cuda" : "cpu";
        return cache_root /
               ("v" + std::to_string(kCacheVersion) +
                "-" + device_name +
                "-" + hex_u64(model_hash)) /
               (hex_u64(image_hash) + ".ep512");
    }

    torch::jit::script::Module module;
    torch::Device device;
    std::mutex mutex;

    bool cache_enabled = false;
    std::filesystem::path cache_root;
    std::uint64_t model_hash = 0;
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

    const cv::Mat rgb = make_rgb_input(*opencv_image);
    std::uint64_t image_hash = 0;
    std::filesystem::path cache_path;

    if (impl_->cache_enabled) {
        image_hash = image_fingerprint(rgb);
        cache_path = impl_->cache_path(image_hash);

        PlaceDescriptor cached;
        if (load_cached_descriptor(
                cache_path,
                impl_->model_hash,
                image_hash,
                cached)) {
            std::fprintf(
                stderr,
                "[EigenPlacesCache] t=%.9f hit=1 "
                "image_hash=%s\n",
                image.t,
                hex_u64(image_hash).c_str());
            return cached;
        }
    }

    c10::InferenceMode inference_mode;
    torch::Tensor input =
        make_eigenplaces_tensor(rgb, impl_->device);
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

    if (impl_->cache_enabled) {
        const bool stored =
            store_cached_descriptor(
                cache_path,
                impl_->model_hash,
                image_hash,
                descriptor);
        std::fprintf(
            stderr,
            "[EigenPlacesCache] t=%.9f hit=0 stored=%d "
            "image_hash=%s\n",
            image.t,
            stored ? 1 : 0,
            hex_u64(image_hash).c_str());
    }

    return descriptor;
}

} // namespace xrslam::extra
