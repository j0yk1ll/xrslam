#include <xrslam/extra/orb_local_descriptor_extractor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <xrslam/extra/opencv_image.h>

namespace xrslam::extra {

namespace {

cv::Mat descriptor_source(const OpenCvImage &image) {
    // 0104b retains raw after ordinary frontend buffers are released. GFTT
    // pixels remain in the same image coordinate system; both historical and
    // future ORB matching must use this retained source consistently.
    const cv::Mat &source =
        !image.raw.empty() ? image.raw : image.image;
    if (source.empty())
        throw std::runtime_error(
            "ORB local descriptor source is unavailable");

    cv::Mat gray;
    if (source.channels() == 1) {
        gray = source;
    } else if (source.channels() == 3) {
        cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    } else if (source.channels() == 4) {
        cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error(
            "ORB local descriptors support 1, 3, or 4 channel images");
    }

    if (gray.depth() != CV_8U)
        throw std::runtime_error(
            "ORB local descriptors expect uint8 camera images");
    return gray;
}

} // namespace

class OrbLocalDescriptorExtractor::Impl {
  public:
    Impl()
        : orb(cv::ORB::create(
              1000, 1.2f, 8, 20, 0, 2,
              cv::ORB::HARRIS_SCORE, 31, 20)) {
        descriptor_dimension =
            static_cast<size_t>(orb->descriptorSize());
        if (descriptor_dimension == 0)
            throw std::runtime_error(
                "OpenCV ORB reported zero descriptor size");
    }

    cv::Ptr<cv::ORB> orb;
    size_t descriptor_dimension = 0;
    std::mutex mutex;
};

OrbLocalDescriptorExtractor::OrbLocalDescriptorExtractor()
    : impl_(std::make_unique<Impl>()) {}

OrbLocalDescriptorExtractor::~OrbLocalDescriptorExtractor() =
    default;

LocalDescriptorType OrbLocalDescriptorExtractor::type() const {
    return LocalDescriptorType::BINARY_U8;
}

size_t OrbLocalDescriptorExtractor::dimension() const {
    return impl_->descriptor_dimension;
}

LocalDescriptorSet OrbLocalDescriptorExtractor::extract(
    const Image &image,
    const std::vector<vector<2>> &points) {
    const auto *opencv_image =
        dynamic_cast<const OpenCvImage *>(&image);
    if (!opencv_image)
        throw std::runtime_error(
            "OrbLocalDescriptorExtractor requires OpenCvImage input");

    LocalDescriptorSet result;
    result.type = LocalDescriptorType::BINARY_U8;
    result.dimension = impl_->descriptor_dimension;
    if (points.empty())
        return result;

    const cv::Mat gray = descriptor_source(*opencv_image);

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(points.size());
    for (size_t point_index = 0;
         point_index < points.size(); ++point_index) {
        const vector<2> &point = points[point_index];
        if (!std::isfinite(point.x()) ||
            !std::isfinite(point.y()) ||
            point.x() < 0.0 || point.y() < 0.0 ||
            point.x() >= static_cast<double>(gray.cols) ||
            point.y() >= static_cast<double>(gray.rows)) {
            continue;
        }

        cv::KeyPoint keypoint(
            cv::Point2f(
                static_cast<float>(point.x()),
                static_cast<float>(point.y())),
            31.0f);
        keypoint.class_id =
            static_cast<int>(point_index);
        keypoint.octave = 0;
        keypoints.emplace_back(keypoint);
    }

    if (keypoints.empty())
        return result;

    cv::Mat descriptors;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->orb->compute(
            gray, keypoints, descriptors);
    }

    if (descriptors.empty())
        return result;
    if (descriptors.type() != CV_8UC1 ||
        descriptors.cols !=
            static_cast<int>(impl_->descriptor_dimension) ||
        descriptors.rows !=
            static_cast<int>(keypoints.size())) {
        throw std::runtime_error(
            "OpenCV ORB returned an unexpected descriptor shape");
    }

    struct DescriptorRow {
        size_t source_index;
        size_t descriptor_row;
    };
    std::vector<DescriptorRow> rows;
    rows.reserve(keypoints.size());
    for (size_t row = 0;
         row < keypoints.size(); ++row) {
        const int class_id =
            keypoints[row].class_id;
        if (class_id < 0 ||
            static_cast<size_t>(class_id) >= points.size()) {
            throw std::runtime_error(
                "OpenCV ORB lost the source keypoint index");
        }
        rows.push_back(
            DescriptorRow{
                static_cast<size_t>(class_id), row});
    }

    std::sort(
        rows.begin(), rows.end(),
        [](const DescriptorRow &a,
           const DescriptorRow &b) {
            return a.source_index < b.source_index;
        });

    result.source_indices.reserve(rows.size());
    result.binary_values.reserve(
        rows.size() * impl_->descriptor_dimension);

    size_t previous_source_index = 0;
    for (size_t output_row = 0;
         output_row < rows.size(); ++output_row) {
        const DescriptorRow &row = rows[output_row];
        if (output_row > 0 &&
            row.source_index <= previous_source_index) {
            throw std::runtime_error(
                "OpenCV ORB returned duplicate source indices");
        }
        previous_source_index = row.source_index;

        result.source_indices.emplace_back(
            row.source_index);
        const std::uint8_t *descriptor =
            descriptors.ptr<std::uint8_t>(
                static_cast<int>(
                    row.descriptor_row));
        result.binary_values.insert(
            result.binary_values.end(),
            descriptor,
            descriptor + impl_->descriptor_dimension);
    }

    return result;
}

} // namespace xrslam::extra
