#ifndef XRSLAM_EXTRA_OPENCV_IMAGE_H
#define XRSLAM_EXTRA_OPENCV_IMAGE_H

#include <algorithm>
#include <cmath>
#include <vector>
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>
#include <xrslam/xrslam.h>

namespace xrslam::extra {

class OpenCvImage : public Image {
  public:
    OpenCvImage();

    uchar *get_rawdata() const override { return raw.data; }

    size_t width() const override { return image.cols; }

    size_t height() const override { return image.rows; }

    size_t level_num() const override { return 3; }

    double evaluate(const vector<2> &u, int level = 0) const override;
    double evaluate(const vector<2> &u, vector<2> &ddu,
                    int level = 0) const override;

    bool has_depth() const override { return !depth_image.empty(); }
    DepthSource depth_source() const override { return depth_source_type; }
    double depth(const vector<2> &u) const override {
        if (depth_image.empty())
            return 0.0;
        const int x = (int)std::lround(u.x());
        const int y = (int)std::lround(u.y());
        if (x < 0 || y < 0 || x >= depth_image.cols || y >= depth_image.rows)
            return 0.0;
        return (double)depth_image.at<float>(y, x);
    }

    double depth_confidence(const vector<2> &u) const override {
        // Physical RGB-D measurements keep their existing weight. This local
        // consistency heuristic is only for learned monocular metric depth.
        if (depth_source_type != DepthSource::MONOCULAR_METRIC)
            return 1.0;
        if (depth_image.empty())
            return 0.0;

        const int x = (int)std::lround(u.x());
        const int y = (int)std::lround(u.y());
        if (x < 0 || y < 0 || x >= depth_image.cols || y >= depth_image.rows)
            return 0.0;

        const double center = (double)depth_image.at<float>(y, x);
        if (!std::isfinite(center) || center <= 0.0)
            return 0.0;

        constexpr int radius = 2; // 5x5 neighborhood
        constexpr size_t min_valid_samples = 13;
        std::vector<double> samples;
        samples.reserve((2 * radius + 1) * (2 * radius + 1));
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int xx = x + dx;
                const int yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= depth_image.cols ||
                    yy >= depth_image.rows)
                    continue;
                const double value = (double)depth_image.at<float>(yy, xx);
                if (std::isfinite(value) && value > 0.0)
                    samples.emplace_back(value);
            }
        }

        if (samples.size() < min_valid_samples)
            return 0.0;

        std::sort(samples.begin(), samples.end());
        const double median = samples[samples.size() / 2];
        if (!std::isfinite(median) || median <= 1.0e-6)
            return 0.0;

        std::vector<double> deviations;
        deviations.reserve(samples.size());
        for (double value : samples)
            deviations.emplace_back(std::abs(value - median));
        std::sort(deviations.begin(), deviations.end());

        // 1.4826 converts MAD to an equivalent Gaussian sigma.
        const double relative_mad =
            1.4826 * deviations[deviations.size() / 2] / median;
        const double center_relative_error = std::abs(center - median) / median;

        // Learned depth near discontinuities is frequently inconsistent with
        // the feature's actual surface. Drop severe cases completely.
        constexpr double max_relative_mad = 0.12;
        constexpr double max_center_relative_error = 0.20;
        if (relative_mad >= max_relative_mad ||
            center_relative_error >= max_center_relative_error)
            return 0.0;

        const double dispersion_confidence =
            1.0 - relative_mad / max_relative_mad;
        const double center_confidence =
            1.0 - center_relative_error / max_center_relative_error;
        const double confidence = dispersion_confidence * center_confidence;

        // Accepted priors remain weak enough to preserve metric anchoring while
        // allowing visual geometry to override locally uncertain predictions.
        return std::max(0.10, std::min(1.0, confidence));
    }

    void detect_keypoints(std::vector<vector<2>> &keypoints,
                          size_t max_points = 1000,
                          double keypoint_distance = 10) const override;
    void track_keypoints(const Image *next_image,
                         const std::vector<vector<2>> &curr_keypoints,
                         std::vector<vector<2>> &next_keypoints,
                         std::vector<char> &result_status) const override;

    void preprocess(double clipLimit, int width, int height) override;
    void retain_place_recognition_source(bool retain) override;
    bool has_place_recognition_source() const override {
        return !raw.empty();
    }
    void correct_distortion(const matrix<3> &intrinsics,
                            const vector<4> &coeffs);
    void release_image_buffer() override;

    cv::Mat image;
    cv::Mat raw;
    cv::Mat depth_image; // CV_32FC1 meters, registered to image pixels
    DepthSource depth_source_type = DepthSource::SENSOR_METRIC;

  private:
    bool retain_raw_for_place_recognition_ = false;
    std::vector<cv::Mat> image_pyramid;
    std::vector<cv::Mat> image_levels;
    std::vector<vector<2>> scale_levels;

    typedef ceres::Grid2D<unsigned char, 1> Grid;
    std::vector<Grid> grid_levels;

    typedef ceres::BiCubicInterpolator<Grid> Interpolator;
    std::vector<Interpolator> interpolator_levels;

    static cv::CLAHE *clahe(double clipLimit, int width, int height);
    static cv::GFTTDetector *gftt(size_t max_points);
    static cv::FastFeatureDetector *fast();
    static cv::ORB *orb();
};

} // namespace xrslam::extra

#endif /* XRSLAM_EXTRA_OPENCV_IMAGE_H */
