#include <xrslam/extra/pidnet_segmenter.h>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5 && argc != 6 && argc != 8) {
        std::fprintf(stderr,
            "usage: %s MODEL.ts IMAGE LABELS.png [cpu|cuda [classes [canvas_width canvas_height]]]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    try {
        const std::string device = argc >= 5 ? argv[4] : "cpu";
        const int classes = argc >= 6 ? std::stoi(argv[5]) : 19;
        const int width = argc == 8 ? std::stoi(argv[6]) : 1024;
        const int height = argc == 8 ? std::stoi(argv[7]) : 512;
        const cv::Mat source = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
        if (source.empty())
            throw std::runtime_error("failed to read input image");
        xrslam::extra::PidNetSegmenter segmenter(
            argv[1], device, static_cast<std::uint32_t>(classes), width, height);
        auto result = segmenter.segment(source);
        if (result.label.type() != CV_8UC1 ||
            result.label.size() != source.size() ||
            result.confidence.type() != CV_32FC1 ||
            result.confidence.size() != source.size())
            throw std::runtime_error("mask shape/type contract violated");
        double min_conf = 0, max_conf = 0;
        cv::minMaxLoc(result.confidence, &min_conf, &max_conf);
        if (!std::isfinite(min_conf) || !std::isfinite(max_conf) ||
            min_conf < 0 || max_conf > 1)
            throw std::runtime_error("confidence out of range");
        double min_label = 0, max_label = 0;
        cv::minMaxLoc(result.label, &min_label, &max_label);
        if (min_label < 0 || max_label >= classes)
            throw std::runtime_error("semantic class label out of range");
        if (!cv::imwrite(argv[3], result.label))
            throw std::runtime_error("cannot write label PNG");
        std::printf("PIDNet smoke PASS: device=%s image=%dx%d classes=%d "
                    "label_range=[%.0f,%.0f] confidence_range=[%.6f,%.6f] "
                    "output=%s\n",
                    device.c_str(), source.cols, source.rows, classes,
                    min_label, max_label, min_conf, max_conf, argv[3]);
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "PIDNet smoke FAIL: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
