#include <xrslam/extra/eigenplaces_descriptor_extractor.h>
#include <xrslam/extra/opencv_image.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s MODEL.ts IMAGE [descriptor.txt]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    cv::Mat raw = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        std::fprintf(stderr, "failed to read image: %s\n", image_path.c_str());
        return EXIT_FAILURE;
    }

    auto image = std::make_shared<xrslam::extra::OpenCvImage>();
    image->raw = raw.clone();
    if (raw.channels() == 1) {
        image->image = raw.clone();
    } else if (raw.channels() == 3) {
        cv::cvtColor(raw, image->image, cv::COLOR_BGR2GRAY);
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, image->image, cv::COLOR_BGRA2GRAY);
    } else {
        std::fprintf(stderr, "unsupported image channel count: %d\n",
                     raw.channels());
        return EXIT_FAILURE;
    }

    try {
        xrslam::extra::EigenPlacesDescriptorExtractor extractor(model_path);
        xrslam::PlaceDescriptor descriptor = extractor.extract(*image);

        double norm_sq = 0.0;
        for (float value : descriptor.values)
            norm_sq += static_cast<double>(value) * value;
        const double norm = std::sqrt(norm_sq);

        std::printf("dimension: %zu\n", descriptor.dimension());
        std::printf("norm: %.9f\n", norm);
        std::printf("first8:");
        const size_t preview = std::min<size_t>(8, descriptor.values.size());
        for (size_t i = 0; i < preview; ++i)
            std::printf(" %.8f", descriptor.values[i]);
        std::printf("\n");

        if (argc == 4) {
            std::ofstream output(argv[3]);
            if (!output) {
                std::fprintf(stderr, "failed to open output: %s\n", argv[3]);
                return EXIT_FAILURE;
            }
            output << std::setprecision(9);
            for (float value : descriptor.values)
                output << value << '\n';
        }

        if (descriptor.dimension() !=
                xrslam::extra::EigenPlacesDescriptorExtractor::
                    kDescriptorDimension ||
            std::abs(norm - 1.0) > 1.0e-4) {
            std::fprintf(stderr, "EigenPlaces descriptor validation failed\n");
            return EXIT_FAILURE;
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "EigenPlaces error: %s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("EigenPlaces C++ smoke test: PASS\n");
    return EXIT_SUCCESS;
}
