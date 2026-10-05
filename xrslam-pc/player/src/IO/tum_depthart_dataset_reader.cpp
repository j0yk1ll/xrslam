#include <tum_depthart_dataset_reader.h>
#include <npy_depth.h>

#include <fstream>
#include <sstream>

namespace {

std::string basename_stem(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    const size_t dot = path.find_last_of('.');
    const size_t end =
        dot == std::string::npos || dot < begin ? path.size() : dot;
    return path.substr(begin, end - begin);
}

} // namespace

TUMDepthARTDatasetReader::TUMDepthARTDatasetReader(
    const std::string &tum_path, void *yaml_config) {
    (void)yaml_config;

    std::ifstream input(tum_path + "/rgb.txt");
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream stream(line);
        double timestamp = 0.0;
        std::string relative_rgb;
        if (!(stream >> timestamp >> relative_rgb))
            continue;

        Frame frame;
        frame.t = timestamp;
        frame.rgb_path = tum_path + "/" + relative_rgb;
        frame.depth_path =
            tum_path + "/depthart/" + basename_stem(relative_rgb) + ".npy";
        frames.push_back(std::move(frame));
    }
}

DatasetReader::NextDataType TUMDepthARTDatasetReader::next() {
    return frames.empty() ? NextDataType::END : NextDataType::CAMERA;
}

void TUMDepthARTDatasetReader::get_image_resolution(int &width, int &height) {
    width = image_width;
    height = image_height;
}

std::pair<double, cv::Mat> TUMDepthARTDatasetReader::read_image() {
    if (frames.empty())
        return {};

    const Frame frame = frames.front();
    frames.pop_front();

    cv::Mat rgb = cv::imread(frame.rgb_path, cv::IMREAD_GRAYSCALE);
    cv::Mat depth;
    if (rgb.empty() || !load_npy_float32_2d(frame.depth_path, depth) ||
        depth.type() != CV_32FC1 || rgb.cols != depth.cols ||
        rgb.rows != depth.rows) {
        current_depth = {};
        return {};
    }

    image_width = rgb.cols;
    image_height = rgb.rows;
    current_depth = {frame.t, depth};
    return {frame.t, rgb};
}

std::pair<double, cv::Mat> TUMDepthARTDatasetReader::read_depth() {
    auto depth = current_depth;
    current_depth = {};
    return depth;
}

std::pair<double, XRSLAMGyroscope>
TUMDepthARTDatasetReader::read_gyroscope() {
    return {};
}

std::pair<double, XRSLAMAcceleration>
TUMDepthARTDatasetReader::read_accelerometer() {
    return {};
}
