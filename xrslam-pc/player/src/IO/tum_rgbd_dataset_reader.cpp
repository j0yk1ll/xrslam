#include <tum_rgbd_dataset_reader.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

std::vector<TUMRGBDDatasetReader::TimedPath>
TUMRGBDDatasetReader::load_index(const std::string &filename,
                                 const std::string &root) {
    std::vector<TimedPath> result;
    std::ifstream input(filename);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream stream(line);
        TimedPath item;
        std::string relative_path;
        if (!(stream >> item.t >> relative_path))
            continue;
        item.path = root + "/" + relative_path;
        result.push_back(std::move(item));
    }
    return result;
}

std::vector<TUMRGBDDatasetReader::RGBDFrame>
TUMRGBDDatasetReader::associate(const std::vector<TimedPath> &rgb,
                                const std::vector<TimedPath> &depth,
                                double max_delta_seconds) {
    std::vector<RGBDFrame> result;
    if (rgb.empty() || depth.empty())
        return result;

    size_t depth_index = 0;
    for (const TimedPath &rgb_item : rgb) {
        while (depth_index + 1 < depth.size() &&
               std::abs(depth[depth_index + 1].t - rgb_item.t) <=
                   std::abs(depth[depth_index].t - rgb_item.t)) {
            ++depth_index;
        }

        const double delta = std::abs(depth[depth_index].t - rgb_item.t);
        if (delta <= max_delta_seconds) {
            result.push_back(
                {rgb_item.t, rgb_item.path, depth[depth_index].path});
        }
    }
    return result;
}

TUMRGBDDatasetReader::TUMRGBDDatasetReader(const std::string &tum_path,
                                           void *yaml_config) {
    (void)yaml_config;

    const auto rgb = load_index(tum_path + "/rgb.txt", tum_path);
    const auto depth = load_index(tum_path + "/depth.txt", tum_path);

    // TUM's association script commonly permits a small RGB/depth timestamp
    // offset. 20 ms is conservative relative to the ~30 Hz streams.
    const auto associated = associate(rgb, depth, 0.020);
    for (const auto &frame : associated)
        frames.push_back(frame);
}

DatasetReader::NextDataType TUMRGBDDatasetReader::next() {
    return frames.empty() ? NextDataType::END : NextDataType::CAMERA;
}

void TUMRGBDDatasetReader::get_image_resolution(int &width, int &height) {
    width = image_width;
    height = image_height;
}

std::pair<double, cv::Mat> TUMRGBDDatasetReader::read_image() {
    if (frames.empty())
        return {};

    const RGBDFrame frame = frames.front();
    frames.pop_front();

    cv::Mat rgb = cv::imread(frame.rgb_path, cv::IMREAD_GRAYSCALE);
    cv::Mat depth = cv::imread(frame.depth_path, cv::IMREAD_UNCHANGED);

    if (rgb.empty() || depth.empty() || depth.type() != CV_16UC1 ||
        rgb.cols != depth.cols || rgb.rows != depth.rows) {
        current_depth = {};
        return {};
    }

    image_width = rgb.cols;
    image_height = rgb.rows;
    current_depth = {frame.t, depth};
    return {frame.t, rgb};
}

std::pair<double, cv::Mat> TUMRGBDDatasetReader::read_depth() {
    auto depth = current_depth;
    current_depth = {};
    return depth;
}

std::pair<double, XRSLAMGyroscope> TUMRGBDDatasetReader::read_gyroscope() {
    return {};
}

std::pair<double, XRSLAMAcceleration>
TUMRGBDDatasetReader::read_accelerometer() {
    return {};
}
