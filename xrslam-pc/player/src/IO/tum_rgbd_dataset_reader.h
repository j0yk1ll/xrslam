#ifndef XRSLAM_PC_TUM_RGBD_DATASET_READER_H
#define XRSLAM_PC_TUM_RGBD_DATASET_READER_H

#include <dataset_reader.h>
#include <deque>
#include <string>

class TUMRGBDDatasetReader : public DatasetReader {
  public:
    TUMRGBDDatasetReader(const std::string &path, void *yaml_config);

    NextDataType next() override;
    void get_image_resolution(int &width, int &height) override;
    std::pair<double, cv::Mat> read_image() override;
    std::pair<double, cv::Mat> read_depth() override;
    double depth_scale() const override { return 1.0 / 5000.0; }
    std::pair<double, XRSLAMGyroscope> read_gyroscope() override;
    std::pair<double, XRSLAMAcceleration> read_accelerometer() override;

  private:
    struct TimedPath {
        double t = 0.0;
        std::string path;
    };
    struct RGBDFrame {
        double t = 0.0;
        std::string rgb_path;
        std::string depth_path;
    };

    static std::vector<TimedPath> load_index(const std::string &filename,
                                             const std::string &root);
    static std::vector<RGBDFrame>
    associate(const std::vector<TimedPath> &rgb,
              const std::vector<TimedPath> &depth,
              double max_delta_seconds);

    std::deque<RGBDFrame> frames;
    std::pair<double, cv::Mat> current_depth;
    int image_width = 0;
    int image_height = 0;
};

#endif // XRSLAM_PC_TUM_RGBD_DATASET_READER_H
