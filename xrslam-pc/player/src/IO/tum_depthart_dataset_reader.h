#ifndef XRSLAM_PC_TUM_DEPTHART_DATASET_READER_H
#define XRSLAM_PC_TUM_DEPTHART_DATASET_READER_H

#include <dataset_reader.h>
#include <deque>
#include <string>

class TUMDepthARTDatasetReader : public DatasetReader {
  public:
    TUMDepthARTDatasetReader(const std::string &path, void *yaml_config);

    NextDataType next() override;
    void get_image_resolution(int &width, int &height) override;
    std::pair<double, cv::Mat> read_image() override;
    std::pair<double, cv::Mat> read_depth() override;
    double depth_scale() const override { return 1.0; }
    XRSLAMDepthSource depth_source() const override {
        return XRSLAM_DEPTH_MONOCULAR_METRIC;
    }
    std::pair<double, XRSLAMGyroscope> read_gyroscope() override;
    std::pair<double, XRSLAMAcceleration> read_accelerometer() override;

  private:
    struct Frame {
        double t = 0.0;
        std::string rgb_path;
        std::string depth_path;
    };

    std::deque<Frame> frames;
    std::pair<double, cv::Mat> current_depth;
    int image_width = 0;
    int image_height = 0;
};

#endif // XRSLAM_PC_TUM_DEPTHART_DATASET_READER_H
