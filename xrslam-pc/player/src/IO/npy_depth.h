#ifndef XRSLAM_PC_NPY_DEPTH_H
#define XRSLAM_PC_NPY_DEPTH_H

#include <opencv2/opencv.hpp>
#include <string>

bool load_npy_float32_2d(const std::string &filename, cv::Mat &output);

#endif // XRSLAM_PC_NPY_DEPTH_H
