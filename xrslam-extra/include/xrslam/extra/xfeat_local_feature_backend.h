#ifndef XRSLAM_EXTRA_XFEAT_LOCAL_FEATURE_BACKEND_H
#define XRSLAM_EXTRA_XFEAT_LOCAL_FEATURE_BACKEND_H

#include <memory>
#include <string>
#include <xrslam/local_feature_backend.h>

namespace xrslam::extra {

// Create an XFeat detector backed by an ONNX model exported with the sparse
// detectAndCompute interface: outputs are keypoints, descriptors and scores.
// Temporal tracking intentionally remains Image::track_keypoints() (KLT for
// OpenCvImage) so 0035 isolates the detector change.
std::shared_ptr<LocalFeatureBackend>
make_xfeat_local_feature_backend(const std::string &model_path);

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_XFEAT_LOCAL_FEATURE_BACKEND_H
