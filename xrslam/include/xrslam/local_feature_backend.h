#ifndef XRSLAM_LOCAL_FEATURE_BACKEND_H
#define XRSLAM_LOCAL_FEATURE_BACKEND_H

#include <xrslam/xrslam.h>

namespace xrslam {

// Front-end correspondence backend used by Frame. The default implementation
// delegates to Image::detect_keypoints()/track_keypoints(), preserving the
// existing GFTT + pyramidal-LK path exactly. The backend remains injectable for other frontends without
// coupling xrslam-core to a specific detector implementation.
class LocalFeatureBackend {
  public:
    virtual ~LocalFeatureBackend() = default;

    virtual void detect_keypoints(
        const Image *image, std::vector<vector<2>> &keypoints,
        size_t max_points, double keypoint_distance) = 0;

    // next_keypoints may contain motion-predicted initial positions. Results
    // remain index-aligned with curr_keypoints so Frame can keep persistent
    // Track identities independent of the backend implementation.
    virtual void track_keypoints(
        const Image *image, const Image *next_image,
        const std::vector<vector<2>> &curr_keypoints,
        std::vector<vector<2>> &next_keypoints,
        std::vector<char> &result_status) = 0;

};

std::shared_ptr<LocalFeatureBackend> local_feature_backend();
void set_local_feature_backend(std::shared_ptr<LocalFeatureBackend> backend);

} // namespace xrslam

#endif // XRSLAM_LOCAL_FEATURE_BACKEND_H
