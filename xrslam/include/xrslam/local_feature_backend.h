#ifndef XRSLAM_LOCAL_FEATURE_BACKEND_H
#define XRSLAM_LOCAL_FEATURE_BACKEND_H

#include <xrslam/xrslam.h>

namespace xrslam {

struct LocalFeatureMatch {
    // Index into an explicitly supplied reference feature list when a backend
    // performs track-conditioned matching. Keyframe-to-keyframe matchers may
    // leave this at its sentinel value and use reference_point instead.
    size_t reference_index = static_cast<size_t>(-1);
    vector<2> reference_point;
    vector<2> current_point;
    double confidence = 0.0;
};

// Front-end correspondence backend used by Frame. The default implementation
// delegates to Image::detect_keypoints()/track_keypoints(), preserving the
// existing GFTT + pyramidal-LK path exactly. Learned backends can be installed
// without coupling xrslam-core to a particular inference runtime.
class LocalFeatureBackend {
  public:
    virtual ~LocalFeatureBackend() = default;

    virtual void detect_keypoints(
        const Image *image, std::vector<vector<2>> &keypoints,
        size_t max_points, double keypoint_distance) = 0;

    // Recovery-anchor replenishment hook. Default backends preserve their
    // ordinary detector exactly. Learned backends may deliberately seed
    // native features on sparse retained anchors so a future matcher can map
    // native feature identity back to persistent XRSLAM Tracks without
    // descriptor sampling or spatial snapping.
    virtual void detect_recovery_anchor_keypoints(
        const Image *image, std::vector<vector<2>> &keypoints,
        size_t max_points, double keypoint_distance) {
        detect_keypoints(image, keypoints, max_points, keypoint_distance);
    }

    // next_keypoints may contain motion-predicted initial positions. Results
    // remain index-aligned with curr_keypoints so Frame can keep persistent
    // Track identities independent of the backend implementation.
    virtual void track_keypoints(
        const Image *image, const Image *next_image,
        const std::vector<vector<2>> &curr_keypoints,
        std::vector<vector<2>> &next_keypoints,
        std::vector<char> &result_status) = 0;

    // Optional wide-baseline recovery hooks. The default backend intentionally
    // does nothing, so builds without a learned matcher preserve the exact
    // GFTT + KLT behavior.
    // Non-zero means the core should retain an image buffer at this frame
    // cadence for lazy wide-baseline recovery. Default backends need no
    // retained anchors.
    virtual size_t recovery_anchor_stride() const { return 0; }

    virtual void cache_keyframe_features(const Image *image) {}
    virtual bool has_cached_features(const Image *image) { return false; }
    virtual bool match_keyframes(
        const Image *reference_image, const Image *current_image,
        std::vector<LocalFeatureMatch> &matches) {
        matches.clear();
        return false;
    }

    // Match descriptors sampled exactly at existing XRSLAM reference-track
    // coordinates against learned current-frame features. reference_index in
    // each returned match indexes reference_points directly.
    virtual bool match_track_keypoints(
        const Image *reference_image,
        const std::vector<vector<2>> &reference_points,
        const Image *current_image,
        std::vector<LocalFeatureMatch> &matches) {
        matches.clear();
        return false;
    }
};

std::shared_ptr<LocalFeatureBackend> local_feature_backend();
void set_local_feature_backend(std::shared_ptr<LocalFeatureBackend> backend);

} // namespace xrslam

#endif // XRSLAM_LOCAL_FEATURE_BACKEND_H
