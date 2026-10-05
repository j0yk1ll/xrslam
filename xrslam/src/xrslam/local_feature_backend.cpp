#include <xrslam/local_feature_backend.h>

#include <mutex>
#include <utility>

namespace xrslam {

namespace {

class ImageLocalFeatureBackend final : public LocalFeatureBackend {
  public:
    void detect_keypoints(const Image *image,
                          std::vector<vector<2>> &keypoints,
                          size_t max_points,
                          double keypoint_distance) override {
        image->detect_keypoints(keypoints, max_points, keypoint_distance);
    }

    void track_keypoints(
        const Image *image, const Image *next_image,
        const std::vector<vector<2>> &curr_keypoints,
        std::vector<vector<2>> &next_keypoints,
        std::vector<char> &result_status) override {
        image->track_keypoints(next_image, curr_keypoints, next_keypoints,
                               result_status);
    }
};

std::shared_ptr<LocalFeatureBackend> make_default_backend() {
    return std::make_shared<ImageLocalFeatureBackend>();
}

std::shared_ptr<LocalFeatureBackend> &backend_storage() {
    static std::shared_ptr<LocalFeatureBackend> backend =
        make_default_backend();
    return backend;
}

std::mutex &backend_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

std::shared_ptr<LocalFeatureBackend> local_feature_backend() {
    std::lock_guard<std::mutex> lock(backend_mutex());
    return backend_storage();
}

void set_local_feature_backend(std::shared_ptr<LocalFeatureBackend> backend) {
    std::lock_guard<std::mutex> lock(backend_mutex());
    backend_storage() =
        backend ? std::move(backend) : make_default_backend();
}

} // namespace xrslam
