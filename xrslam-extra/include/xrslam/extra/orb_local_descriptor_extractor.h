#ifndef XRSLAM_EXTRA_ORB_LOCAL_DESCRIPTOR_EXTRACTOR_H
#define XRSLAM_EXTRA_ORB_LOCAL_DESCRIPTOR_EXTRACTOR_H

#include <memory>

#include <xrslam/place_recognition.h>

namespace xrslam::extra {

// Classical baseline for place-recognition geometry: compute ORB descriptors
// at caller-supplied XRSLAM/GFTT landmark pixels. This class never detects or
// inserts frontend keypoints.
class OrbLocalDescriptorExtractor final : public LocalDescriptorExtractor {
  public:
    OrbLocalDescriptorExtractor();
    ~OrbLocalDescriptorExtractor() override;

    OrbLocalDescriptorExtractor(
        const OrbLocalDescriptorExtractor &) = delete;
    OrbLocalDescriptorExtractor &operator=(
        const OrbLocalDescriptorExtractor &) = delete;

    LocalDescriptorType type() const override;
    size_t dimension() const override;
    LocalDescriptorSet
    extract(const Image &image,
            const std::vector<vector<2>> &points) override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_ORB_LOCAL_DESCRIPTOR_EXTRACTOR_H
