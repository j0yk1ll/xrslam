#ifndef XRSLAM_EXTRA_EIGENPLACES_DESCRIPTOR_EXTRACTOR_H
#define XRSLAM_EXTRA_EIGENPLACES_DESCRIPTOR_EXTRACTOR_H

#include <memory>
#include <string>

#include <xrslam/place_recognition.h>

namespace xrslam::extra {

// TorchScript-backed EigenPlaces ResNet50 / 512-D descriptor extraction.
// The implementation deliberately lives in xrslam-extra so xrslam-core stays
// independent of LibTorch and EigenPlaces.
class EigenPlacesDescriptorExtractor final : public PlaceDescriptorExtractor {
  public:
    static constexpr size_t kDescriptorDimension = 512;

    explicit EigenPlacesDescriptorExtractor(const std::string &model_path,
                                             bool use_cuda = false);
    ~EigenPlacesDescriptorExtractor() override;

    EigenPlacesDescriptorExtractor(const EigenPlacesDescriptorExtractor &) =
        delete;
    EigenPlacesDescriptorExtractor &
    operator=(const EigenPlacesDescriptorExtractor &) = delete;

    size_t dimension() const override { return kDescriptorDimension; }
    PlaceDescriptor extract(const Image &image) override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_EIGENPLACES_DESCRIPTOR_EXTRACTOR_H
