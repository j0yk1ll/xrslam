#ifndef XRSLAM_EXTRA_USEARCH_PLACE_DATABASE_H
#define XRSLAM_EXTRA_USEARCH_PLACE_DATABASE_H

#include <cstddef>
#include <memory>
#include <vector>

#include <xrslam/place_recognition.h>

namespace xrslam::extra {

// USearch-backed implementation of the dependency-independent PlaceDatabase
// interface. The first retrieval experiments intentionally use unquantized
// 512-D float32 descriptors and squared-L2 distance.
class USearchPlaceDatabase final : public PlaceDatabase {
  public:
    static constexpr size_t kDescriptorDimension = 512;

    USearchPlaceDatabase();
    ~USearchPlaceDatabase() override;

    USearchPlaceDatabase(USearchPlaceDatabase &&) noexcept;
    USearchPlaceDatabase &operator=(USearchPlaceDatabase &&) noexcept;

    USearchPlaceDatabase(const USearchPlaceDatabase &) = delete;
    USearchPlaceDatabase &operator=(const USearchPlaceDatabase &) = delete;

    size_t dimension() const override;
    size_t size() const override;
    void clear() override;

    // Capacity reservation is a backend-specific optimization; it is not part
    // of the dependency-independent PlaceDatabase interface.
    void reserve(size_t capacity);

    bool add(size_t frame_id, const PlaceDescriptor &descriptor) override;
    std::vector<PlaceCandidate>
    search(const PlaceDescriptor &descriptor, size_t k) const override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<PlaceDatabase> make_usearch_place_database();

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_USEARCH_PLACE_DATABASE_H
