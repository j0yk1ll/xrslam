#include <xrslam/extra/usearch_place_database.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <usearch/index_dense.hpp>

namespace xrslam::extra {

namespace {

using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;
using unum::usearch::scalar_kind_t;

index_dense_t make_index() {
    metric_punned_t metric(
        USearchPlaceDatabase::kDescriptorDimension,
        metric_kind_t::l2sq_k,
        scalar_kind_t::f32_k);
    return index_dense_t::make(metric);
}

bool valid_descriptor(const PlaceDescriptor &descriptor) {
    return descriptor.values.size() ==
           USearchPlaceDatabase::kDescriptorDimension;
}

} // namespace

class USearchPlaceDatabase::Impl {
  public:
    index_dense_t index = make_index();
    std::unordered_set<size_t> frame_ids;
};

USearchPlaceDatabase::USearchPlaceDatabase()
    : impl_(std::make_unique<Impl>()) {}

USearchPlaceDatabase::~USearchPlaceDatabase() = default;
USearchPlaceDatabase::USearchPlaceDatabase(USearchPlaceDatabase &&) noexcept =
    default;
USearchPlaceDatabase &USearchPlaceDatabase::operator=(
    USearchPlaceDatabase &&) noexcept = default;

size_t USearchPlaceDatabase::dimension() const {
    return kDescriptorDimension;
}

size_t USearchPlaceDatabase::size() const {
    return impl_->index.size();
}

void USearchPlaceDatabase::clear() {
    impl_->index.clear();
    impl_->frame_ids.clear();
}

void USearchPlaceDatabase::reserve(size_t capacity) {
    if (!impl_->index.try_reserve(capacity))
        throw std::runtime_error("USearch failed to reserve index capacity");
    impl_->frame_ids.reserve(capacity);
}

bool USearchPlaceDatabase::add(
    size_t frame_id, const PlaceDescriptor &descriptor) {
    if (!valid_descriptor(descriptor))
        return false;
    if (impl_->frame_ids.find(frame_id) != impl_->frame_ids.end())
        return false;

    auto result = impl_->index.add(frame_id, descriptor.values.data());
    if (!result)
        return false;

    impl_->frame_ids.emplace(frame_id);
    return true;
}

std::vector<PlaceCandidate> USearchPlaceDatabase::search(
    const PlaceDescriptor &descriptor, size_t k) const {
    std::vector<PlaceCandidate> candidates;
    if (!valid_descriptor(descriptor) || k == 0 || impl_->index.size() == 0)
        return candidates;

    const size_t wanted = std::min(k, impl_->index.size());
    auto result = impl_->index.search(descriptor.values.data(), wanted);
    if (!result)
        return candidates;

    candidates.reserve(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        // PlaceCandidate is ordered as keyframe ID followed by distance.
        // Aggregate construction avoids coupling this backend to member names.
        candidates.emplace_back(PlaceCandidate{
            static_cast<size_t>(result[i].member.key),
            static_cast<float>(result[i].distance)});
    }
    return candidates;
}

std::shared_ptr<PlaceDatabase> make_usearch_place_database() {
    return std::make_shared<USearchPlaceDatabase>();
}

} // namespace xrslam::extra
