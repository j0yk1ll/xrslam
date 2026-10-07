#include <xrslam/place_recognition.h>

#include <algorithm>
#include <limits>

namespace xrslam {

namespace {

size_t popcount_u8(std::uint8_t value) {
    value =
        static_cast<std::uint8_t>(
            value -
            ((value >> 1) & 0x55u));
    value =
        static_cast<std::uint8_t>(
            (value & 0x33u) +
            ((value >> 2) & 0x33u));
    return static_cast<size_t>(
        (value + (value >> 4)) & 0x0fu);
}

size_t hamming_distance(
    const std::uint8_t *a,
    const std::uint8_t *b,
    size_t dimension) {
    size_t distance = 0;
    for (size_t i = 0; i < dimension; ++i) {
        distance += popcount_u8(
            static_cast<std::uint8_t>(a[i] ^ b[i]));
    }
    return distance;
}

} // namespace

LocalDescriptorMatchResult
match_binary_descriptors_mutual_nn(
    const LocalDescriptorSet &reference,
    const LocalDescriptorSet &current) {
    LocalDescriptorMatchResult result;

    if (!reference.valid() || !current.valid() ||
        reference.type != LocalDescriptorType::BINARY_U8 ||
        current.type != LocalDescriptorType::BINARY_U8 ||
        reference.dimension != current.dimension ||
        reference.size() == 0 || current.size() == 0) {
        return result;
    }

    const size_t reference_count = reference.size();
    const size_t current_count = current.size();
    const size_t dimension = reference.dimension;
    const size_t invalid_index =
        static_cast<size_t>(-1);
    const size_t max_distance =
        std::numeric_limits<size_t>::max();

    std::vector<size_t> best_current(
        reference_count, invalid_index);
    std::vector<size_t> best_current_distance(
        reference_count, max_distance);
    std::vector<size_t> best_reference(
        current_count, invalid_index);
    std::vector<size_t> best_reference_distance(
        current_count, max_distance);

    for (size_t i = 0; i < reference_count; ++i) {
        const std::uint8_t *reference_descriptor =
            reference.binary_values.data() +
            i * dimension;

        for (size_t j = 0; j < current_count; ++j) {
            const std::uint8_t *current_descriptor =
                current.binary_values.data() +
                j * dimension;
            const size_t distance =
                hamming_distance(
                    reference_descriptor,
                    current_descriptor,
                    dimension);

            if (distance < best_current_distance[i]) {
                best_current_distance[i] = distance;
                best_current[i] = j;
            }
            if (distance < best_reference_distance[j]) {
                best_reference_distance[j] = distance;
                best_reference[j] = i;
            }
        }
    }

    result.nearest_neighbors.reserve(reference_count);
    result.mutual_matches.reserve(
        std::min(reference_count, current_count));

    for (size_t i = 0; i < reference_count; ++i) {
        const size_t j = best_current[i];
        if (j == invalid_index)
            continue;

        LocalDescriptorMatch match;
        match.reference_descriptor_index = i;
        match.current_descriptor_index = j;
        match.distance = best_current_distance[i];
        result.nearest_neighbors.emplace_back(match);

        if (j < best_reference.size() &&
            best_reference[j] == i) {
            result.mutual_matches.emplace_back(match);
        }
    }

    return result;
}

bool PlaceKeyframeStore::add(const PlaceKeyframe &keyframe) {
    if (key_to_index_.count(keyframe.key) != 0)
        return false;

    const size_t index = keyframes_.size();
    keyframes_.push_back(keyframe);
    key_to_index_.emplace(keyframe.key, index);
    return true;
}

const PlaceKeyframe *PlaceKeyframeStore::find(PlaceKey key) const {
    const auto it = key_to_index_.find(key);
    if (it == key_to_index_.end())
        return nullptr;
    return &keyframes_[it->second];
}

void PlaceKeyframeStore::clear() {
    keyframes_.clear();
    key_to_index_.clear();
}

} // namespace xrslam
