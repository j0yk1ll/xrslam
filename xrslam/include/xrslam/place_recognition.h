#ifndef XRSLAM_PLACE_RECOGNITION_H
#define XRSLAM_PLACE_RECOGNITION_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <xrslam/xrslam.h>

namespace xrslam {

using PlaceKey = std::uint64_t;

struct PlaceDescriptor {
    std::vector<float> values;

    bool empty() const { return values.empty(); }
    size_t dimension() const { return values.size(); }
};

enum class LocalDescriptorType : std::uint8_t {
    BINARY_U8 = 0,
    FLOAT32 = 1,
};

// Value-owned local descriptors sampled at explicitly supplied image points.
// source_indices maps every descriptor row back to the input point that
// produced it. Binary and float storage are both represented so the core
// archive does not depend on ORB, XFeat, or a particular matching runtime.
struct LocalDescriptorSet {
    LocalDescriptorType type = LocalDescriptorType::BINARY_U8;
    size_t dimension = 0;
    std::vector<size_t> source_indices;
    std::vector<std::uint8_t> binary_values;
    std::vector<float> float_values;

    size_t scalar_count() const {
        return type == LocalDescriptorType::BINARY_U8
                   ? binary_values.size()
                   : float_values.size();
    }

    size_t size() const {
        return dimension == 0 ? 0 : scalar_count() / dimension;
    }

    bool valid() const {
        if (dimension == 0 || scalar_count() % dimension != 0 ||
            source_indices.size() != size())
            return false;
        if (type == LocalDescriptorType::BINARY_U8)
            return float_values.empty();
        return binary_values.empty();
    }
};

struct PlaceCandidate {
    PlaceKey key = 0;
    float distance = 0.0f;
};

// Value-owned place-recognition record. The descriptor stays expressed only
// through the backend-agnostic PlaceDescriptor type; concrete EigenPlaces and
// database implementation details remain outside xrslam-core.
struct PlaceKeyframe {
    PlaceKey key = 0;
    size_t frame_id = 0;
    double timestamp = 0.0;
    PlaceDescriptor descriptor;
};

class PlaceDescriptorExtractor {
  public:
    virtual ~PlaceDescriptorExtractor() = default;

    virtual size_t dimension() const = 0;
    virtual PlaceDescriptor extract(const Image &image) = 0;
};

// Descriptor sampling for geometric verification is deliberately separate
// from LocalFeatureBackend. Enabling place recognition must not replace the
// ordinary GFTT detector or KLT temporal tracker.
class LocalDescriptorExtractor {
  public:
    virtual ~LocalDescriptorExtractor() = default;

    virtual LocalDescriptorType type() const = 0;
    virtual size_t dimension() const = 0;
    virtual LocalDescriptorSet
    extract(const Image &image,
            const std::vector<vector<2>> &points) = 0;
};

class PlaceDatabase {
  public:
    virtual ~PlaceDatabase() = default;

    virtual size_t dimension() const = 0;
    virtual size_t size() const = 0;
    virtual void clear() = 0;

    // Returns false for an invalid descriptor, a dimension mismatch, or a
    // duplicate key. Implementations must not partially insert on failure.
    virtual bool add(PlaceKey key, const PlaceDescriptor &descriptor) = 0;

    // Candidates are ordered from best to worst according to the concrete
    // database metric. No acceptance threshold is encoded at this layer;
    // geometric verification owns acceptance.
    virtual std::vector<PlaceCandidate>
    search(const PlaceDescriptor &descriptor, size_t max_results) const = 0;
};

class PlaceKeyframeStore {
  public:
    bool add(const PlaceKeyframe &keyframe);
    const PlaceKeyframe *find(PlaceKey key) const;

    void clear();
    size_t size() const { return keyframes_.size(); }
    bool empty() const { return keyframes_.empty(); }

    const std::vector<PlaceKeyframe> &entries() const { return keyframes_; }

  private:
    std::vector<PlaceKeyframe> keyframes_;
    std::unordered_map<PlaceKey, size_t> key_to_index_;
};

} // namespace xrslam

#endif // XRSLAM_PLACE_RECOGNITION_H
