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
