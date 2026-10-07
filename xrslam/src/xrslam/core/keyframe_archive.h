#ifndef XRSLAM_KEYFRAME_ARCHIVE_H
#define XRSLAM_KEYFRAME_ARCHIVE_H

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include <xrslam/common.h>
#include <xrslam/estimation/state.h>
#include <xrslam/place_recognition.h>

namespace xrslam {

// Value-owned observation retained after live Frame/Track objects leave the
// sliding window. Local descriptor rows are stored separately on the owning
// ArchivedKeyframe and map back to this vector by observation index.
struct ArchivedLandmarkObservation {
    size_t track_id;
    size_t keypoint_index;
    vector<3> bearing;
    vector<2> pixel;
    vector<3> landmark_world;
};

// Persistent geometric-verification snapshot. No member owns or aliases a
// Frame, Track, Image, or solver object.
struct ArchivedKeyframe {
    size_t frame_id;
    double timestamp;
    PoseState body_pose;
    PoseState camera_pose;
    matrix<3> K;
    std::vector<ArchivedLandmarkObservation> observations;

    // A keyframe with zero eligible 3D observations is still complete; the
    // explicit flag distinguishes that case from "not extracted yet".
    bool local_descriptors_complete = false;
    LocalDescriptorSet local_descriptors;
};

class KeyframeArchive {
  public:
    // Returns true on first insertion. Existing records are refreshed in place
    // so active keyframes retain their newest optimized values without
    // increasing archive cardinality.
    bool upsert(ArchivedKeyframe keyframe) {
        const size_t frame_id = keyframe.frame_id;
        auto it = keyframes_.find(frame_id);
        if (it == keyframes_.end()) {
            keyframes_.emplace(frame_id, std::move(keyframe));
            insertion_order_.push_back(frame_id);
            return true;
        }

        // Geometry is refreshed after every optimization. Local descriptors
        // are immutable appearance data sampled once from the accepted
        // keyframe source, so preserve them across geometry refreshes.
        if (it->second.local_descriptors_complete &&
            !keyframe.local_descriptors_complete) {
            keyframe.local_descriptors_complete = true;
            keyframe.local_descriptors =
                std::move(it->second.local_descriptors);
        }
        it->second = std::move(keyframe);
        return false;
    }

    bool contains(size_t frame_id) const {
        return keyframes_.find(frame_id) != keyframes_.end();
    }

    size_t size() const { return keyframes_.size(); }

    const ArchivedKeyframe *get(size_t frame_id) const {
        const auto it = keyframes_.find(frame_id);
        if (it == keyframes_.end())
            return nullptr;
        return &it->second;
    }

    const std::vector<size_t> &insertion_order() const {
        return insertion_order_;
    }

  private:
    std::unordered_map<size_t, ArchivedKeyframe> keyframes_;
    std::vector<size_t> insertion_order_;
};

} // namespace xrslam

#endif // XRSLAM_KEYFRAME_ARCHIVE_H
