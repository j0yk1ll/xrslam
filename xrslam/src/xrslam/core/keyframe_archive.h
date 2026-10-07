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
        // keyframe source. Their source_indices, however, refer to the
        // observation vector that existed at extraction time, so they must be
        // remapped when a later optimization refresh changes that vector.
        if (it->second.local_descriptors_complete &&
            !keyframe.local_descriptors_complete) {
            keyframe.local_descriptors_complete = true;

            const ArchivedKeyframe &previous = it->second;
            const LocalDescriptorSet &previous_descriptors =
                previous.local_descriptors;

            LocalDescriptorSet remapped;
            remapped.type = previous_descriptors.type;
            remapped.dimension = previous_descriptors.dimension;

            std::unordered_map<size_t, size_t>
                keypoint_to_observation;
            keypoint_to_observation.reserve(
                keyframe.observations.size());
            for (size_t observation_index = 0;
                 observation_index < keyframe.observations.size();
                 ++observation_index) {
                keypoint_to_observation.emplace(
                    keyframe.observations[observation_index]
                        .keypoint_index,
                    observation_index);
            }

            if (previous_descriptors.valid()) {
                for (size_t row = 0;
                     row < previous_descriptors.size();
                     ++row) {
                    const size_t previous_observation_index =
                        previous_descriptors.source_indices[row];
                    if (previous_observation_index >=
                        previous.observations.size()) {
                        continue;
                    }

                    const ArchivedLandmarkObservation &old_observation =
                        previous.observations[
                            previous_observation_index];
                    const auto new_it =
                        keypoint_to_observation.find(
                            old_observation.keypoint_index);
                    if (new_it == keypoint_to_observation.end())
                        continue;

                    const size_t new_observation_index =
                        new_it->second;
                    if (keyframe.observations[new_observation_index]
                            .track_id != old_observation.track_id) {
                        continue;
                    }

                    remapped.source_indices.emplace_back(
                        new_observation_index);

                    const size_t begin =
                        row * previous_descriptors.dimension;
                    const size_t end =
                        begin + previous_descriptors.dimension;
                    if (previous_descriptors.type ==
                        LocalDescriptorType::BINARY_U8) {
                        remapped.binary_values.insert(
                            remapped.binary_values.end(),
                            previous_descriptors.binary_values.begin() +
                                begin,
                            previous_descriptors.binary_values.begin() +
                                end);
                    } else {
                        remapped.float_values.insert(
                            remapped.float_values.end(),
                            previous_descriptors.float_values.begin() +
                                begin,
                            previous_descriptors.float_values.begin() +
                                end);
                    }
                }
            }

            keyframe.local_descriptors = std::move(remapped);
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
