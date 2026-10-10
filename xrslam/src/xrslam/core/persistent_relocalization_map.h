#ifndef XRSLAM_PERSISTENT_RELOCALIZATION_MAP_H
#define XRSLAM_PERSISTENT_RELOCALIZATION_MAP_H

#include <xrslam/core/keyframe_archive.h>
#include <xrslam/core/place_graph_4dof_shadow.h>
#include <xrslam/place_recognition.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace xrslam {

// One finished tracking session, in that session's local world frame.
// Old and new session coordinates must NOT be treated as interchangeable
// until a future relocalization stage establishes an explicit transform.
struct RelocalizationReferenceSession {
    size_t session_id = 0;
    KeyframeArchive archive;
    PlaceKeyframeStore places;
    // Transform from this session's local map frame into the output-global
    // frame at capture time. This is a value snapshot, not a live graph edge.
    std::optional<PlaceGraph4DoFCorrection> global_alignment;
};

struct RelocalizationReferenceMatch {
    size_t session_id = 0;
    const ArchivedKeyframe *geometry = nullptr;
    const PlaceKeyframe *place = nullptr;
};

// Owned by the FrontendWorker, not a disposable SlidingWindowTracker.
// Contains values (never Frame/Track/Image pointers), and is only touched on
// the frontend worker thread. Only the last published local-to-global
// mapping is snapshotted with each completed session; no graph is mutated.
class PersistentRelocalizationMap {
  public:
    bool capture(RelocalizationReferenceSession session) {
        if (session.archive.size() == 0 && session.places.empty())
            return false;
        session.session_id = ++next_session_id_;
        sessions_.emplace_back(std::move(session));
        return true;
    }

    // Retrieve the saved local->global alignment only for a specific
    // historical session. Missing is equivalent to identity output mapping.
    std::optional<PlaceGraph4DoFCorrection>
    global_alignment_for_session(size_t session_id) const {
        for (const auto &session : sessions_) {
            if (session.session_id == session_id)
                return session.global_alignment;
        }
        return std::nullopt;
    }

    size_t session_count() const { return sessions_.size(); }
    size_t last_session_id() const { return next_session_id_; }

    size_t archived_keyframe_count() const {
        size_t count = 0;
        for (const auto &session : sessions_)
            count += session.archive.size();
        return count;
    }

    size_t place_keyframe_count() const {
        size_t count = 0;
        for (const auto &session : sessions_)
            count += session.places.size();
        return count;
    }

    size_t candidate_reference_count() const {
        size_t count = 0;
        for (const auto &session : sessions_) {
            for (const auto &place : session.places.entries()) {
                const ArchivedKeyframe *geometry =
                    session.archive.get(place.frame_id);
                if (geometry && geometry->local_descriptors_complete &&
                    geometry->local_descriptors.valid() &&
                    geometry->local_descriptors.size() != 0 &&
                    !geometry->observations.empty()) {
                    ++count;
                }
            }
        }
        return count;
    }

    // Returns the snapshot AND its coordinate-frame identity together.
    // A candidate requires archived geometry and sampled local descriptors.
    // PnP inlier filtering and pose acceptance are separate future steps.
    RelocalizationReferenceMatch find(PlaceKey key) const {
        for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
            const PlaceKeyframe *place = it->places.find(key);
            if (!place)
                continue;
            const ArchivedKeyframe *geometry =
                it->archive.get(place->frame_id);
            if (!geometry ||
                !geometry->local_descriptors_complete ||
                !geometry->local_descriptors.valid() ||
                geometry->local_descriptors.size() == 0 ||
                geometry->observations.empty()) {
                return {};
            }
            return {it->session_id, geometry, place};
        }
        return {};
    }

  private:
    size_t next_session_id_ = 0;
    std::vector<RelocalizationReferenceSession> sessions_;
};

} // namespace xrslam

#endif // XRSLAM_PERSISTENT_RELOCALIZATION_MAP_H
