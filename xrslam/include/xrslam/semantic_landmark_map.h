#ifndef XRSLAM_SEMANTIC_LANDMARK_MAP_H
#define XRSLAM_SEMANTIC_LANDMARK_MAP_H

#include <xrslam/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xrslam {

// A *single* archived observation. Position is in the owning VIO session's
// local world frame: it is not a graph-corrected global coordinate.
struct SemanticLandmarkVote {
    std::uint64_t session_id = 0;
    std::uint64_t frame_id = 0;
    std::size_t track_id = 0;
    std::uint8_t label = 0;
    float confidence = 0.0f; // top-1 softmax probability at this pixel
    std::array<double, 3> position_session_world{};
};

// A semantic *consensus*, not a calibrated posterior distribution. Multiple
// views of one track contribute independent capped class votes, and the label
// becomes stable only after at least two matching accepted keyframe views.
struct SemanticFusedLandmark {
    std::uint64_t session_id = 0;
    std::size_t track_id = 0;
    std::array<double, 3> position_session_world{};
    std::uint64_t latest_frame_id = 0;
    std::size_t observation_count = 0;
    std::uint8_t winning_label = 0;
    std::size_t winning_label_views = 0;
    double consensus_fraction = 0.0; // winning class weight / all class weights
    double mean_pixel_confidence = 0.0;
    bool stable = false;
};

// Pure C++ value-owned fusion. Caller owns synchronization. Idempotent by
// (session, track, frame), even when a mask is processed more than once.
// No Frame/Track pointers, LibTorch, OpenCV, ROS, or estimator dependencies.
class SemanticLandmarkMap {
  public:
    explicit SemanticLandmarkMap(std::uint32_t num_classes)
        : num_classes_(num_classes) {
        if (num_classes == 0 || num_classes > 255)
            throw std::invalid_argument("semantic classes must be 1..255");
    }

    // Returns true iff a new, valid view was fused. Rejects non-finite
    // geometry/confidence, out-of-taxonomy labels, and duplicate views.
    bool observe(const SemanticLandmarkVote &vote) {
        if (vote.session_id == 0 ||
            vote.label >= num_classes_ ||
            !std::isfinite(vote.confidence) ||
            vote.confidence < kMinimumConfidence ||
            vote.confidence > 1.0f ||
            !std::all_of(vote.position_session_world.begin(),
                         vote.position_session_world.end(),
                         [](double v) { return std::isfinite(v); }))
            return false;

        const auto id = std::make_pair(vote.session_id, vote.track_id);
        auto it = landmarks_.find(id);
        if (it == landmarks_.end()) {
            Evidence ev;
            ev.label_weights.resize(num_classes_, 0.0);
            ev.label_views.resize(num_classes_, 0);
            it = landmarks_.emplace(id, std::move(ev)).first;
        }
        Evidence &ev = it->second;
        if (!ev.seen_frames.insert(vote.frame_id).second)
            return false;

        const bool was_stable = stable(ev);
        const double weight = static_cast<double>(vote.confidence);
        ev.label_weights[vote.label] += weight;
        ++ev.label_views[vote.label];
        ev.total_weight += weight;
        ++ev.observation_count;
        for (std::size_t d = 0; d < 3; ++d) {
            // Weighted session-local position (not an optimization variable).
            ev.weighted_position[d] +=
                weight * vote.position_session_world[d];
        }
        ev.latest_frame_id = std::max(ev.latest_frame_id, vote.frame_id);
        if (was_stable != stable(ev)) {
            if (stable(ev)) ++stable_count_;
            else --stable_count_;
        }
        return true;
    }

    std::optional<SemanticFusedLandmark> find(
        std::uint64_t session_id, std::size_t track_id) const {
        const auto it = landmarks_.find({session_id, track_id});
        if (it == landmarks_.end()) return std::nullopt;
        return project(session_id, track_id, it->second);
    }

    std::size_t size() const { return landmarks_.size(); }
    std::size_t stable_count() const { return stable_count_; }
    std::uint32_t num_classes() const { return num_classes_; }

    // Copy-only map export for a semantic map consumer; keyed by session and
    // track identity. Never returns references to mutable worker state.
    std::vector<SemanticFusedLandmark> snapshot(
        std::optional<std::uint64_t> session_id = std::nullopt,
        bool stable_only = false) const {
        std::vector<SemanticFusedLandmark> out;
        for (const auto &[key, ev] : landmarks_) {
            if (session_id && key.first != *session_id) continue;
            if (stable_only && !stable(ev)) continue;
            out.emplace_back(project(key.first, key.second, ev));
        }
        return out;
    }

  private:
    static constexpr float kMinimumConfidence = 0.50f;
    static constexpr double kMinimumConsensus = 0.65;
    static constexpr std::size_t kMinimumAgreeingViews = 2;

    struct Evidence {
        std::vector<double> label_weights;
        std::vector<std::size_t> label_views;
        std::set<std::uint64_t> seen_frames;
        std::array<double, 3> weighted_position{};
        double total_weight = 0.0;
        std::size_t observation_count = 0;
        std::uint64_t latest_frame_id = 0;
    };

    std::size_t winning_class(const Evidence &ev) const {
        return static_cast<std::size_t>(std::distance(
            ev.label_weights.begin(),
            std::max_element(ev.label_weights.begin(), ev.label_weights.end())));
    }
    bool stable(const Evidence &ev) const {
        if (ev.total_weight <= 0.0) return false;
        const auto winner = winning_class(ev);
        return ev.label_views[winner] >= kMinimumAgreeingViews &&
            ev.label_weights[winner] / ev.total_weight >= kMinimumConsensus;
    }
    SemanticFusedLandmark project(std::uint64_t session_id,
                                   std::size_t track_id,
                                   const Evidence &ev) const {
        SemanticFusedLandmark out;
        out.session_id = session_id;
        out.track_id = track_id;
        out.latest_frame_id = ev.latest_frame_id;
        out.observation_count = ev.observation_count;
        const auto winner = winning_class(ev);
        out.winning_label = static_cast<std::uint8_t>(winner);
        out.winning_label_views = ev.label_views[winner];
        out.consensus_fraction = ev.total_weight > 0.0 ?
            ev.label_weights[winner] / ev.total_weight : 0.0;
        out.mean_pixel_confidence = ev.observation_count ?
            ev.total_weight / static_cast<double>(ev.observation_count) : 0.0;
        for (std::size_t d = 0; d < 3; ++d)
            out.position_session_world[d] =
                ev.weighted_position[d] / ev.total_weight;
        out.stable = stable(ev);
        return out;
    }

    const std::uint32_t num_classes_;
    std::map<std::pair<std::uint64_t, std::size_t>, Evidence> landmarks_;
    std::size_t stable_count_ = 0;
};

} // namespace xrslam

#endif // XRSLAM_SEMANTIC_LANDMARK_MAP_H
