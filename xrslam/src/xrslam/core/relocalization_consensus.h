#ifndef XRSLAM_RELOCALIZATION_CONSENSUS_H
#define XRSLAM_RELOCALIZATION_CONSENSUS_H

#include <xrslam/estimation/state.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace xrslam {

// Every pose is expressed in the historical session's local map coordinates.
// No implicit transform between different sessions is permitted.
struct RelocalizationInlierLandmark {
    size_t current_keypoint_index = 0;
    size_t original_track_id = 0;
    vector<3> world_point = vector<3>::Zero();
};

struct RelocalizationPnPHypothesis {
    size_t session_id = 0;
    size_t reference_frame_id = 0;
    size_t current_frame_id = 0;
    double timestamp = 0.0;
    size_t inliers = 0;
    double reprojection_rmse_px = 0.0;
    PoseState camera_pose;
    std::vector<RelocalizationInlierLandmark> landmarks;
};

struct RelocalizationConfirmedSeed {
    RelocalizationPnPHypothesis first;
    RelocalizationPnPHypothesis second;
    double dt_s = 0.0;
    double position_difference_m = 0.0;
    double orientation_difference_deg = 0.0;
};

// Same-session two-frame consensus. This is a candidate recovery seed, NOT
// estimator reinitialization. The VIO must still be bootstrapped separately.
class RelocalizationConsensus {
  public:
    void reset() {
        previous_.clear();
        confirmed_.reset();
    }

    const std::optional<RelocalizationConfirmedSeed> &confirmed() const {
        return confirmed_;
    }

    std::optional<RelocalizationConfirmedSeed> observe(
        const std::vector<RelocalizationPnPHypothesis> &hypotheses) {
        if (confirmed_) return std::nullopt; // one confirmation per loss

        // Compare only distinct camera frames, ordered by timestamp. Both
        // images must independently pass geometric verification.
        double best_score = -std::numeric_limits<double>::infinity();
        std::optional<RelocalizationConfirmedSeed> selected;
        for (const auto &now : hypotheses) {
            if (!valid(now)) continue;
            for (const auto &before : previous_) {
                if (now.session_id != before.session_id ||
                    now.current_frame_id == before.current_frame_id ||
                    now.timestamp <= before.timestamp) continue;
                const double dt = now.timestamp - before.timestamp;
                if (dt < 0.01 || dt > 1.0) continue;
                const double dp =
                    (now.camera_pose.p - before.camera_pose.p).norm();
                const quaternion dq =
                    before.camera_pose.q.conjugate() * now.camera_pose.q;
                const double w = std::max(0.0, std::min(1.0,
                    std::abs(dq.normalized().w())));
                const double dr = 2.0 * std::acos(w) *
                    180.0 / 3.14159265358979323846;
                // Conservative motion envelope for the 20 Hz EuRoC test;
                // a future real-time version should use IMU propagation.
                if (!std::isfinite(dp) || !std::isfinite(dr) ||
                    dp > (0.2 + 3.0 * dt) || dr > 20.0) continue;
                const double score =
                    static_cast<double>(now.inliers + before.inliers) -
                    0.1 * (now.reprojection_rmse_px +
                           before.reprojection_rmse_px);
                if (score > best_score) {
                    best_score = score;
                    selected = RelocalizationConfirmedSeed{
                        before, now, dt, dp, dr};
                }
            }
        }

        // Hold no old hypotheses if the current frame has no verified pose.
        // This demands successive successful probe frames, rather than
        // combining matches across an unconstrained visual gap.
        previous_.clear();
        for (const auto &hypothesis : hypotheses)
            if (valid(hypothesis)) previous_.push_back(hypothesis);
        if (selected) confirmed_ = selected;
        return selected;
    }

  private:
    static bool valid(const RelocalizationPnPHypothesis &h) {
        return h.session_id != 0 && h.inliers >= 8 &&
               std::isfinite(h.timestamp) &&
               std::isfinite(h.reprojection_rmse_px) &&
               h.reprojection_rmse_px <= 4.0 &&
               h.camera_pose.p.allFinite() &&
               h.camera_pose.q.coeffs().allFinite() &&
               h.camera_pose.q.squaredNorm() > 1.0e-10;
    }
    std::vector<RelocalizationPnPHypothesis> previous_;
    std::optional<RelocalizationConfirmedSeed> confirmed_;
};

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_CONSENSUS_H
