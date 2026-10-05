#ifndef XRSLAM_RECOVERY_LEARNED_MAP_H
#define XRSLAM_RECOVERY_LEARNED_MAP_H

#include <xrslam/common.h>

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace xrslam {

struct RecoveryLearnedObservationSet {
    size_t reference_frame_id = nil();
    std::vector<vector<3>> landmarks_world;
    std::vector<vector<2>> observations_pixel;
    size_t pnp_inliers = 0;
    double pnp_ratio = 0.0;

    // Independent wide-baseline recovery hypothesis in camera world pose.
    // This is a seed only; the real estimator never consumes it directly.
    bool pnp_pose_valid = false;
    quaternion pnp_q_wc = quaternion::Identity();
    vector<3> pnp_p_wc = vector<3>::Zero();
};

struct RecoveryLearnedAnchor {
    std::unordered_map<size_t, vector<3>> landmarks;
    std::unordered_map<size_t, double> landmark_angle_deg;
    std::unordered_map<size_t, double> landmark_reprojection_px;

    // Both supports are evaluated lazily only for anchors that are actually
    // used for recovery. Slot 0 is anchor-10; slot 1 is anchor+10.
    std::array<size_t, 2> support_raw{{0, 0}};
    std::array<size_t, 2> support_epipolar{{0, 0}};
    std::array<size_t, 2> support_triangulated{{0, 0}};
    bool past_support_attempted = false;
    bool future_support_attempted = false;
};

inline std::mutex &recovery_learned_map_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_map<size_t, RecoveryLearnedAnchor> &
recovery_learned_map() {
    static std::unordered_map<size_t, RecoveryLearnedAnchor> map;
    return map;
}

inline std::unordered_map<size_t, RecoveryLearnedObservationSet> &
recovery_learned_observation_map() {
    static std::unordered_map<size_t, RecoveryLearnedObservationSet> map;
    return map;
}

inline bool has_recovery_learned_anchor(size_t frame_id) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    return recovery_learned_map().count(frame_id) != 0;
}

inline void set_recovery_learned_anchor(
    size_t frame_id, const RecoveryLearnedAnchor &anchor) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    recovery_learned_map()[frame_id] = anchor;
}

inline std::optional<RecoveryLearnedAnchor>
get_recovery_learned_anchor(size_t frame_id) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    const auto &map = recovery_learned_map();
    const auto it = map.find(frame_id);
    if (it == map.end())
        return std::nullopt;
    return it->second;
}

inline void set_recovery_learned_observations(
    size_t frame_id, const RecoveryLearnedObservationSet &observations) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    recovery_learned_observation_map()[frame_id] = observations;
}

inline std::optional<RecoveryLearnedObservationSet>
get_recovery_learned_observations(size_t frame_id) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    const auto &map = recovery_learned_observation_map();
    const auto it = map.find(frame_id);
    if (it == map.end())
        return std::nullopt;
    return it->second;
}

inline void erase_recovery_learned_observations(size_t frame_id) {
    std::lock_guard<std::mutex> lock(recovery_learned_map_mutex());
    recovery_learned_observation_map().erase(frame_id);
}

} // namespace xrslam

#endif // XRSLAM_RECOVERY_LEARNED_MAP_H
