#ifndef XRSLAM_RECOVERY_POSE_CACHE_H
#define XRSLAM_RECOVERY_POSE_CACHE_H

#include <xrslam/estimation/state.h>

#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace xrslam {

// Diagnostic-only bridge from optimized sliding-window camera poses to the
// learned recovery path. Frame IDs are immutable, so this does not alter
// Frame layout or estimator state.
inline std::mutex &recovery_pose_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_map<size_t, PoseState> &recovery_pose_cache() {
    static std::unordered_map<size_t, PoseState> cache;
    return cache;
}

inline void set_recovery_camera_pose_world(size_t frame_id,
                                           const PoseState &pose) {
    std::lock_guard<std::mutex> lock(recovery_pose_cache_mutex());
    recovery_pose_cache()[frame_id] = pose;
}

inline std::optional<PoseState>
get_recovery_camera_pose_world(size_t frame_id) {
    std::lock_guard<std::mutex> lock(recovery_pose_cache_mutex());
    const auto &cache = recovery_pose_cache();
    const auto it = cache.find(frame_id);
    if (it == cache.end())
        return std::nullopt;
    return it->second;
}

} // namespace xrslam

#endif // XRSLAM_RECOVERY_POSE_CACHE_H
