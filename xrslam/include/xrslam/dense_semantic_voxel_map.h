#ifndef XRSLAM_DENSE_SEMANTIC_VOXEL_MAP_H
#define XRSLAM_DENSE_SEMANTIC_VOXEL_MAP_H

#include <xrslam/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace xrslam {

// A *surface-sample voxel cloud*, not a TSDF, occupancy grid or mesh.
// Coordinates remain in their originating VIO session's local world frame.
struct DenseSemanticVoxelPoint {
    std::uint64_t session_id = 0;
    std::int32_t ix = 0, iy = 0, iz = 0;
    std::array<double, 3> xyz{};
    std::array<std::uint8_t, 3> rgb{};
    std::uint8_t semantic_class = 0;
    double consensus = 0.0; // weighted class agreement, not posterior probability
    std::uint32_t observation_count = 0; // one vote per frame and voxel
    bool stable = false;
};

struct DenseSemanticSample {
    std::uint64_t session_id = 0;
    std::uint64_t frame_id = 0;
    std::array<double, 3> xyz{};
    std::array<std::uint8_t, 3> rgb{};
    std::uint8_t label = 0;
    float confidence = 0.0f;
};

// Caller must synchronize observe()/snapshot(). The owning PIDNet worker
// holds its mutex for both. Geometry is output-only; no estimator pointers.
class DenseSemanticVoxelMap {
  public:
    DenseSemanticVoxelMap(std::uint32_t classes, double voxel_size_m = 0.03,
                          std::size_t max_voxels = 250000)
        : classes_(classes), voxel_size_m_(voxel_size_m),
          max_voxels_(max_voxels) {
        if (classes == 0 || classes > 255 || !std::isfinite(voxel_size_m) ||
            voxel_size_m < 0.01 || voxel_size_m > 0.20 ||
            max_voxels == 0 || max_voxels > 1000000)
            throw std::invalid_argument("invalid dense semantic voxel options");
    }

    bool observe(const DenseSemanticSample &sample) {
        if (sample.session_id == 0 || sample.label >= classes_ ||
            !std::isfinite(sample.confidence) || sample.confidence < 0.50f ||
            sample.confidence > 1.0f)
            return false;
        for (double c : sample.xyz)
            if (!std::isfinite(c)) return false;
        std::array<std::int32_t, 3> cell{};
        for (int i = 0; i < 3; ++i) {
            const double v = std::floor(sample.xyz[i] / voxel_size_m_);
            if (!std::isfinite(v) ||
                v < std::numeric_limits<std::int32_t>::min() ||
                v > std::numeric_limits<std::int32_t>::max()) return false;
            cell[i] = static_cast<std::int32_t>(v);
        }
        const Key key{sample.session_id, cell[0], cell[1], cell[2]};
        auto it = voxels_.find(key);
        if (it == voxels_.end()) {
            if (voxels_.size() >= max_voxels_) {
                ++capacity_rejections_;
                return false;
            }
            Evidence e;
            e.class_weights.resize(classes_, 0.0);
            e.class_views.resize(classes_, 0);
            it = voxels_.emplace(key, std::move(e)).first;
        }
        Evidence &e = it->second;
        // Jobs execute on a FIFO inference worker; the first accepted sample
        // for this (voxel, keyframe) wins. No pixel-density amplification.
        if (e.has_last_frame && e.last_frame_id == sample.frame_id)
            return false;
        e.has_last_frame = true;
        e.last_frame_id = sample.frame_id;
        const double weight = sample.confidence;
        e.class_weights[sample.label] += weight;
        ++e.class_views[sample.label];
        e.total_weight += weight;
        ++e.views;
        for (int i = 0; i < 3; ++i) {
            e.xyz_weighted[i] += weight * sample.xyz[i];
            e.rgb_weighted[i] += weight * sample.rgb[i];
        }
        return true;
    }

    std::vector<DenseSemanticVoxelPoint> snapshot(bool stable_only = false) const {
        // Preserve 0123a output exactly when quality filtering is disabled.
        return snapshot_filtered(stable_only, 1, 0);
    }

    // Read-only spatial and temporal support filtering: a cell must be seen
    // in at least min_views different accepted keyframes and have at least
    // min_neighbors other sufficiently observed cells in its 26-neighborhood.
    // There is no cross-session support, and no destructive voxel removal.
    std::vector<DenseSemanticVoxelPoint> snapshot_filtered(
        bool stable_only, std::uint32_t min_views,
        std::uint32_t min_neighbors) const {
        if (min_views < 1 || min_views > 64 || min_neighbors > 26)
            throw std::invalid_argument("invalid dense voxel snapshot filters");
        std::vector<DenseSemanticVoxelPoint> result;
        result.reserve(voxels_.size());
        for (const auto &[key, e] : voxels_) {
            if (e.views < min_views) continue;
            if (min_neighbors > 0) {
                std::uint32_t supporters = 0;
                const auto session = std::get<0>(key);
                const std::int64_t x = std::get<1>(key);
                const std::int64_t y = std::get<2>(key);
                const std::int64_t z = std::get<3>(key);
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            const std::int64_t nx = x + dx;
                            const std::int64_t ny = y + dy;
                            const std::int64_t nz = z + dz;
                            if (nx < std::numeric_limits<std::int32_t>::min() ||
                                nx > std::numeric_limits<std::int32_t>::max() ||
                                ny < std::numeric_limits<std::int32_t>::min() ||
                                ny > std::numeric_limits<std::int32_t>::max() ||
                                nz < std::numeric_limits<std::int32_t>::min() ||
                                nz > std::numeric_limits<std::int32_t>::max())
                                continue;
                            const auto it = voxels_.find(Key{
                                session, static_cast<std::int32_t>(nx),
                                static_cast<std::int32_t>(ny),
                                static_cast<std::int32_t>(nz)});
                            if (it != voxels_.end() &&
                                it->second.views >= min_views)
                                ++supporters;
                        }
                if (supporters < min_neighbors) continue;
            }
            const auto winner = static_cast<std::size_t>(std::distance(
                e.class_weights.begin(), std::max_element(
                    e.class_weights.begin(), e.class_weights.end())));
            const double consensus = e.class_weights[winner] / e.total_weight;
            const bool stable = e.class_views[winner] >= 2 && consensus >= 0.65;
            if (stable_only && !stable) continue;
            DenseSemanticVoxelPoint p;
            p.session_id = std::get<0>(key);
            p.ix = std::get<1>(key);
            p.iy = std::get<2>(key);
            p.iz = std::get<3>(key);
            p.semantic_class = static_cast<std::uint8_t>(winner);
            p.consensus = consensus;
            p.observation_count = e.views;
            p.stable = stable;
            for (int i = 0; i < 3; ++i) {
                p.xyz[i] = e.xyz_weighted[i] / e.total_weight;
                p.rgb[i] = static_cast<std::uint8_t>(std::clamp(
                    std::lround(e.rgb_weighted[i] / e.total_weight), 0l, 255l));
            }
            result.push_back(p);
        }
        return result;
    }

    std::size_t size() const { return voxels_.size(); }
    std::size_t capacity_rejections() const { return capacity_rejections_; }
    double voxel_size_m() const { return voxel_size_m_; }

    // q_xyzw rotates optical camera coordinates to session-local world.
    static std::array<double, 3> camera_to_session_world(
        const std::array<double, 4> &q_xyzw,
        const std::array<double, 3> &translation,
        const std::array<double, 3> &point_camera) {
        const double x = q_xyzw[0], y = q_xyzw[1], z = q_xyzw[2], w = q_xyzw[3];
        const double n = x*x + y*y + z*z + w*w;
        if (!std::isfinite(n) || n < 1.e-12)
            throw std::invalid_argument("non-finite or zero camera quaternion");
        const std::array<double, 3> t{
            2.0 * (y * point_camera[2] - z * point_camera[1]) * (1.0/n),
            2.0 * (z * point_camera[0] - x * point_camera[2]) * (1.0/n),
            2.0 * (x * point_camera[1] - y * point_camera[0]) * (1.0/n)
        };
        // R(q) v = v + w*(2/n)*(q_vec x v) + (2/n)*(q_vec x (q_vec x v))
        const std::array<double, 3> cross{
            y * t[2] - z * t[1],
            z * t[0] - x * t[2],
            x * t[1] - y * t[0]
        };
        return {
            translation[0] + point_camera[0] + w*t[0] + cross[0],
            translation[1] + point_camera[1] + w*t[1] + cross[1],
            translation[2] + point_camera[2] + w*t[2] + cross[2]
        };
    }

  private:
    using Key = std::tuple<std::uint64_t, std::int32_t, std::int32_t, std::int32_t>;
    struct Evidence {
        std::vector<double> class_weights;
        std::vector<std::uint32_t> class_views;
        std::array<double, 3> xyz_weighted{};
        std::array<double, 3> rgb_weighted{};
        double total_weight = 0;
        std::uint32_t views = 0;
        std::uint64_t last_frame_id = 0;
        bool has_last_frame = false;
    };
    std::uint32_t classes_;
    double voxel_size_m_;
    std::size_t max_voxels_;
    std::size_t capacity_rejections_ = 0;
    std::map<Key, Evidence> voxels_;
};

} // namespace xrslam
#endif // XRSLAM_DENSE_SEMANTIC_VOXEL_MAP_H
