#ifndef XRSLAM_DENSE_SEMANTIC_DEPTH_FILTER_H
#define XRSLAM_DENSE_SEMANTIC_DEPTH_FILTER_H

#include <xrslam/version.h>
#include <xrslam/semantic_mapping.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace xrslam {

// Opt-in output-only quality settings. Baseline 0123a is bit-for-bit
// unchanged unless enabled=true. All distances are physical meters.
struct DenseSemanticQualityOptions {
    bool enabled = false;
    double depth_edge_abs_m = 0.08;
    double depth_edge_relative = 0.03;
    std::uint32_t min_views = 2;
    std::uint32_t min_neighbors = 2; // 26-connected occupied voxel cells
    bool reject_missing_neighbors = true;

    void validate() const {
        if (!std::isfinite(depth_edge_abs_m) ||
            depth_edge_abs_m < 0.001 || depth_edge_abs_m > 0.50 ||
            !std::isfinite(depth_edge_relative) ||
            depth_edge_relative < 0.0 || depth_edge_relative > 0.30 ||
            min_views < 1 || min_views > 64 || min_neighbors > 26)
            throw std::invalid_argument("invalid dense semantic quality options");
    }
};

// A conservative 4-connected depth-discontinuity test applied to a *registered*
// metric-Z depth snapshot. Samples at image edges or adjacent to depth holes
// are excluded by default (fail closed). It does not touch the depth that VIO
// consumes and it does not change camera poses or landmark geometry.
inline bool dense_semantic_depth_is_consistent(
    const SemanticMetricDepthBuffer &depth, std::uint32_t x,
    std::uint32_t y, const DenseSemanticQualityOptions &options) {
    if (!depth.valid() || x >= depth.width || y >= depth.height)
        return false;
    const auto offset = static_cast<std::size_t>(y) * depth.width + x;
    const double z = depth.meters[offset];
    if (!std::isfinite(z) || z < 0.25 || z > 5.0)
        return false;
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        const auto nx = static_cast<std::int64_t>(x) + dx[k];
        const auto ny = static_cast<std::int64_t>(y) + dy[k];
        if (nx < 0 || ny < 0 || nx >= depth.width || ny >= depth.height) {
            if (options.reject_missing_neighbors) return false;
            continue;
        }
        const double neighbor = depth.meters[
            static_cast<std::size_t>(ny) * depth.width +
            static_cast<std::size_t>(nx)];
        if (!std::isfinite(neighbor) || neighbor < 0.25 || neighbor > 5.0) {
            if (options.reject_missing_neighbors) return false;
            continue;
        }
        const double tolerance = std::max(
            options.depth_edge_abs_m,
            options.depth_edge_relative * std::max(z, neighbor));
        if (std::abs(z - neighbor) > tolerance) return false;
    }
    return true;
}

} // namespace xrslam
#endif // XRSLAM_DENSE_SEMANTIC_DEPTH_FILTER_H
