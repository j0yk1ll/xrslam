#ifndef XRSLAM_PLACE_GRAPH_4DOF_SHADOW_H
#define XRSLAM_PLACE_GRAPH_4DOF_SHADOW_H

#include <xrslam/common.h>

namespace xrslam {

class KeyframeArchive;

struct PlaceGraph4DoFEdge {
    size_t event_id = 0;
    size_t reference_frame_id = 0;
    size_t current_frame_id = 0;
    vector<3> relative_translation = vector<3>::Zero();
    double relative_yaw_deg = 0.0;
    double residual_gain = 1.0;
};

struct PlaceGraph4DoFCorrection {
    size_t frame_id = 0;
    double timestamp = 0.0;
    double yaw_deg = 0.0;
    vector<3> translation = vector<3>::Zero();
};

struct PlaceGraph4DoFSolveResult {
    bool usable = false;
    size_t node_count = 0;
    size_t odometry_edge_count = 0;
    size_t loop_edge_count = 0;
    size_t iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::vector<PlaceGraph4DoFCorrection> corrections;

    const PlaceGraph4DoFCorrection *find_correction(
        size_t frame_id) const {
        const auto it = std::find_if(
            corrections.begin(), corrections.end(),
            [frame_id](const PlaceGraph4DoFCorrection &correction) {
                return correction.frame_id == frame_id;
            });
        return it == corrections.end() ? nullptr : &*it;
    }
};

PlaceGraph4DoFSolveResult solve_place_graph_4dof_shadow(
    const KeyframeArchive &archive,
    const std::vector<PlaceGraph4DoFEdge> &loop_edges);

} // namespace xrslam

#endif // XRSLAM_PLACE_GRAPH_4DOF_SHADOW_H
