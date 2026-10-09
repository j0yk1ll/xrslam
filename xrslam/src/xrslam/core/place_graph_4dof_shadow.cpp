#include <ceres/ceres.h>

#include <xrslam/core/keyframe_archive.h>
#include <xrslam/core/place_graph_4dof_shadow.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace xrslam {

namespace {

double wrap_degrees(double angle_deg) {
    return std::remainder(angle_deg, 360.0);
}

void pose_yaw_pitch_roll_deg(
    const PoseState &pose,
    double &yaw_deg,
    double &pitch_deg,
    double &roll_deg) {
    const matrix<3> R = pose.q.matrix();
    yaw_deg = std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI;
    pitch_deg = std::asin(std::max(-1.0, std::min(1.0, -R(2, 0)))) *
                180.0 / M_PI;
    roll_deg = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;
}

template <typename T>
T wrapped_angle_deg(const T &angle_deg) {
    using std::atan2;
    using std::cos;
    using std::sin;
    const T angle_rad = angle_deg * T(M_PI / 180.0);
    return atan2(sin(angle_rad), cos(angle_rad)) * T(180.0 / M_PI);
}

template <typename T>
void yaw_pitch_roll_matrix(
    const T &yaw_deg,
    double pitch_deg,
    double roll_deg,
    T R[9]) {
    using std::cos;
    using std::sin;

    const T yaw = yaw_deg * T(M_PI / 180.0);
    const T pitch = T(pitch_deg * M_PI / 180.0);
    const T roll = T(roll_deg * M_PI / 180.0);

    const T cy = cos(yaw);
    const T sy = sin(yaw);
    const T cp = cos(pitch);
    const T sp = sin(pitch);
    const T cr = cos(roll);
    const T sr = sin(roll);

    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

struct FourDoFCost {
    FourDoFCost(
        const vector<3> &relative_translation,
        double relative_yaw_deg,
        double pitch_i_deg,
        double roll_i_deg,
        double yaw_residual_scale,
        double residual_gain)
        : relative_translation(relative_translation),
          relative_yaw_deg(relative_yaw_deg),
          pitch_i_deg(pitch_i_deg),
          roll_i_deg(roll_i_deg),
          yaw_residual_scale(yaw_residual_scale),
          residual_gain(residual_gain) {}

    template <typename T>
    bool operator()(
        const T *const yaw_i_deg,
        const T *const p_i,
        const T *const yaw_j_deg,
        const T *const p_j,
        T *residuals) const {
        T R_w_i[9];
        yaw_pitch_roll_matrix(
            yaw_i_deg[0], pitch_i_deg, roll_i_deg, R_w_i);

        const T dx = p_j[0] - p_i[0];
        const T dy = p_j[1] - p_i[1];
        const T dz = p_j[2] - p_i[2];

        // R_i^T * (p_j - p_i).
        residuals[0] =
            (R_w_i[0] * dx + R_w_i[3] * dy + R_w_i[6] * dz -
             T(relative_translation.x())) *
            T(residual_gain);
        residuals[1] =
            (R_w_i[1] * dx + R_w_i[4] * dy + R_w_i[7] * dz -
             T(relative_translation.y())) *
            T(residual_gain);
        residuals[2] =
            (R_w_i[2] * dx + R_w_i[5] * dy + R_w_i[8] * dz -
             T(relative_translation.z())) *
            T(residual_gain);
        residuals[3] =
            wrapped_angle_deg(
                yaw_j_deg[0] - yaw_i_deg[0] - T(relative_yaw_deg)) *
            T(yaw_residual_scale) * T(residual_gain);
        return true;
    }

    vector<3> relative_translation;
    double relative_yaw_deg;
    double pitch_i_deg;
    double roll_i_deg;
    double yaw_residual_scale;
    double residual_gain;
};

struct OptimizationNode {
    size_t frame_id = 0;
    double timestamp = 0.0;
    PoseState initial_pose;
    double initial_yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
    double yaw_deg = 0.0;
    vector<3> position = vector<3>::Zero();
};

} // namespace

PlaceGraph4DoFSolveResult solve_place_graph_4dof_shadow(
    const KeyframeArchive &archive,
    const std::vector<PlaceGraph4DoFEdge> &loop_edges) {
    PlaceGraph4DoFSolveResult result;
    if (loop_edges.empty() || archive.size() == 0)
        return result;

    const std::vector<size_t> &order = archive.insertion_order();
    std::unordered_map<size_t, size_t> archive_index;
    archive_index.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i)
        archive_index.emplace(order[i], i);

    size_t first_index = order.size();
    size_t last_index = 0;
    std::vector<PlaceGraph4DoFEdge> valid_loop_edges;
    valid_loop_edges.reserve(loop_edges.size());
    for (const PlaceGraph4DoFEdge &edge : loop_edges) {
        const auto reference_it =
            archive_index.find(edge.reference_frame_id);
        const auto current_it =
            archive_index.find(edge.current_frame_id);
        if (reference_it == archive_index.end() ||
            current_it == archive_index.end()) {
            continue;
        }
        if (!edge.relative_translation.allFinite() ||
            !std::isfinite(edge.relative_yaw_deg) ||
            !std::isfinite(edge.residual_gain) ||
            edge.residual_gain <= 0.0) {
            continue;
        }
        first_index = std::min(first_index, reference_it->second);
        last_index = std::max(last_index, current_it->second);
        valid_loop_edges.emplace_back(edge);
    }

    if (valid_loop_edges.empty() || first_index > last_index)
        return result;

    std::vector<OptimizationNode> nodes;
    nodes.reserve(last_index - first_index + 1);
    std::unordered_map<size_t, size_t> local_index;
    local_index.reserve(last_index - first_index + 1);

    for (size_t archive_i = first_index;
         archive_i <= last_index;
         ++archive_i) {
        const ArchivedKeyframe *archived = archive.get(order[archive_i]);
        if (!archived || !archived->body_pose.p.allFinite() ||
            !archived->body_pose.q.coeffs().allFinite()) {
            return result;
        }

        OptimizationNode node;
        node.frame_id = archived->frame_id;
        node.timestamp = archived->timestamp;
        node.initial_pose = archived->body_pose;
        pose_yaw_pitch_roll_deg(
            node.initial_pose,
            node.initial_yaw_deg,
            node.pitch_deg,
            node.roll_deg);
        node.yaw_deg = node.initial_yaw_deg;
        node.position = node.initial_pose.p;

        local_index.emplace(node.frame_id, nodes.size());
        nodes.emplace_back(node);
    }

    if (nodes.size() < 2)
        return result;

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership =
        ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    for (OptimizationNode &node : nodes) {
        problem.AddParameterBlock(&node.yaw_deg, 1);
        problem.AddParameterBlock(node.position.data(), 3);
    }
    problem.SetParameterBlockConstant(&nodes.front().yaw_deg);
    problem.SetParameterBlockConstant(nodes.front().position.data());

    size_t odometry_edge_count = 0;
    for (size_t j = 0; j < nodes.size(); ++j) {
        for (size_t gap = 1; gap <= 4 && gap <= j; ++gap) {
            const size_t i = j - gap;
            const OptimizationNode &node_i = nodes[i];
            const OptimizationNode &node_j = nodes[j];

            const vector<3> relative_translation =
                node_i.initial_pose.q.conjugate() *
                (node_j.initial_pose.p - node_i.initial_pose.p);
            const double relative_yaw_deg = wrap_degrees(
                node_j.initial_yaw_deg - node_i.initial_yaw_deg);

            auto *cost =
                new ceres::AutoDiffCostFunction<FourDoFCost, 4, 1, 3, 1, 3>(
                    new FourDoFCost(
                        relative_translation,
                        relative_yaw_deg,
                        node_i.pitch_deg,
                        node_i.roll_deg,
                        1.0,
                        1.0));
            problem.AddResidualBlock(
                cost, nullptr,
                &nodes[i].yaw_deg, nodes[i].position.data(),
                &nodes[j].yaw_deg, nodes[j].position.data());
            ++odometry_edge_count;
        }
    }

    ceres::CauchyLoss loop_loss(0.1);
    size_t added_loop_edge_count = 0;
    for (const PlaceGraph4DoFEdge &edge : valid_loop_edges) {
        const auto reference_it = local_index.find(edge.reference_frame_id);
        const auto current_it = local_index.find(edge.current_frame_id);
        if (reference_it == local_index.end() ||
            current_it == local_index.end()) {
            continue;
        }

        const size_t i = reference_it->second;
        const size_t j = current_it->second;
        auto *cost =
            new ceres::AutoDiffCostFunction<FourDoFCost, 4, 1, 3, 1, 3>(
                new FourDoFCost(
                    edge.relative_translation,
                    edge.relative_yaw_deg,
                    nodes[i].pitch_deg,
                    nodes[i].roll_deg,
                    0.1,
                    edge.residual_gain));
        problem.AddResidualBlock(
            cost, &loop_loss,
            &nodes[i].yaw_deg, nodes[i].position.data(),
            &nodes[j].yaw_deg, nodes[j].position.data());
        ++added_loop_edge_count;
    }

    if (added_loop_edge_count == 0)
        return result;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.node_count = nodes.size();
    result.odometry_edge_count = odometry_edge_count;
    result.loop_edge_count = added_loop_edge_count;
    result.iterations = static_cast<size_t>(
        summary.num_successful_steps + summary.num_unsuccessful_steps);
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.usable = summary.IsSolutionUsable();

    result.corrections.reserve(nodes.size());
    for (const OptimizationNode &node : nodes) {
        const double correction_yaw_deg = wrap_degrees(
            node.yaw_deg - node.initial_yaw_deg);
        quaternion correction_q(
            Eigen::AngleAxisd(
                correction_yaw_deg * M_PI / 180.0,
                Eigen::Vector3d::UnitZ()));
        correction_q.normalize();

        PlaceGraph4DoFCorrection correction;
        correction.frame_id = node.frame_id;
        correction.timestamp = node.timestamp;
        correction.yaw_deg = correction_yaw_deg;
        correction.translation =
            node.position - correction_q * node.initial_pose.p;
        if (!std::isfinite(correction.yaw_deg) ||
            !correction.translation.allFinite()) {
            result.usable = false;
        }
        result.corrections.emplace_back(correction);
    }

    return result;
}

} // namespace xrslam
