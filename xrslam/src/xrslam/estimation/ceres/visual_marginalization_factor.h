#ifndef XRSLAM_CERES_VISUAL_MARGINALIZATION_FACTOR_H
#define XRSLAM_CERES_VISUAL_MARGINALIZATION_FACTOR_H

#include <ceres/ceres.h>
#include <xrslam/estimation/ceres/reprojection_factor.h>
#include <xrslam/estimation/marginalization_factor.h>

namespace xrslam {

class CeresVisualMarginalizationFactor : public MarginalizationFactor,
                                         public ceres::CostFunction {
  public:
    static constexpr int VS_SIZE = 6;
    static constexpr int VS_Q = 0;
    static constexpr int VS_P = 3;

    explicit CeresVisualMarginalizationFactor(Map *map)
        : MarginalizationFactor(map) {
        frames.resize(map->frame_num() - 1);
        pose_linearization_point.resize(map->frame_num() - 1);
        motion_linearization_point.clear();

        for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
            frames[i] = map->get_frame(i);
            pose_linearization_point[i] = frames[i]->pose;
        }

        infovec.setZero(VS_SIZE * frames.size());
        sqrt_inv_cov.setZero(VS_SIZE * frames.size(),
                             VS_SIZE * frames.size());

        // Carry the initial world-frame gauge. Monocular scale is anchored
        // independently by the fixed inverse-depth landmark.
        if (!frames.empty()) {
            sqrt_inv_cov.block<3, 3>(VS_Q, VS_Q) =
                1.0e15 * matrix<3>::Identity();
            sqrt_inv_cov.block<3, 3>(VS_P, VS_P) =
                1.0e15 * matrix<3>::Identity();
        }

        configure_cost_function();
    }

    bool Evaluate(const double *const *parameters, double *residuals,
                  double **jacobians) const override {
        const size_t n = frames.size();

        for (size_t i = 0; i < n; ++i) {
            const_map<quaternion> q(parameters[5 * i + 0]);
            const_map<vector<3>> p(parameters[5 * i + 1]);
            map<vector<3>> rq(&residuals[VS_SIZE * i + VS_Q]);
            map<vector<3>> rp(&residuals[VS_SIZE * i + VS_P]);

            rq = logmap(pose_linearization_point[i].q.conjugate() * q);
            rp = p - pose_linearization_point[i].p;
        }

        if (jacobians) {
            for (size_t i = 0; i < n; ++i) {
                if (jacobians[5 * i + 0]) {
                    map<matrix<Eigen::Dynamic, 4, true>> dr_dq(
                        jacobians[5 * i + 0], n * VS_SIZE, 4);
                    const_map<vector<3>> rq(
                        &residuals[VS_SIZE * i + VS_Q]);
                    dr_dq.setZero();
                    dr_dq.block<3, 3>(VS_SIZE * i + VS_Q, 0) =
                        right_jacobian(rq).inverse();
                    dr_dq = sqrt_inv_cov * dr_dq;
                }

                if (jacobians[5 * i + 1]) {
                    map<matrix<Eigen::Dynamic, 3, true>> dr_dp(
                        jacobians[5 * i + 1], n * VS_SIZE, 3);
                    dr_dp.setZero();
                    dr_dp.block<3, 3>(VS_SIZE * i + VS_P, 0).setIdentity();
                    dr_dp = sqrt_inv_cov * dr_dp;
                }
                for (size_t k = 2; k < 5; ++k) {
                    if (jacobians[5 * i + k]) {
                        map<matrix<Eigen::Dynamic, 3, true>> dr_dm(
                            jacobians[5 * i + k], n * VS_SIZE, 3);
                        dr_dm.setZero();
                    }
                }
            }
        }

        map<vector<>> full_residual(residuals, n * VS_SIZE);
        full_residual = sqrt_inv_cov * full_residual + infovec;
        return true;
    }

    void marginalize(size_t index) override {
        runtime_assert(index < base_map->frame_num(),
                       "visual marginalization index out of range");

        struct LandmarkInfo {
            double mat = 0.0;
            double vec = 0.0;
            std::unordered_map<size_t, matrix<1, VS_SIZE>> h;
        };

        const size_t frame_count = base_map->frame_num();
        const size_t victim_slot = frame_count - 1;
        const size_t full_size = frame_count * VS_SIZE;

        matrix<> infomat = matrix<>::Zero(full_size, full_size);
        vector<> infovec_full = vector<>::Zero(full_size);

        std::unordered_map<Frame *, size_t> frame_indices;
        for (size_t i = 0; i < frame_count; ++i) {
            if (i < index)
                frame_indices[base_map->get_frame(i)] = i;
            else if (i > index)
                frame_indices[base_map->get_frame(i)] = i - 1;
            else
                frame_indices[base_map->get_frame(i)] = victim_slot;
        }

        // Add the previous visual marginalization prior.
        if (!frames.empty()) {
            std::vector<const double *> parameters(frames.size() * 5);
            std::vector<double *> jacobian_ptrs(frames.size() * 5);
            std::vector<matrix<Eigen::Dynamic, Eigen::Dynamic, true>>
                jacobians(frames.size() * 5);

            for (size_t i = 0; i < frames.size(); ++i) {
                parameters[5 * i + 0] = frames[i]->pose.q.coeffs().data();
                parameters[5 * i + 1] = frames[i]->pose.p.data();
                parameters[5 * i + 2] = frames[i]->motion.v.data();
                parameters[5 * i + 3] = frames[i]->motion.bg.data();
                parameters[5 * i + 4] = frames[i]->motion.ba.data();

                jacobians[5 * i + 0].resize(frames.size() * VS_SIZE, 4);
                jacobians[5 * i + 1].resize(frames.size() * VS_SIZE, 3);
                jacobian_ptrs[5 * i + 0] = jacobians[5 * i + 0].data();
                jacobian_ptrs[5 * i + 1] = jacobians[5 * i + 1].data();
                for (size_t k = 2; k < 5; ++k) {
                    jacobians[5 * i + k].resize(frames.size() * VS_SIZE, 3);
                    jacobian_ptrs[5 * i + k] =
                        jacobians[5 * i + k].data();
                }
            }

            vector<> residual(frames.size() * VS_SIZE);
            Evaluate(parameters.data(), residual.data(), jacobian_ptrs.data());

            std::vector<matrix<>> state_jacobians(frames.size());
            for (size_t i = 0; i < frames.size(); ++i) {
                matrix<> &J = state_jacobians[i];
                J.setZero(frames.size() * VS_SIZE, VS_SIZE);
                J.block(0, VS_Q, frames.size() * VS_SIZE, 3) =
                    jacobians[5 * i + 0].leftCols(3);
                J.block(0, VS_P, frames.size() * VS_SIZE, 3) =
                    jacobians[5 * i + 1];
            }

            for (size_t i = 0; i < frames.size(); ++i) {
                const size_t ii = frame_indices.at(frames[i]);
                for (size_t j = 0; j < frames.size(); ++j) {
                    const size_t jj = frame_indices.at(frames[j]);
                    infomat.block<VS_SIZE, VS_SIZE>(VS_SIZE * ii,
                                                    VS_SIZE * jj) +=
                        state_jacobians[i].transpose() * state_jacobians[j];
                }
                infovec_full.segment<VS_SIZE>(VS_SIZE * ii) +=
                    state_jacobians[i].transpose() * residual;
            }
        }

        // Add all reprojection information touched by the frame being removed.
        std::map<Track *, LandmarkInfo, compare<Track *>> landmark_info;
        Frame *frame_victim = base_map->get_frame(index);

        for (size_t k = 0; k < frame_victim->keypoint_num(); ++k) {
            Track *track = frame_victim->get_track(k);
            if (!track || !track->all_tagged(TT_VALID, TT_TRIANGULATED))
                continue;

            Frame *frame_ref = track->first_frame();
            if (!frame_ref->tag(FT_KEYFRAME))
                continue;
            if (frame_indices.count(frame_ref) == 0)
                continue;

            const size_t ref_index = frame_indices.at(frame_ref);

            for (const auto &[frame_tgt, keypoint_index] :
                 track->keypoint_map()) {
                if (frame_tgt == frame_ref)
                    continue;
                if (frame_indices.count(frame_tgt) == 0)
                    continue;

                const size_t tgt_index = frame_indices.at(frame_tgt);
                auto *cost = static_cast<CeresReprojectionErrorFactor *>(
                    frame_tgt->reprojection_error_factors[keypoint_index]
                        .get());

                std::array<const double *, 5> parameters = {
                    frame_tgt->pose.q.coeffs().data(),
                    frame_tgt->pose.p.data(),
                    frame_ref->pose.q.coeffs().data(),
                    frame_ref->pose.p.data(),
                    &(track->landmark.inv_depth)};

                vector<2> residual;
                matrix<2, 4, true> dq_tgt, dq_ref;
                matrix<2, 3, true> dp_tgt, dp_ref;
                vector<2> ddepth;
                std::array<double *, 5> jacobians = {
                    dq_tgt.data(), dp_tgt.data(), dq_ref.data(),
                    dp_ref.data(), ddepth.data()};

                if (!cost->Evaluate(parameters.data(), residual.data(),
                                    jacobians.data()))
                    continue;
                if (!residual.allFinite() || !dq_tgt.allFinite() ||
                    !dp_tgt.allFinite() || !dq_ref.allFinite() ||
                    !dp_ref.allFinite() || !ddepth.allFinite())
                    continue;

                matrix<2, VS_SIZE> J_tgt, J_ref;
                J_tgt << dq_tgt.leftCols<3>(), dp_tgt;
                J_ref << dq_ref.leftCols<3>(), dp_ref;

                const size_t tgt_offset = VS_SIZE * tgt_index;
                const size_t ref_offset = VS_SIZE * ref_index;

                infomat.block<VS_SIZE, VS_SIZE>(tgt_offset, tgt_offset) +=
                    J_tgt.transpose() * J_tgt;
                infomat.block<VS_SIZE, VS_SIZE>(ref_offset, ref_offset) +=
                    J_ref.transpose() * J_ref;
                infomat.block<VS_SIZE, VS_SIZE>(tgt_offset, ref_offset) +=
                    J_tgt.transpose() * J_ref;
                infomat.block<VS_SIZE, VS_SIZE>(ref_offset, tgt_offset) +=
                    J_ref.transpose() * J_tgt;

                infovec_full.segment<VS_SIZE>(tgt_offset) +=
                    J_tgt.transpose() * residual;
                infovec_full.segment<VS_SIZE>(ref_offset) +=
                    J_ref.transpose() * residual;

                // A fixed inverse-depth landmark is already a scale/gauge
                // constraint, so only its pose information is accumulated.
                if (track->tag(TT_FIX_INVD))
                    continue;

                LandmarkInfo &linfo = landmark_info[track];
                linfo.mat += ddepth.squaredNorm();
                linfo.vec += ddepth.dot(residual);

                if (linfo.h.count(tgt_index) == 0)
                    linfo.h[tgt_index].setZero();
                if (linfo.h.count(ref_index) == 0)
                    linfo.h[ref_index].setZero();

                linfo.h[tgt_index] += ddepth.transpose() * J_tgt;
                linfo.h[ref_index] += ddepth.transpose() * J_ref;
            }
        }

        // Schur-complement inverse-depth landmarks.
        for (const auto &[track, info] : landmark_info) {
            if (info.mat <= 1.0e-12 || !std::isfinite(info.mat))
                continue;
            const double inv = 1.0 / info.mat;

            for (const auto &[ii, h_i] : info.h) {
                for (const auto &[jj, h_j] : info.h) {
                    infomat.block<VS_SIZE, VS_SIZE>(VS_SIZE * ii,
                                                    VS_SIZE * jj) -=
                        h_i.transpose() * inv * h_j;
                }
                infovec_full.segment<VS_SIZE>(VS_SIZE * ii) -=
                    h_i.transpose() * inv * info.vec;
            }
        }

        // Schur-complement the victim pose. Use a pseudo-inverse because a
        // visual problem may retain weak gauge directions numerically.
        const size_t keep_size = VS_SIZE * victim_slot;
        const matrix<VS_SIZE, VS_SIZE> Hmm =
            infomat.block<VS_SIZE, VS_SIZE>(keep_size, keep_size);

        Eigen::SelfAdjointEigenSolver<matrix<VS_SIZE, VS_SIZE>> mm_solver(
            0.5 * (Hmm + Hmm.transpose()));
        vector<VS_SIZE> mm_inv_values =
            (mm_solver.eigenvalues().array() > 1.0e-10)
                .select(mm_solver.eigenvalues().cwiseInverse(), 0.0);
        const matrix<VS_SIZE, VS_SIZE> Hmm_inv =
            mm_solver.eigenvectors() * mm_inv_values.asDiagonal() *
            mm_solver.eigenvectors().transpose();

        matrix<> H_keep = infomat.block(0, 0, keep_size, keep_size);
        vector<> b_keep = infovec_full.head(keep_size);
        if (keep_size > 0) {
            const matrix<> Hkm =
                infomat.block(0, keep_size, keep_size, VS_SIZE);
            H_keep -= Hkm * Hmm_inv *
                      infomat.block(keep_size, 0, VS_SIZE, keep_size);
            b_keep -= Hkm * Hmm_inv *
                      infovec_full.segment<VS_SIZE>(keep_size);
        }

        H_keep = 0.5 * (H_keep + H_keep.transpose());

        Eigen::SelfAdjointEigenSolver<matrix<>> solver(H_keep);
        vector<> lambdas =
            (solver.eigenvalues().array() > 1.0e-8)
                .select(solver.eigenvalues(), 0.0);
        vector<> lambdas_inv =
            (solver.eigenvalues().array() > 1.0e-8)
                .select(solver.eigenvalues().cwiseInverse(), 0.0);

        sqrt_inv_cov = lambdas.cwiseSqrt().asDiagonal() *
                       solver.eigenvectors().transpose();
        infovec = lambdas_inv.cwiseSqrt().asDiagonal() *
                  solver.eigenvectors().transpose() * b_keep;

        frames.resize(frame_count - 1);
        pose_linearization_point.resize(frame_count - 1);
        motion_linearization_point.clear();

        for (size_t i = 0; i < frame_count; ++i) {
            if (i == index)
                continue;
            const size_t j = i > index ? i - 1 : i;
            frames[j] = base_map->get_frame(i);
            pose_linearization_point[j] = frames[j]->pose;
        }

        configure_cost_function();
    }

  private:
    void configure_cost_function() {
        set_num_residuals(static_cast<int>(frames.size() * VS_SIZE));
        mutable_parameter_block_sizes()->clear();
        for (size_t i = 0; i < frames.size(); ++i) {
            mutable_parameter_block_sizes()->push_back(4); // q
            mutable_parameter_block_sizes()->push_back(3); // p
            mutable_parameter_block_sizes()->push_back(3); // v (unused)
            mutable_parameter_block_sizes()->push_back(3); // bg (unused)
            mutable_parameter_block_sizes()->push_back(3); // ba (unused)
        }
    }
};

} // namespace xrslam

#endif // XRSLAM_CERES_VISUAL_MARGINALIZATION_FACTOR_H
