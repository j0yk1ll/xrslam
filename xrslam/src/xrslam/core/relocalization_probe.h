#ifndef XRSLAM_RELOCALIZATION_PROBE_H
#define XRSLAM_RELOCALIZATION_PROBE_H

#include <xrslam/core/persistent_relocalization_map.h>
#include <xrslam/core/relocalization_consensus.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/map/frame.h>
#include <xrslam/place_recognition.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace xrslam {

// pnp.h defines non-inline functions and must not be included from another
// translation unit. Reuse the same function already linked from SWT instead.
matrix<4> find_pnp_matrix(
    const std::vector<vector<3>> &Xs,
    const std::vector<vector<2>> &xs,
    std::vector<char> &inlier_mask,
    double threshold, double confidence, size_t max_iteration, int seed);

// Recovery-time visual verification only. This function NEVER mutates VIO,
// active tracks, historical geometry, or the frozen 4-DoF graph.
// The recovered camera pose, when valid, belongs to the reported OLD session.
inline std::vector<RelocalizationPnPHypothesis> probe_relocalization_reference(
    Frame *frame,
    const PersistentRelocalizationMap &history,
    PlaceDescriptorExtractor *global_extractor,
    PlaceDatabase *database,
    LocalDescriptorExtractor *local_extractor) {
    if (!frame || !frame->image || !global_extractor || !database ||
        !local_extractor ||
        local_extractor->type() != LocalDescriptorType::BINARY_U8 ||
        !frame->image->has_place_recognition_source()) {
        std::fprintf(stderr,
            "[RelocalizationLostProbe] event=skip current=%zu "
            "reason=missing_image_or_extractor state_mutation=0\n",
            frame ? frame->id() : size_t(0));
        return {};
    }

    try {
        const PlaceDescriptor descriptor =
            global_extractor->extract(*frame->image);
        if (descriptor.empty() ||
            descriptor.dimension() != database->dimension()) {
            std::fprintf(stderr,
                "[RelocalizationLostProbe] event=skip current=%zu "
                "reason=invalid_global_descriptor state_mutation=0\n",
                frame->id());
            return {};
        }
        const std::vector<PlaceCandidate> results =
            database->search(descriptor, 20);

        std::vector<vector<2>> current_pixels;
        std::vector<size_t> current_pixel_to_keypoint;
        current_pixels.reserve(frame->keypoint_num());
        current_pixel_to_keypoint.reserve(frame->keypoint_num());
        for (size_t k = 0; k < frame->keypoint_num(); ++k) {
            const vector<2> pixel =
                apply_k(frame->get_keypoint(k), frame->K);
            if (pixel.allFinite()) {
                current_pixels.push_back(pixel);
                current_pixel_to_keypoint.push_back(k);
            }
        }
        if (current_pixels.size() < 6) {
            std::fprintf(stderr,
                "[RelocalizationLostProbe] event=skip current=%zu "
                "reason=few_current_keypoints count=%zu state_mutation=0\n",
                frame->id(), current_pixels.size());
            return {};
        }
        const LocalDescriptorSet current_descriptors =
            local_extractor->extract(*frame->image, current_pixels);
        if (!current_descriptors.valid() ||
            current_descriptors.type != LocalDescriptorType::BINARY_U8 ||
            current_descriptors.dimension != local_extractor->dimension() ||
            current_descriptors.size() == 0) {
            std::fprintf(stderr,
                "[RelocalizationLostProbe] event=skip current=%zu "
                "reason=invalid_local_descriptors state_mutation=0\n",
                frame->id());
            return {};
        }

        size_t retained_candidates = 0;
        size_t verified_candidates = 0;
        std::vector<RelocalizationPnPHypothesis> verified_hypotheses;
        const matrix<3> K_inv = frame->K.inverse();
        if (!K_inv.allFinite()) return {};

        for (size_t rank = 0; rank < results.size(); ++rank) {
            const auto candidate = results[rank];
            const RelocalizationReferenceMatch reference =
                history.find(candidate.key);
            if (!reference.geometry || !reference.place) continue;
            ++retained_candidates;
            const ArchivedKeyframe &old = *reference.geometry;
            if (old.local_descriptors.type !=
                    LocalDescriptorType::BINARY_U8 ||
                old.local_descriptors.dimension !=
                    current_descriptors.dimension) continue;

            const LocalDescriptorMatchResult matched =
                match_binary_descriptors_mutual_nn(
                    old.local_descriptors, current_descriptors);
            std::vector<vector<3>> world_points;
            std::vector<vector<2>> normalized_points;
            std::vector<vector<2>> pixels;
            std::vector<RelocalizationInlierLandmark> associated;
            for (const LocalDescriptorMatch &m : matched.mutual_matches) {
                // Use the unchanged frozen loop verification's Hamming bound.
                if (m.distance > 60 ||
                    m.reference_descriptor_index >=
                        old.local_descriptors.source_indices.size() ||
                    m.current_descriptor_index >=
                        current_descriptors.source_indices.size()) continue;
                const size_t obs = old.local_descriptors.source_indices[
                    m.reference_descriptor_index];
                const size_t point = current_descriptors.source_indices[
                    m.current_descriptor_index];
                if (obs >= old.observations.size() ||
                    point >= current_pixels.size()) continue;
                const vector<3> &world = old.observations[obs].landmark_world;
                const vector<2> &pixel = current_pixels[point];
                const vector<3> normalized_h =
                    K_inv * vector<3>{pixel.x(), pixel.y(), 1.0};
                if (!world.allFinite() || !normalized_h.allFinite() ||
                    std::abs(normalized_h.z()) < 1.e-12) continue;
                world_points.push_back(world);
                normalized_points.push_back(normalized_h.hnormalized());
                pixels.push_back(pixel);
                associated.push_back(RelocalizationInlierLandmark{
                    current_pixel_to_keypoint[point],
                    old.observations[obs].track_id, world});
            }
            if (world_points.size() < 6) continue;
            std::vector<char> mask;
            const matrix<4> T_cw = find_pnp_matrix(
                world_points, normalized_points, mask,
                1.0 / frame->K(0, 0), 0.999, 1000, 0);
            if (!T_cw.allFinite() || mask.size() != world_points.size())
                continue;
            const matrix<3> R_cw = T_cw.block<3,3>(0,0);
            const vector<3> t_cw = T_cw.block<3,1>(0,3);
            if (R_cw.determinant() < 0.9 || R_cw.determinant() > 1.1)
                continue;
            size_t inliers = 0;
            size_t positive_depth = 0;
            double squared_error = 0;
            for (size_t i = 0; i < mask.size(); ++i) {
                if (!mask[i]) continue;
                ++inliers;
                const vector<3> p = R_cw * world_points[i] + t_cw;
                if (!p.allFinite() || p.z() <= 1.e-6) continue;
                ++positive_depth;
                const vector<2> projected = apply_k(p, frame->K);
                squared_error += (projected - pixels[i]).squaredNorm();
            }
            const double ratio = double(inliers) / world_points.size();
            const double rmse = positive_depth ?
                std::sqrt(squared_error / positive_depth) : 1.e30;
            // The 8-inlier/0.50 ratio gates match the frozen verifier;
            // cheirality and pixel RMSE are additional safety checks.
            const bool verified = inliers >= 8 && ratio >= 0.50 &&
                positive_depth == inliers && rmse <= 4.0;
            const matrix<3> R_wc = R_cw.transpose();
            const vector<3> p_wc = -R_wc * t_cw;
            const quaternion q_wc(R_wc);
            if (verified && p_wc.allFinite() &&
                q_wc.coeffs().allFinite()) {
                ++verified_candidates;
                RelocalizationPnPHypothesis hypothesis;
                hypothesis.session_id = reference.session_id;
                hypothesis.reference_frame_id = old.frame_id;
                hypothesis.current_frame_id = frame->id();
                hypothesis.timestamp = frame->image->t;
                hypothesis.inliers = inliers;
                hypothesis.reprojection_rmse_px = rmse;
                hypothesis.camera_pose.p = p_wc;
                hypothesis.camera_pose.q = q_wc.normalized();
                for (size_t i = 0; i < mask.size(); ++i)
                    if (mask[i]) hypothesis.landmarks.push_back(associated[i]);
                verified_hypotheses.push_back(std::move(hypothesis));
            }
            std::fprintf(stderr,
                "[RelocalizationLostProbe] event=candidate "
                "current=%zu t=%.9f session=%zu reference=%zu "
                "rank=%zu correspondences=%zu inliers=%zu "
                "inlier_ratio=%.6f rmse_px=%.6f verified=%d "
                "p_wc=%.9f,%.9f,%.9f "
                "q_wc=%.9f,%.9f,%.9f,%.9f "
                "state_mutation=0\n",
                frame->id(), frame->image->t, reference.session_id,
                old.frame_id, rank, world_points.size(), inliers,
                ratio, rmse, verified ? 1 : 0,
                p_wc.x(), p_wc.y(), p_wc.z(),
                q_wc.x(), q_wc.y(), q_wc.z(), q_wc.w());
        }
        std::fprintf(stderr,
            "[RelocalizationLostProbe] event=summary current=%zu "
            "t=%.9f database_candidates=%zu retained_candidates=%zu "
            "verified=%zu state_mutation=0\n",
            frame->id(), frame->image->t, results.size(),
            retained_candidates, verified_candidates);
        return verified_hypotheses;
    } catch (const std::exception &e) {
        std::fprintf(stderr,
            "[RelocalizationLostProbe] event=skip current=%zu "
            "reason=exception error=%s state_mutation=0\n",
            frame->id(), e.what());
        return {};
    }
}

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_PROBE_H
