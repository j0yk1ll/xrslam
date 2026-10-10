#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <optional>
#include <xrslam/core/relocalization_probe.h>
#include <xrslam/core/relocalization_bootstrap.h>
#include <xrslam/core/relocalization_tracking_health.h>
#include <xrslam/core/relocalization_auto_loss.h>
#include <xrslam/common.h>
#include <xrslam/core/detail.h>
#include <xrslam/core/feature_tracker.h>
#include <xrslam/core/frontend_worker.h>
#include <xrslam/core/initializer.h>
#include <xrslam/core/sliding_window_tracker.h>
#include <xrslam/geometry/stereo.h>
#include <xrslam/localizer/localizer.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>
#include <xrslam/map/track.h>

namespace xrslam {

FrontendWorker::FrontendWorker(XRSLAM::Detail *detail,
                               std::shared_ptr<Config> config)
    : detail(detail), config(config) {
    initializer = std::make_unique<Initializer>(config);

    latest_state = {{}, nil(), {}, {}};
}

FrontendWorker::~FrontendWorker() = default;

bool FrontendWorker::empty() const { return pending_frame_ids.empty(); }

void FrontendWorker::work(std::unique_lock<std::mutex> &l) {
    if (initializer) {
        size_t pending_frame_id = pending_frame_ids.front();
        pending_frame_ids.clear();
        l.unlock();
        std::unique_ptr<SlidingWindowTracker> recovery_candidate;
        std::optional<RelocalizationConfirmedSeed> attempted_seed;
        size_t recovery_anchor_tracks = 0;
        size_t recovery_transfer_tracks = 0;
        synchronized(detail->feature_tracker->map) {
            // Only probe a retained OLD session, and only with the opt-in
            // diagnostic flag. No recovery pose reaches the live estimator.
            const char *probe = std::getenv("XRSLAM_RELOCALIZATION_PROBE");
            if (probe && std::string(probe) == "1" &&
                persistent_relocalization_map_.session_count() != 0 &&
                !relocalization_consensus_.confirmed() &&
                pending_frame_id % 5 == 0) {
                const size_t index =
                    detail->feature_tracker->map->frame_index_by_id(
                        pending_frame_id);
                if (index != nil()) {
                    const auto hypotheses = probe_relocalization_reference(
                        detail->feature_tracker->map->get_frame(index),
                        persistent_relocalization_map_,
                        detail->place_descriptor_extractor(),
                        detail->place_database(),
                        detail->local_descriptor_extractor());
                    const auto confirmed =
                        relocalization_consensus_.observe(hypotheses);
                    if (confirmed) {
                        const auto &seed = confirmed->second;
                        std::fprintf(stderr,
                            "[RelocalizationConsensus] event=confirmed "
                            "session=%zu first_current=%zu current=%zu "
                            "first_reference=%zu reference=%zu "
                            "t=%.9f dt_s=%.6f "
                            "position_disagreement_m=%.6f "
                            "rotation_disagreement_deg=%.6f "
                            "inliers=%zu rmse_px=%.6f "
                            "p_wc=%.9f,%.9f,%.9f "
                            "q_wc=%.9f,%.9f,%.9f,%.9f "
                            "estimator_mutation=0\n",
                            seed.session_id,
                            confirmed->first.current_frame_id,
                            seed.current_frame_id,
                            confirmed->first.reference_frame_id,
                            seed.reference_frame_id, seed.timestamp,
                            confirmed->dt_s,
                            confirmed->position_difference_m,
                            confirmed->orientation_difference_deg,
                            seed.inliers, seed.reprojection_rmse_px,
                            seed.camera_pose.p.x(),
                            seed.camera_pose.p.y(),
                            seed.camera_pose.p.z(),
                            seed.camera_pose.q.x(),
                            seed.camera_pose.q.y(),
                            seed.camera_pose.q.z(),
                            seed.camera_pose.q.w());
                    }
                }
            }
            const char *handoff = std::getenv("XRSLAM_RELOCALIZATION_HANDOFF");
            if (handoff && std::string(handoff) == "1" &&
                relocalization_consensus_.confirmed()) {
                const auto &seed = *relocalization_consensus_.confirmed();
                // Never retry a previously attempted seed across initializer
                // frames; there is exactly one controlled attempt per loss.
                if (relocalization_bootstrap_steps_ == 0) {
                    relocalization_bootstrap_steps_ = 1;
                    attempted_seed = seed;
                    recovery_candidate = make_relocalization_bootstrap(
                        detail->feature_tracker->map.get(), seed,
                        relocalization_last_motion_, config,
                        recovery_anchor_tracks);
                    if (recovery_candidate) {
                        recovery_candidate->feature_tracking_map =
                            detail->feature_tracker->map;
                        recovery_candidate->set_detail(detail);
                        recovery_candidate->mirror_frame(
                            detail->feature_tracker->map.get(),
                            seed.second.current_frame_id);
                        if (recovery_candidate->map->frame_num() == 2) {
                            recovery_transfer_tracks =
                                count_relocalization_mapped_tracks(
                                    recovery_candidate->map->get_frame(1));
                        }
                    }
                    std::fprintf(stderr,
                        "[RelocalizationBootstrap] event=prepared "
                        "session=%zu first_current=%zu current=%zu "
                        "seeded=%zu transferred=%zu candidate=%d\n",
                        seed.first.session_id, seed.first.current_frame_id,
                        seed.second.current_frame_id,
                        recovery_anchor_tracks, recovery_transfer_tracks,
                        recovery_candidate ? 1 : 0);
                }
            }
            initializer->mirror_keyframe_map(detail->feature_tracker->map.get(),
                                             pending_frame_id);
        }
        // With recovered 3D anchors, bypass the failed monocular SfM
        // startup and try the ordinary visual+IMU update on an isolated map.
        // Do not publish it unless that update agrees with PnP and retains
        // real, temporally tracked visual observations.
        if (recovery_candidate && attempted_seed &&
            recovery_transfer_tracks >= 8) {
            const bool solved = recovery_candidate->track();
            const auto [t, body_pose, motion] =
                recovery_candidate->get_latest_state();
            const auto &hypothesis = attempted_seed->second;
            const quaternion camera_q = body_pose.q *
                recovery_candidate->map->get_frame(
                    recovery_candidate->map->frame_num() - 1)->camera.q_cs;
            const auto *last = recovery_candidate->map->get_frame(
                recovery_candidate->map->frame_num() - 1);
            const vector<3> camera_p = body_pose.p +
                body_pose.q * last->camera.p_cs;
            const double dp = (camera_p - hypothesis.camera_pose.p).norm();
            const double dr = relocalization_pose_disagreement_deg(
                camera_q, hypothesis.camera_pose.q);
            const bool accepted = solved && t == hypothesis.timestamp &&
                body_pose.p.allFinite() &&
                body_pose.q.coeffs().allFinite() &&
                motion.v.allFinite() &&
                dp <= 1.0 && dr <= 20.0;
            std::fprintf(stderr,
                "[RelocalizationBootstrap] event=handoff "
                "current=%zu accepted=%d visual_tracks=%zu "
                "delta_position_m=%.6f delta_rotation_deg=%.6f "
                "timestamp=%.9f\n",
                pending_frame_id, accepted ? 1 : 0,
                recovery_transfer_tracks, dp, dr, t);
            if (accepted) {
                sliding_window_tracker = std::move(recovery_candidate);
                relocalization_bootstrap_tracking_ = true;
                relocalization_active_reference_session_id_ =
                    attempted_seed->first.session_id;
                relocalization_bootstrap_steps_ = 0;
                relocalization_bootstrap_good_steps_ = 0;
            }
        }
        if (!sliding_window_tracker)
            sliding_window_tracker = initializer->initialize();
        if (sliding_window_tracker) {
#if defined(XRSLAM_IOS)
            synchronized(detail->feature_tracker->keymap) {
                detail->feature_tracker->synchronize_keymap(
                    sliding_window_tracker->map.get());
            }
#endif
            if (config->visual_localization_enable() &&
                global_localization_state()) {
                localizer = std::make_unique<Localizer>(config);
                sliding_window_tracker->map->create_virtual_object_manager(localizer.get());
            } else {
                sliding_window_tracker->map->create_virtual_object_manager();
            }
            sliding_window_tracker->feature_tracking_map = detail->feature_tracker->map;
            sliding_window_tracker->set_detail(detail);
            std::unique_lock lk(latest_state_mutex);
            auto [t, pose, motion] = sliding_window_tracker->get_latest_state();
            latest_state = {t, pending_frame_id, pose, motion};
            global_drift_available_ = false;
            relocalization_global_alignment_active_ = false;
            if (relocalization_bootstrap_tracking_) {
                const auto saved_alignment =
                    persistent_relocalization_map_.global_alignment_for_session(
                        relocalization_active_reference_session_id_);
                if (saved_alignment &&
                    std::isfinite(saved_alignment->yaw_deg) &&
                    saved_alignment->translation.allFinite() &&
                    saved_alignment->timestamp <= t) {
                    // Recovered PnP world coordinates are the *reference*
                    // session's VIO coordinates. Restore only its published
                    // output transform; estimator states remain unchanged.
                    global_drift_snapshot_ = *saved_alignment;
                    global_drift_available_ = true;
                    relocalization_global_alignment_active_ = true;
                }
                std::fprintf(stderr,
                    "[RelocalizationGlobalAnchor] event=restored "
                    "session=%zu current=%zu t=%.9f active=%d "
                    "yaw_deg=%.9f translation=%.9f,%.9f,%.9f "
                    "state_mutation=0\n",
                    relocalization_active_reference_session_id_,
                    pending_frame_id, t,
                    relocalization_global_alignment_active_ ? 1 : 0,
                    relocalization_global_alignment_active_
                        ? global_drift_snapshot_.yaw_deg : 0.0,
                    relocalization_global_alignment_active_
                        ? global_drift_snapshot_.translation.x() : 0.0,
                    relocalization_global_alignment_active_
                        ? global_drift_snapshot_.translation.y() : 0.0,
                    relocalization_global_alignment_active_
                        ? global_drift_snapshot_.translation.z() : 0.0);
            }
            lk.unlock();
            if (persistent_relocalization_map_.session_count() != 0) {
                std::fprintf(
                    stderr,
                    "[RelocalizationReferenceMap] event=new_session "
                    "retained_sessions=%zu retained_archived=%zu "
                    "retained_places=%zu candidate_references=%zu "
                    "estimator_mutation=0\n",
                    persistent_relocalization_map_.session_count(),
                    persistent_relocalization_map_.archived_keyframe_count(),
                    persistent_relocalization_map_.place_keyframe_count(),
                    persistent_relocalization_map_.candidate_reference_count());
            }
            if (relocalization_bootstrap_tracking_) {
                std::fprintf(stderr,
                    "[RelocalizationBootstrap] event=new_vio_session "
                    "current=%zu source=historical_pnp\n", pending_frame_id);
            }
            initializer.reset();
            relocalization_auto_loss_gate_.reset();
        }
    } else if (sliding_window_tracker) {
        size_t pending_frame_id = pending_frame_ids.front();
        pending_frame_ids.pop_front();
        l.unlock();
        synchronized(detail->feature_tracker->map) {
            sliding_window_tracker->mirror_frame(
                detail->feature_tracker->map.get(), pending_frame_id);
        }
        bool tracking_ok = sliding_window_tracker->track();
        // Experimental automatic-loss gate. The passive 0121g measurements
        // and criteria are unchanged. A persistent run enters the SAME
        // failure path as a normal tracker failure. This is opt-in because
        // its calibration is based on one healthy EuRoC sequence so far.
        const char *auto_loss = std::getenv(
            "XRSLAM_RELOCALIZATION_AUTO_LOSS");
        if (tracking_ok && auto_loss && std::string(auto_loss) == "1") {
            Frame *current = sliding_window_tracker->map->get_frame(
                sliding_window_tracker->map->frame_num() - 1);
            if (!current->subframes.empty())
                current = current->subframes.back().get();
            const RelocalizationTrackingHealth health =
                measure_relocalization_tracking_health(current);
            const double inlier_fraction = health.reprojectable == 0 ? 0.0 :
                static_cast<double>(health.inliers_6px) /
                static_cast<double>(health.reprojectable);
            const bool suspect = !health.pose_finite ||
                health.mapped < 8 ||
                (health.reprojectable >= 8 && inlier_fraction < 0.20);
            const bool trigger = relocalization_auto_loss_gate_.observe(suspect);
            std::fprintf(stderr,
                "[RelocalizationAutoLoss] event=sample "
                "frame=%zu t=%.9f suspect=%d mapped=%zu "
                "reprojectable=%zu inlier6_fraction=%.6f "
                "pose_finite=%d observed=%zu streak=%zu "
                "grace=%zu required=%zu trigger=%d\n",
                health.frame_id, health.timestamp, suspect ? 1 : 0,
                health.mapped, health.reprojectable, inlier_fraction,
                health.pose_finite ? 1 : 0,
                relocalization_auto_loss_gate_.observed_frames(),
                relocalization_auto_loss_gate_.suspicious_run(),
                RelocalizationAutoLossGate::grace_frames,
                RelocalizationAutoLossGate::required_suspicious_frames,
                trigger ? 1 : 0);
            if (trigger) {
                tracking_ok = false;
                std::fprintf(stderr,
                    "[RelocalizationAutoLoss] event=triggered "
                    "frame=%zu t=%.9f streak=%zu "
                    "mapped=%zu inlier6_fraction=%.6f "
                    "pose_finite=%d source=tracking_health\n",
                    health.frame_id, health.timestamp,
                    relocalization_auto_loss_gate_.suspicious_run(),
                    health.mapped, inlier_fraction,
                    health.pose_finite ? 1 : 0);
            }
        } else {
            relocalization_auto_loss_gate_.reset();
        }
        if (tracking_ok) {
            if (relocalization_bootstrap_tracking_) {
                Frame *last = sliding_window_tracker->map->get_frame(
                    sliding_window_tracker->map->frame_num() - 1);
                if (!last->subframes.empty())
                    last = last->subframes.back().get();
                const size_t visual_tracks =
                    count_relocalization_mapped_tracks(last);
                const bool good = visual_tracks >= 8 &&
                    last->pose.p.allFinite() &&
                    last->pose.q.coeffs().allFinite();
                ++relocalization_bootstrap_steps_;
                if (good) ++relocalization_bootstrap_good_steps_;
                std::fprintf(stderr,
                    "[RelocalizationBootstrap] event=tracking_step "
                    "current=%zu step=%zu visual_tracks=%zu good=%d "
                    "timestamp=%.9f\n",
                    pending_frame_id, relocalization_bootstrap_steps_,
                    visual_tracks, good ? 1 : 0,
                    last->image ? last->image->t : 0.0);
                if (relocalization_bootstrap_steps_ == 20) {
                    std::fprintf(stderr,
                        "[RelocalizationBootstrap] event=followup "
                        "steps=20 good_steps=%zu sustained=%d\n",
                        relocalization_bootstrap_good_steps_,
                        relocalization_bootstrap_good_steps_ >= 15 ? 1 : 0);
                }
            }
#if defined(XRSLAM_IOS)
            synchronized(detail->feature_tracker->keymap) {
                detail->feature_tracker->synchronize_keymap(
                    sliding_window_tracker->map.get());
            }
#endif
            std::unique_lock lk(latest_state_mutex);
            auto [t, pose, motion] = sliding_window_tracker->get_latest_state();
            latest_state = {t, pending_frame_id, pose, motion};
            PlaceGraph4DoFCorrection correction;
            if (!relocalization_bootstrap_tracking_ &&
                sliding_window_tracker->get_latest_global_correction(
                    correction)) {
                const bool new_snapshot =
                    !global_drift_available_ ||
                    global_drift_snapshot_.frame_id != correction.frame_id;
                global_drift_snapshot_ = correction;
                global_drift_available_ = true;
                if (new_snapshot) {
                    std::fprintf(
                        stderr,
                        "[PlaceGlobalDriftSnapshotShadow] "
                        "frame_id=%zu t=%.9f "
                        "yaw_correction_deg=%.9f "
                        "translation=%.9f,%.9f,%.9f "
                        "state_mutation=0\n",
                        correction.frame_id, correction.timestamp,
                        correction.yaw_deg, correction.translation.x(),
                        correction.translation.y(),
                        correction.translation.z());
                }
            }
            lk.unlock();
        } else {
            auto [last_t, last_pose, last_motion] =
                sliding_window_tracker->get_latest_state();
            relocalization_last_motion_ = last_motion;
            relocalization_bootstrap_tracking_ = false;
            relocalization_global_alignment_active_ = false;
            relocalization_active_reference_session_id_ = 0;
            relocalization_bootstrap_steps_ = 0;
            relocalization_bootstrap_good_steps_ = 0;
            // The correction at the time of loss maps THIS session's local
            // archive coordinates to the same global output frame as the
            // last published pose. It must not be exposed during loss.
            std::optional<PlaceGraph4DoFCorrection> lost_alignment;
            {
                std::unique_lock lk(latest_state_mutex);
                if (global_drift_available_)
                    lost_alignment = global_drift_snapshot_;
                latest_state = {{}, nil(), {}, {}};
                global_drift_available_ = false;
            }
            // Preserve historical, value-owned place and geometry records
            // before destroying this VIO session. Never mix its local world
            // coordinates into the newly initialized local estimator/graph.
            auto reference =
                sliding_window_tracker->take_relocalization_reference();
            reference.global_alignment = lost_alignment;
            const size_t archived = reference.archive.size();
            const size_t places = reference.places.size();
            const bool captured =
                persistent_relocalization_map_.capture(std::move(reference));
            std::fprintf(stderr,
                "[RelocalizationGlobalAnchor] event=lost "
                "session=%zu active=%d yaw_deg=%.9f "
                "translation=%.9f,%.9f,%.9f state_mutation=0\n",
                captured ? persistent_relocalization_map_.last_session_id() : 0,
                captured && lost_alignment ? 1 : 0,
                lost_alignment ? lost_alignment->yaw_deg : 0.0,
                lost_alignment ? lost_alignment->translation.x() : 0.0,
                lost_alignment ? lost_alignment->translation.y() : 0.0,
                lost_alignment ? lost_alignment->translation.z() : 0.0);
            relocalization_consensus_.reset();
            std::fprintf(
                stderr,
                "[RelocalizationReferenceMap] event=tracking_lost "
                "captured=%d session=%zu archived=%zu places=%zu "
                "retained_sessions=%zu retained_archived=%zu "
                "retained_places=%zu candidate_references=%zu "
                    "estimator_mutation=0\n",
                captured ? 1 : 0,
                persistent_relocalization_map_.last_session_id(),
                archived, places,
                persistent_relocalization_map_.session_count(),
                persistent_relocalization_map_.archived_keyframe_count(),
                persistent_relocalization_map_.place_keyframe_count(),
                persistent_relocalization_map_.candidate_reference_count());
            initializer = std::make_unique<Initializer>(config);
            sliding_window_tracker.reset();
        }
    }
}

void FrontendWorker::issue_frame(Frame *frame) {
    auto l = lock();
    pending_frame_ids.push_back(frame->id());
    resume(l);
}

std::tuple<double, size_t, PoseState, MotionState>
FrontendWorker::get_latest_state() const {
    std::unique_lock lk(latest_state_mutex);
    return latest_state;
}

bool FrontendWorker::get_latest_global_correction(
    PlaceGraph4DoFCorrection &correction) const {
    std::unique_lock lk(latest_state_mutex);
    if (!global_drift_available_)
        return false;
    correction = global_drift_snapshot_;
    return true;
}

size_t FrontendWorker::create_virtual_object() {
    auto l = lock();
    if (sliding_window_tracker) {
        return sliding_window_tracker->map->create_virtual_object();
    } else {
        return nil();
    }
    l.unlock();
}

OutputObject FrontendWorker::get_virtual_object_pose_by_id(size_t id) {
    auto l = lock();
    if (sliding_window_tracker) {
        return sliding_window_tracker->map->get_virtual_object_pose_by_id(id);
    } else {
        return {{0.0, 0.0, 0.0, 1.0}, {1000.0, 1000.0, 1000.0}, 1};
    }
    l.unlock();
}

SysState FrontendWorker::get_system_state() const {
    if (initializer) {
        return SysState::SYS_INITIALIZING;
    } else if (sliding_window_tracker) {
        return SysState::SYS_TRACKING;
    }
    return SysState::SYS_UNKNOWN;
}

void FrontendWorker::query_frame() {
    if (localizer)
        localizer->query_frame();
}

bool FrontendWorker::global_localization_state() const {
    return global_localization_flag;
}

void FrontendWorker::set_global_localization_state(bool state) {
    global_localization_flag = state;
}

} // namespace xrslam
