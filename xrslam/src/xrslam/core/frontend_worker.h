#ifndef XRSLAM_FRONTEND_WORKER_H
#define XRSLAM_FRONTEND_WORKER_H
#include <xrslam/common.h>
#include <xrslam/core/place_graph_4dof_shadow.h>
#include <xrslam/core/persistent_relocalization_map.h>
#include <xrslam/core/relocalization_consensus.h>
#include <xrslam/core/relocalization_auto_loss.h>
#include <xrslam/estimation/state.h>
#include <xrslam/utility/worker.h>

namespace xrslam {

class Config;
class Frame;
class Initializer;
class SlidingWindowTracker;
class Localizer;

class FrontendWorker : public Worker {
  public:
    FrontendWorker(XRSLAM::Detail *detail, std::shared_ptr<Config> config);
    ~FrontendWorker();

    bool empty() const override;
    void work(std::unique_lock<std::mutex> &l) override;

    void issue_frame(Frame *frame);

    size_t create_virtual_object();
    OutputObject get_virtual_object_pose_by_id(size_t id);

    std::tuple<double, size_t, PoseState, MotionState> get_latest_state() const;
    // Copies the latest usable graph correction under latest_state_mutex.
    // Returns false before any solve and during tracking reinitialization;
    // a verified historical PnP handoff can restore a saved alignment.
    bool get_latest_global_correction(
        PlaceGraph4DoFCorrection &correction) const;
    // Same-worker-thread access for future relocalization verification.
    const PersistentRelocalizationMap &relocalization_reference_map() const {
        return persistent_relocalization_map_;
    }
    SysState get_system_state() const;

    bool global_localization_state() const;
    void set_global_localization_state(bool state);
    void query_frame();

    std::unique_ptr<Localizer> localizer;

  private:
    std::deque<size_t> pending_frame_ids;

    XRSLAM::Detail *detail;
    std::shared_ptr<Config> config;
    std::unique_ptr<Initializer> initializer;
    std::unique_ptr<SlidingWindowTracker> sliding_window_tracker;

    std::tuple<double, size_t, PoseState, MotionState> latest_state;
    mutable std::mutex latest_state_mutex;
    PlaceGraph4DoFCorrection global_drift_snapshot_;
    bool global_drift_available_ = false;

    // Shadow-only two-frame verification; no estimator/graph mutation.
    RelocalizationConsensus relocalization_consensus_;
    MotionState relocalization_last_motion_;
    bool relocalization_bootstrap_tracking_ = false;
    RelocalizationAutoLossGate relocalization_auto_loss_gate_;
    size_t relocalization_active_reference_session_id_ = 0;
    // The recovered VIO local map is expressed in the old session's frame.
    // Keep its saved output alignment while the new graph has no cross-session
    // gauge constraint; never feed this into the estimator itself.
    bool relocalization_global_alignment_active_ = false;
    size_t relocalization_bootstrap_steps_ = 0;
    size_t relocalization_bootstrap_good_steps_ = 0;

    // Retained across SlidingWindowTracker destruction. Each capture keeps
    // its own world-frame identity and never feeds the frozen place graph.
    PersistentRelocalizationMap persistent_relocalization_map_;

    bool global_localization_flag = false;
};

} // namespace xrslam

#endif // XRSLAM_FRONTEND_WORKER_H
