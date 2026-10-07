#ifndef XRSLAM_SLIDING_WINDOW_TRACKER_H
#define XRSLAM_SLIDING_WINDOW_TRACKER_H

#include <xrslam/common.h>
#include <xrslam/core/keyframe_archive.h>
#include <xrslam/estimation/state.h>
#include <xrslam/place_recognition.h>

#include <unordered_set>

namespace xrslam {

class Config;
class Frame;
class Map;

class SlidingWindowTracker {
  public:
    SlidingWindowTracker(std::unique_ptr<Map> keyframe_map,
                         std::shared_ptr<Config> config);
    ~SlidingWindowTracker();

    void mirror_frame(Map *feature_tracking_map, size_t frame_id);

    void localize_newframe();
    void track_landmark();
    void refine_window();
    void slide_window();
    bool manage_keyframe();

    void refine_subwindow();

    bool judge_track_status();
    void fuse_imu_track();
    bool check_frames_rpe(Track *track, const vector<3> &p);
    void predict_RT(Frame *frame_i, Frame *frame_j, matrix<3> &R, vector<3> &t);
    bool filter_parsac_2d2d(Frame *frame_i, Frame *frame_j,
                            std::vector<char> &mask,
                            std::vector<size_t> &pts_to_index);
    void update_track_status();
    void synchronize_feature_tracking_landmarks();

    std::vector<vector<3>> m_P3D;
    std::vector<vector<2>> m_P2D;
    std::vector<size_t> m_lens;
    std::vector<int> m_indices_map;

    bool track();

    std::tuple<double, PoseState, MotionState> get_latest_state() const;

    double m_th;
    std::unique_ptr<Map> map;
    std::shared_ptr<Map> feature_tracking_map;

    void set_detail(XRSLAM::Detail *detail) { this->detail = detail; }
  private:
    struct PendingPlaceRetrieval {
        size_t frame_id = 0;
        std::vector<PlaceCandidate> candidates;
    };

    struct PlaceNeighborhoodEventShadowState {
        size_t event_id = 0;
        double reference_t_min = 0.0;
        double reference_t_max = 0.0;
        size_t consecutive_age = 0;
        size_t last_current_frame_id = 0;
    };

    void archive_optimized_keyframes();
    void diagnose_orb_association(Frame *frame);
    void diagnose_retrieved_place_candidates(
        Frame *frame, const std::vector<PlaceCandidate> &candidates);
    void diagnose_pending_retrieved_place_candidates();
    void extract_place_descriptors();

    KeyframeArchive keyframe_archive_;
    PlaceKeyframeStore place_keyframes_;
    std::vector<PendingPlaceRetrieval>
        pending_place_retrieval_candidates_;
    size_t previous_verified_current_frame_id_ =
        static_cast<size_t>(-1);
    std::vector<PlaceKey> previous_verified_place_keys_;
    std::vector<PlaceKey> previous_temporal_dt2_place_keys_;
    std::vector<PlaceKey> previous_temporal_dt5_place_keys_;
    std::vector<PlaceKey> previous_temporal_dt10_place_keys_;
    std::vector<PlaceNeighborhoodEventShadowState>
        previous_dt5_neighborhood_events_;
    std::vector<PlaceNeighborhoodEventShadowState>
        previous_dt10_neighborhood_events_;
    size_t next_place_neighborhood_event_id_ = 1;
    size_t local_descriptor_keyframe_count_ = 0;
    std::unordered_set<size_t> orb_association_processed_;
    std::shared_ptr<Config> config;

    XRSLAM::Detail *detail = nullptr;
};

} // namespace xrslam

#endif // XRSLAM_SLIDING_WINDOW_TRACKER_H
