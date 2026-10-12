#ifndef XRSLAM_EXTRA_PIDNET_SEMANTIC_WORKER_H
#define XRSLAM_EXTRA_PIDNET_SEMANTIC_WORKER_H

#include <xrslam/semantic_mapping.h>
#include <xrslam/semantic_landmark_map.h>
#include <xrslam/extra/pidnet_segmenter.h>

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xrslam::extra {

struct SemanticLandmarkSample {
    std::size_t track_id = 0;
    std::size_t keypoint_index = 0;
    std::uint8_t label = 0;
    float confidence = 0.0f;
    std::array<double, 3> position_session_world{};
};

struct SemanticKeyframeRecord {
    std::uint64_t session_id = 0;
    std::uint64_t frame_id = 0;
    double timestamp = 0.0;
    std::uint32_t width = 0, height = 0;
    std::uint32_t num_classes = 0;
    std::string taxonomy;
    std::array<double, 3> camera_position{};
    std::array<double, 4> camera_quaternion_xyzw{};
    std::array<double, 9> K_row_major{};
    std::vector<std::uint8_t> labels_png;      // compressed losslessly
    std::vector<std::uint8_t> confidence_png;  // uint8 quantized 0..255
    std::vector<SemanticLandmarkSample> landmarks;
};

// In-process asynchronous PIDNet executor and session-aware semantic archive.
// Records survive sliding-window destruction and VIO tracking-session changes
// for as long as the Detail instance owning this worker exists.
class PidNetSemanticWorker final : public SemanticKeyframeSink {
  public:
    PidNetSemanticWorker(const std::string &model_path,
                         const std::string &device,
                         std::uint32_t classes,
                         int canvas_width,
                         int canvas_height,
                         std::string taxonomy,
                         std::string dump_dir = {},
                         std::size_t max_pending = 2);
    ~PidNetSemanticWorker() override;

    PidNetSemanticWorker(const PidNetSemanticWorker &) = delete;
    PidNetSemanticWorker &operator=(const PidNetSemanticWorker &) = delete;

    bool submit(SemanticKeyframeSnapshot &&snapshot) noexcept override;

    std::optional<SemanticKeyframeRecord> find(
        std::uint64_t session_id, std::uint64_t frame_id) const;
    std::size_t record_count() const;

    // Persistent session-local semantic landmark map; immutable copies.
    std::optional<SemanticFusedLandmark> find_landmark(
        std::uint64_t session_id, std::size_t track_id) const;
    std::vector<SemanticFusedLandmark> landmark_snapshot(
        std::optional<std::uint64_t> session_id = std::nullopt,
        bool stable_only = false) const;
    std::size_t landmark_count() const;
    std::size_t stable_landmark_count() const;

  private:
    void run() noexcept;
    void process(SemanticKeyframeSnapshot snapshot);

    PidNetSegmenter segmenter_;
    const std::string taxonomy_;
    const std::string dump_dir_;
    const std::size_t max_pending_;

    mutable std::mutex mutex_;
    std::condition_variable wakeup_;
    std::deque<SemanticKeyframeSnapshot> pending_;
    std::map<std::pair<std::uint64_t, std::uint64_t>,
             SemanticKeyframeRecord> records_;
    SemanticLandmarkMap landmark_map_; // guarded by mutex_
    bool stopping_ = false;
    std::uint64_t dropped_ = 0;
    std::thread thread_;
};

} // namespace xrslam::extra

#endif // XRSLAM_EXTRA_PIDNET_SEMANTIC_WORKER_H
