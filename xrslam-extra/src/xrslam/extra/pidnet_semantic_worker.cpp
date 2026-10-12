#include <xrslam/extra/pidnet_semantic_worker.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace xrslam::extra {

PidNetSemanticWorker::PidNetSemanticWorker(
    const std::string &model_path, const std::string &device,
    std::uint32_t classes, int canvas_width, int canvas_height,
    std::string taxonomy, std::string dump_dir, std::size_t max_pending)
    : segmenter_(model_path, device, classes, canvas_width, canvas_height),
      taxonomy_(std::move(taxonomy)), dump_dir_(std::move(dump_dir)),
      max_pending_(max_pending), landmark_map_(classes) {
    if (max_pending_ == 0 || max_pending_ > 64)
        throw std::invalid_argument("semantic worker queue must have 1..64 slots");
    if (taxonomy_.empty())
        throw std::invalid_argument("semantic taxonomy must not be empty");
    if (!dump_dir_.empty())
        std::filesystem::create_directories(dump_dir_);
    thread_ = std::thread([this] { run(); });
}

PidNetSemanticWorker::~PidNetSemanticWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wakeup_.notify_all();
    // Drain the bounded queue so accepted jobs are not silently discarded.
    if (thread_.joinable()) thread_.join();
}

bool PidNetSemanticWorker::submit(SemanticKeyframeSnapshot &&snapshot) noexcept {
    try {
        if (!snapshot.image.valid() || snapshot.session_id == 0 ||
            !std::isfinite(snapshot.timestamp))
            return false;
        const auto id = std::make_pair(snapshot.session_id, snapshot.frame_id);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || records_.count(id) ||
            pending_.size() >= max_pending_) {
            if (pending_.size() >= max_pending_)
                ++dropped_;
            return false;
        }
        pending_.emplace_back(std::move(snapshot));
        wakeup_.notify_one();
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<SemanticFusedLandmark> PidNetSemanticWorker::find_landmark(
    std::uint64_t session_id, std::size_t track_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.find(session_id, track_id);
}

std::vector<SemanticFusedLandmark> PidNetSemanticWorker::landmark_snapshot(
    std::optional<std::uint64_t> session_id, bool stable_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.snapshot(session_id, stable_only);
}

std::size_t PidNetSemanticWorker::landmark_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.size();
}

std::size_t PidNetSemanticWorker::stable_landmark_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return landmark_map_.stable_count();
}

std::size_t PidNetSemanticWorker::record_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
}

std::optional<SemanticKeyframeRecord> PidNetSemanticWorker::find(
    std::uint64_t session_id, std::uint64_t frame_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find({session_id, frame_id});
    if (it == records_.end()) return std::nullopt;
    return it->second;
}

void PidNetSemanticWorker::run() noexcept {
    for (;;) {
        SemanticKeyframeSnapshot job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wakeup_.wait(lock, [this] {return stopping_ || !pending_.empty();});
            if (pending_.empty()) return;
            job = std::move(pending_.front());
            pending_.pop_front();
        }
        try {
            process(std::move(job));
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                "[PIDNetSemantic] event=error session=%llu frame=%llu error=%s\n",
                static_cast<unsigned long long>(job.session_id),
                static_cast<unsigned long long>(job.frame_id), e.what());
        } catch (...) {
            std::fprintf(stderr, "[PIDNetSemantic] event=error unknown=1\n");
        }
    }
}

void PidNetSemanticWorker::process(SemanticKeyframeSnapshot job) {
    const int type = CV_MAKETYPE(CV_8U, static_cast<int>(job.image.channels));
    // Job owns its packed pixels until inference has completed.
    const cv::Mat input(static_cast<int>(job.image.height),
                        static_cast<int>(job.image.width), type,
                        job.image.bytes.data());
    const SemanticSegmentationResult output = segmenter_.segment(input);
    if (output.label.size() != input.size() ||
        output.confidence.size() != input.size() ||
        output.label.type() != CV_8UC1 ||
        output.confidence.type() != CV_32FC1)
        throw std::runtime_error("PIDNet worker: unexpected mask dimensions/type");

    SemanticKeyframeRecord rec;
    rec.session_id = job.session_id;
    rec.frame_id = job.frame_id;
    rec.timestamp = job.timestamp;
    rec.width = job.image.width;
    rec.height = job.image.height;
    rec.num_classes = output.num_classes;
    rec.taxonomy = taxonomy_;
    rec.camera_position = job.camera_position;
    rec.camera_quaternion_xyzw = job.camera_quaternion_xyzw;
    rec.K_row_major = job.K_row_major;

    if (!cv::imencode(".png", output.label, rec.labels_png))
        throw std::runtime_error("PIDNet worker: failed to encode label PNG");
    cv::Mat quantized_confidence;
    output.confidence.convertTo(quantized_confidence, CV_8UC1, 255.0);
    if (!cv::imencode(".png", quantized_confidence, rec.confidence_png))
        throw std::runtime_error("PIDNet worker: failed to encode confidence PNG");

    rec.landmarks.reserve(job.observations.size());
    for (const auto &obs : job.observations) {
        if (!std::isfinite(obs.u) || !std::isfinite(obs.v)) continue;
        const int x = static_cast<int>(std::lround(obs.u));
        const int y = static_cast<int>(std::lround(obs.v));
        if (x < 0 || y < 0 || x >= output.label.cols ||
            y >= output.label.rows) continue;
        rec.landmarks.push_back({obs.track_id, obs.keypoint_index,
            output.label.at<std::uint8_t>(y, x),
            output.confidence.at<float>(y, x),
            obs.position_session_world});
    }

    if (!dump_dir_.empty()) {
        const auto base = std::to_string(job.session_id) + "_" +
                          std::to_string(job.frame_id);
        const std::filesystem::path dir(dump_dir_);
        if (!cv::imwrite((dir / (base + "_labels.png")).string(), output.label))
            throw std::runtime_error("PIDNet worker: failed to write label PNG");
    }

    // A track may appear multiple times in a keyframe; choose the
    // highest-confidence pixel per track. One keyframe is *one* vote per
    // landmark, not one vote per feature/keypoint alias.
    std::map<std::size_t, SemanticLandmarkVote> votes;
    for (const auto &sample : rec.landmarks) {
        SemanticLandmarkVote vote;
        vote.session_id = rec.session_id;
        vote.frame_id = rec.frame_id;
        vote.track_id = sample.track_id;
        vote.label = sample.label;
        vote.confidence = sample.confidence;
        vote.position_session_world = sample.position_session_world;
        const auto it = votes.find(sample.track_id);
        if (it == votes.end() || vote.confidence > it->second.confidence)
            votes[sample.track_id] = vote;
    }

    const auto samples = rec.landmarks.size();
    std::size_t count = 0, fused = 0, landmark_count = 0, stable_count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = std::make_pair(rec.session_id, rec.frame_id);
        if (records_.count(id)) return; // never double-count a keyframe
        for (const auto &[track, vote] : votes) {
            if (landmark_map_.observe(vote)) ++fused;
        }
        records_.emplace(id, std::move(rec));
        count = records_.size();
        landmark_count = landmark_map_.size();
        stable_count = landmark_map_.stable_count();
    }
    std::fprintf(stderr,
        "[PIDNetSemantic] event=processed session=%llu frame=%llu "
        "t=%.9f size=%ux%u sampled=%zu stored=%zu "
        "fused_views=%zu landmark_tracks=%zu stable_tracks=%zu "
        "state_mutation=0\n",
        static_cast<unsigned long long>(job.session_id),
        static_cast<unsigned long long>(job.frame_id), job.timestamp,
        job.image.width, job.image.height, samples, count,
        fused, landmark_count, stable_count);
}

} // namespace xrslam::extra
