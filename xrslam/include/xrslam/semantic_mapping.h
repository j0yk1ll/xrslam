#ifndef XRSLAM_SEMANTIC_MAPPING_H
#define XRSLAM_SEMANTIC_MAPPING_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <xrslam/version.h>

namespace xrslam {

// Core-only value types. No LibTorch, OpenCV, ROS, Frame or Track lifetime.
// Pixels share the same undistorted coordinate system as archived observations.
struct SemanticPixelBuffer {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0; // 1=gray, 3=BGR, 4=BGRA
    std::vector<std::uint8_t> bytes; // packed rows; owns all memory

    bool valid() const {
        return width > 0 && height > 0 &&
               (channels == 1 || channels == 3 || channels == 4) &&
               bytes.size() == static_cast<std::size_t>(width) *
                               height * channels;
    }
};

struct SemanticArchiveObservation {
    std::size_t track_id = 0;
    std::size_t keypoint_index = 0;
    double u = 0.0;
    double v = 0.0;
    // Optimized landmark position in this VIO session local world.
    // Not globally aligned and never sent back to the estimator.
    std::array<double, 3> position_session_world{};
};

struct SemanticKeyframeSnapshot {
    std::uint64_t session_id = 0; // new active VIO session, starts at 1
    std::uint64_t frame_id = 0;
    double timestamp = 0.0;
    SemanticPixelBuffer image;
    // Session-local optimized camera pose at first archival. Latest geometry
    // remains in KeyframeArchive, which can be refreshed independently.
    std::array<double, 3> camera_position{};
    std::array<double, 4> camera_quaternion_xyzw{};
    std::array<double, 9> K_row_major{};
    std::vector<SemanticArchiveObservation> observations;
};

class SemanticKeyframeSink {
  public:
    virtual ~SemanticKeyframeSink() = default;
    // Never perform neural inference in submit(). Queue may be full; dropping
    // semantic work must never stop or back-pressure visual-inertial tracking.
    virtual bool submit(SemanticKeyframeSnapshot &&snapshot) noexcept = 0;
};

} // namespace xrslam

#endif // XRSLAM_SEMANTIC_MAPPING_H
