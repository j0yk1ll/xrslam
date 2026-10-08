#ifndef XRSLAM_MARGINALIZATION_FACTOR_H
#define XRSLAM_MARGINALIZATION_FACTOR_H

#include <xrslam/common.h>
#include <xrslam/map/frame.h>
#include <xrslam/map/map.h>

namespace xrslam {

class MarginalizationFactor {
  public:
    virtual ~MarginalizationFactor() = default;
    virtual void marginalize(size_t index) = 0;

    // Clone the stored prior into a new world gauge. Concrete backends that
    // support this must transform both their linearization state and
    // information basis. The default keeps non-inertial backends unchanged.
    virtual std::unique_ptr<MarginalizationFactor> clone_rebased_world(
        const std::vector<Frame *> &rebased_frames,
        const quaternion &world_q,
        const vector<3> &world_p) const {
        return nullptr;
    }

    const std::vector<Frame *> &linearization_frames() const { return frames; }

  protected:
    MarginalizationFactor(const MarginalizationFactor &) = default;

    MarginalizationFactor(Map *map) : base_map(map) {
        frames.resize(map->frame_num() - 1);
        pose_linearization_point.resize(map->frame_num() - 1);
        motion_linearization_point.resize(map->frame_num() - 1);
        for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
            Frame *frame = map->get_frame(i);
            frames[i] = frame;
            pose_linearization_point[i] = frame->pose;
            motion_linearization_point[i] = frame->motion;
        }
        infovec.setZero(ES_SIZE * (map->frame_num() - 1));
        sqrt_inv_cov.setZero(ES_SIZE * (map->frame_num() - 1),
                             ES_SIZE * (map->frame_num() - 1));
        sqrt_inv_cov.block<3, 3>(ES_P, ES_P) = 1.0e15 * matrix<3>::Identity();
        sqrt_inv_cov.block<3, 3>(ES_Q, ES_Q) = 1.0e15 * matrix<3>::Identity();
    }

    std::vector<PoseState> pose_linearization_point;
    std::vector<MotionState> motion_linearization_point;
    matrix<> sqrt_inv_cov;
    vector<> infovec;
    std::vector<Frame *> frames;
    Map *base_map;
};

} // namespace xrslam

#endif // XRSLAM_MARGINALIZATION_FACTOR_H
