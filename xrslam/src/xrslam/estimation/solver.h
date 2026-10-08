#ifndef XRSLAM_SOLVER_H
#define XRSLAM_SOLVER_H

#include <xrslam/common.h>
#include <xrslam/estimation/marginalization_factor.h>
#include <xrslam/estimation/preintegration_factor.h>
#include <xrslam/estimation/reprojection_factor.h>
#include <xrslam/estimation/rotation_factor.h>

namespace xrslam {

class Map;
class Frame;
class Track;

class Solver {
    struct SolverDetails;
    Solver();

  public:
    virtual ~Solver();

    static void init(Config *config);

    static std::unique_ptr<Solver> create();

    static std::unique_ptr<ReprojectionErrorFactor>
    create_reprojection_error_factor(Frame *frame, Track *track);
    static std::unique_ptr<ReprojectionPriorFactor>
    create_reprojection_prior_factor(Frame *frame, Track *track);
    static std::unique_ptr<RotationPriorFactor>
    create_rotation_prior_factor(Frame *frame, Track *track);
    static std::unique_ptr<PreIntegrationErrorFactor>
    create_preintegration_error_factor(Frame *frame_i, Frame *frame_j,
                                       const PreIntegrator &preintegration);
    static std::unique_ptr<PreIntegrationPriorFactor>
    create_preintegration_prior_factor(Frame *frame_i, Frame *frame_j,
                                       const PreIntegrator &preintegration);
    static std::unique_ptr<MarginalizationFactor>
    create_marginalization_factor(Map *map);

    virtual void add_frame_states(Frame *frame, bool with_motion = true);
    virtual void add_track_states(Track *track);

    // Clone-only diagnostic equivalents of ordinary inverse-depth track
    // states and reprojection factors. They intentionally avoid allocating
    // Track objects so shadow experiments do not advance global track IDs.
    virtual bool add_shadow_track_state(
        double *inv_depth, Track *source_track);
    virtual bool add_shadow_reprojection(
        Frame *frame, size_t keypoint_index,
        Frame *reference_frame, size_t reference_keypoint_index,
        double *inv_depth);

    // Recovery-only visual observation with a fixed world landmark. The
    // residual uses the same bearing-tangent whitening and robust loss as
    // XRSLAM's ordinary reprojection factors.
    virtual void add_learned_world_reprojection(
        Frame *frame, const vector<3> &landmark_world,
        const vector<2> &observation_pixel);

    virtual void add_factor(ReprojectionErrorFactor *rpecost);
    virtual void add_factor(ReprojectionPriorFactor *rppcost);
    virtual void add_factor(RotationPriorFactor *ropcost);
    virtual void add_factor(PreIntegrationErrorFactor *piecost);
    virtual void add_factor(PreIntegrationPriorFactor *pipcost);
    virtual void add_factor(MarginalizationFactor *marcost);

    // Bind an existing marginalization cost to alternate Frame storage.
    // This is used by clone-only recovery diagnostics; the factor's stored
    // linearization point/information are read unchanged.
    virtual bool add_marginalization_factor_for_frames(
        MarginalizationFactor *marcost,
        const std::vector<Frame *> &frames);

    template <typename T> void put_factor(std::unique_ptr<T> &&factor) {
        add_factor(factor.get());
        manage_factor(std::move(factor));
    }

    virtual bool solve(bool verbose = false);

  protected:
    virtual void
    manage_factor(std::unique_ptr<ReprojectionErrorFactor> &&factor);
    virtual void
    manage_factor(std::unique_ptr<ReprojectionPriorFactor> &&factor);
    virtual void manage_factor(std::unique_ptr<RotationPriorFactor> &&factor);
    virtual void
    manage_factor(std::unique_ptr<PreIntegrationErrorFactor> &&factor);
    virtual void
    manage_factor(std::unique_ptr<PreIntegrationPriorFactor> &&factor);
    virtual void manage_factor(std::unique_ptr<MarginalizationFactor> &&factor);
    std::unique_ptr<SolverDetails> details;
};

} // namespace xrslam

#endif // XRSLAM_SOLVER_H
