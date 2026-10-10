#ifndef XRSLAM_RELOCALIZATION_AUTO_LOSS_H
#define XRSLAM_RELOCALIZATION_AUTO_LOSS_H

#include <cstddef>

namespace xrslam {

// Experimental consecutive-frame tracking-failure gate.
// The detector's input is the *unchanged* 0121g diagnostic suspect bit.
// It observes only successful tracker steps and is reset at session boundaries.
class RelocalizationAutoLossGate {
  public:
    static constexpr size_t grace_frames = 20;
    static constexpr size_t required_suspicious_frames = 5;

    bool observe(bool suspect) {
        ++observed_frames_;
        if (observed_frames_ <= grace_frames) {
            suspicious_run_ = 0;
            return false;
        }
        suspicious_run_ = suspect ? suspicious_run_ + 1 : 0;
        return suspicious_run_ >= required_suspicious_frames;
    }

    void reset() {
        observed_frames_ = 0;
        suspicious_run_ = 0;
    }

    size_t observed_frames() const { return observed_frames_; }
    size_t suspicious_run() const { return suspicious_run_; }

  private:
    size_t observed_frames_ = 0;
    size_t suspicious_run_ = 0;
};

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_AUTO_LOSS_H
