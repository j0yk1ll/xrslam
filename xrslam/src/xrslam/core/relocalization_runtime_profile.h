#ifndef XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H
#define XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace xrslam {

// Transitional high-level opt-in: the original individual feature gates
// continue to work when this variable is absent. The v1 profile enables
// historical retrieval, ORB/PnP, graph-corrected global output and automatic
// loss/recovery together. Does not enable estimator-mutating recovery commits.
inline bool relocalization_v1_profile_enabled() {
    const char *profile = std::getenv("XRSLAM_RELOCALIZATION_PROFILE");
    if (!profile || profile[0] == '\0' || std::string(profile) == "off")
        return false;
    if (std::string(profile) == "v1")
        return true;
    throw std::runtime_error(
        "XRSLAM_RELOCALIZATION_PROFILE must be 'off' or 'v1'");
}

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H
