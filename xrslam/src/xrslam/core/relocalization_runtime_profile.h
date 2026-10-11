#ifndef XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H
#define XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace xrslam {

// Relocalization v1 is explicitly enabled by one runtime profile.
// When unset or off, the v1 retrieval/graph/recovery gates stay off.
// The v1 profile enables
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

// High-volume, observation-level logs are off in normal operation.
// Failure transitions and accepted/rejected recovery handoffs remain logged.
// This switch controls output only; it never bypasses health or PnP checks.
inline bool relocalization_verbose_diagnostics() {
    const char *setting = std::getenv("XRSLAM_DIAGNOSTICS_VERBOSE");
    return setting && std::string(setting) == "1";
}

} // namespace xrslam
#endif // XRSLAM_RELOCALIZATION_RUNTIME_PROFILE_H
