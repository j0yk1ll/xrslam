#ifndef XRSLAM_PERSISTENT_RELOCALIZATION_MAP_IO_H
#define XRSLAM_PERSISTENT_RELOCALIZATION_MAP_IO_H

#include <xrslam/core/persistent_relocalization_map.h>
#include <string>

namespace xrslam {

// Versioned, checksum-protected, crash-replaceable localization reference
// archive. This is NOT a PLY or a navigation occupancy map.
// Both operations return false and populate error on IO/format errors.
// No SLAM mutable state is accessed by these functions; caller must serialize
// at a safe ownership boundary (after stopping worker threads).
bool save_relocalization_map(const PersistentRelocalizationMap &map,
                             const std::string &path,
                             const std::string &map_identity,
                             std::string *error = nullptr);
bool load_relocalization_map(PersistentRelocalizationMap &map,
                             const std::string &path,
                             const std::string &map_identity,
                             std::string *error = nullptr);

} // namespace xrslam
#endif
