#ifndef XRSLAM_PC_SEMANTIC_PLY_WRITER_H
#define XRSLAM_PC_SEMANTIC_PLY_WRITER_H

// ASCII PLY export for session-scoped, value-owned read-only semantic points.
// No SLAM, Torch, or OpenCV access occurs in this writer.
#include "XRSLAM.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace xrslam_pc_semantic {

struct OutputAlignment {
    double yaw_deg = 0.0;
    std::array<double, 3> translation{{0.0, 0.0, 0.0}};
};

using OutputAlignments = std::map<std::uint64_t, OutputAlignment>;

// Explicit alignment, not an inferred XRSLAM loop-closure correction.
// Whitespace-separated ASCII rows: session_id yaw_deg tx ty tz.
// Every exported session must have an entry when alignment is requested.
inline bool read_alignments(const std::string &path, OutputAlignments &out,
                            std::string &error) {
    std::ifstream file(path);
    if (!file) {
        error = "cannot open alignment file: " + path;
        return false;
    }
    OutputAlignments parsed;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        const auto non_ws = line.find_first_not_of(" \t\r\n");
        if (non_ws == std::string::npos || line[non_ws] == '#') continue;
        std::istringstream row(line);
        row.imbue(std::locale::classic());
        std::uint64_t session = 0;
        OutputAlignment transform;
        std::string extra;
        if (!(row >> session >> transform.yaw_deg >>
              transform.translation[0] >> transform.translation[1] >>
              transform.translation[2]) || (row >> extra) || session == 0 ||
            !std::isfinite(transform.yaw_deg) ||
            !std::all_of(transform.translation.begin(), transform.translation.end(),
                         [](double v) { return std::isfinite(v); }) ||
            !parsed.emplace(session, transform).second) {
            error = "invalid or duplicate alignment at line " +
                    std::to_string(line_no);
            return false;
        }
    }
    if (!file.eof() && file.fail()) {
        error = "I/O error reading alignments";
        return false;
    }
    out = std::move(parsed);
    return true;
}

inline std::array<double, 3> output_position(
    const XRSLAMSemanticMapPoint &point, const OutputAlignment &alignment) {
    constexpr double pi = 3.14159265358979323846264338327950288;
    const double a = alignment.yaw_deg * pi / 180.0;
    const double c = std::cos(a), s = std::sin(a);
    return {{c * point.xyz[0] - s * point.xyz[1] + alignment.translation[0],
             s * point.xyz[0] + c * point.xyz[1] + alignment.translation[1],
             point.xyz[2] + alignment.translation[2]}};
}

inline std::array<unsigned, 3> label_rgb(unsigned class_id) {
    // Standard Cityscapes-19 display palette. Colors are visualization only.
    static constexpr unsigned char colors[19][3] = {
        {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
        {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
        {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
        {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100},
        {0, 80, 100}, {0, 0, 230}, {119, 11, 32}};
    if (class_id < 19)
        return {{colors[class_id][0], colors[class_id][1], colors[class_id][2]}};
    return {{(class_id * 53u + 31u) & 255u,
             (class_id * 97u + 73u) & 255u,
             (class_id * 193u + 11u) & 255u}};
}

// Writes <prefix>_session_<id>.ply for each distinct VIO session. With
// explicit per-session alignment, writes <prefix>_aligned_session_<id>.ply.
// Never conflates session-local frames or silently invents an alignment.
inline bool write_ply_files(const std::string &prefix,
                            const std::vector<XRSLAMSemanticMapPoint> &points,
                            const OutputAlignments *alignments,
                            std::vector<std::string> &written,
                            std::string &error) {
    if (prefix.empty()) {
        error = "empty PLY prefix";
        return false;
    }
    std::map<std::uint64_t, std::vector<XRSLAMSemanticMapPoint>> sessions;
    for (const auto &point : points) {
        if (point.session_id == 0 || !std::isfinite(point.xyz[0]) ||
            !std::isfinite(point.xyz[1]) || !std::isfinite(point.xyz[2]) ||
            !std::isfinite(point.consensus) ||
            !std::isfinite(point.mean_pixel_confidence)) {
            error = "nonfinite or invalid semantic point";
            return false;
        }
        sessions[point.session_id].push_back(point);
    }
    if (alignments) {
        for (const auto &entry : sessions) {
            if (alignments->find(entry.first) == alignments->end()) {
                error = "missing explicit output alignment for session " +
                        std::to_string(entry.first);
                return false;
            }
        }
    }
    // Prevalidate all sessions before creating files.
    for (const auto &entry : sessions)
        for (const auto &p : entry.second)
            if (p.observation_count > std::numeric_limits<std::uint32_t>::max() ||
                p.winning_label_views > std::numeric_limits<std::uint32_t>::max()) {
                error = "observation/view count exceeds PLY uint32 range";
                return false;
            }

    for (auto &entry : sessions) {
        const std::uint64_t session = entry.first;
        auto &group = entry.second;
        std::sort(group.begin(), group.end(), [](const auto &a, const auto &b) {
            return a.track_id < b.track_id;
        });
        const std::string path = prefix + (alignments ? "_aligned_session_" : "_session_") +
                                 std::to_string(session) + ".ply";
        std::ofstream file(path, std::ios::out | std::ios::trunc);
        file.imbue(std::locale::classic());
        if (!file) {
            error = "cannot write " + path;
            return false;
        }
        file << "ply\nformat ascii 1.0\n"
             << "comment generated_by XRSLAM_0122d\n"
             << "comment session_id " << session << "\n"
             << "comment coordinate_frame "
             << (alignments ? "explicit_output_alignment" : "vio_session_local") << "\n"
             << "comment semantic_colors cityscapes19_visualization_only\n";
        if (alignments) {
            const auto &a = alignments->at(session);
            file << std::setprecision(17)
                 << "comment alignment_yaw_deg " << a.yaw_deg << "\n"
                 << "comment alignment_translation " << a.translation[0] << ' '
                 << a.translation[1] << ' ' << a.translation[2] << "\n";
        }
        file << "element vertex " << group.size() << "\n"
             << "property double x\nproperty double y\nproperty double z\n"
             << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             << "property uchar semantic_class\n"
             << "property float consensus\nproperty float mean_pixel_confidence\n"
             << "property uint observation_count\nproperty uint winning_label_views\n"
             << "property uchar stable\n"
             << "property uint track_id_lo\nproperty uint track_id_hi\n"
             << "property uint latest_frame_id_lo\nproperty uint latest_frame_id_hi\n"
             << "end_header\n";
        file << std::setprecision(17);
        for (const auto &p : group) {
            const auto rgb = label_rgb(p.semantic_class);
            const std::array<double, 3> xyz = alignments ?
                output_position(p, alignments->at(session)) :
                std::array<double, 3>{{p.xyz[0], p.xyz[1], p.xyz[2]}};
            file << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << ' '
                 << rgb[0] << ' ' << rgb[1] << ' ' << rgb[2] << ' '
                 << static_cast<unsigned>(p.semantic_class) << ' '
                 << p.consensus << ' ' << p.mean_pixel_confidence << ' '
                 << p.observation_count << ' ' << p.winning_label_views << ' '
                 << static_cast<unsigned>(p.stable) << ' '
                 << static_cast<std::uint32_t>(p.track_id) << ' '
                 << static_cast<std::uint32_t>(p.track_id >> 32) << ' '
                 << static_cast<std::uint32_t>(p.latest_frame_id) << ' '
                 << static_cast<std::uint32_t>(p.latest_frame_id >> 32) << '\n';
        }
        file.close();
        if (!file) {
            error = "I/O failure writing " + path;
            return false;
        }
        written.push_back(path);
    }
    return true;
}

} // namespace xrslam_pc_semantic

#endif // XRSLAM_PC_SEMANTIC_PLY_WRITER_H
