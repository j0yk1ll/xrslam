#ifndef XRSLAM_PC_DENSE_SEMANTIC_PLY_WRITER_H
#define XRSLAM_PC_DENSE_SEMANTIC_PLY_WRITER_H

#include <XRSLAM.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace xrslam_pc_dense {

inline std::array<unsigned, 3> cityscapes19_color(unsigned class_id) {
    static constexpr unsigned char palette[19][3] = {
        {128,64,128}, {244,35,232}, {70,70,70}, {102,102,156},
        {190,153,153}, {153,153,153}, {250,170,30}, {220,220,0},
        {107,142,35}, {152,251,152}, {70,130,180}, {220,20,60},
        {255,0,0}, {0,0,142}, {0,0,70}, {0,60,100},
        {0,80,100}, {0,0,230}, {119,11,32}
    };
    if (class_id >= 19) return {128,128,128};
    return {palette[class_id][0], palette[class_id][1], palette[class_id][2]};
}

inline bool write_ply_files(const std::string &prefix,
                            const std::vector<XRSLAMDenseSemanticPoint> &points,
                            std::vector<std::string> &paths, std::string &error,
                            double voxel_size_m, bool label_colors = false) {
    if (prefix.empty() || !std::isfinite(voxel_size_m) || voxel_size_m <= 0) {
        error = "invalid dense PLY path/voxel size";
        return false;
    }
    std::map<std::uint64_t, std::vector<XRSLAMDenseSemanticPoint>> by_session;
    for (const auto &p : points) {
        if (!p.session_id) { error = "zero session ID"; return false; }
        for (double value : p.xyz)
            if (!std::isfinite(value)) { error = "nonfinite XYZ"; return false; }
        by_session[p.session_id].push_back(p);
    }
    for (const auto &[session, samples] : by_session) {
        const auto path = prefix + "_session_" + std::to_string(session) + ".ply";
        std::filesystem::path outpath(path);
        if (!outpath.parent_path().empty())
            std::filesystem::create_directories(outpath.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out) {error = "cannot write " + path; return false;}
        out << "ply\nformat ascii 1.0\n";
        out << "comment coordinate_frame vio_session_local\n";
        out << "comment semantic_source PIDNet\n";
        out << "comment session_id " << session << "\n";
        out << "comment voxel_size_m " << std::setprecision(9) << voxel_size_m << "\n";
        out << "comment geometry depth_backprojected_surface_samples_not_tsdf\n";
        out << "comment colors " << (label_colors ? "cityscapes19_visualization_only" : "canonical_rgb") << "\n";
        out << "element vertex " << samples.size() << "\n";
        out << "property double x\nproperty double y\nproperty double z\n";
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
        out << "property uchar semantic_class\nproperty float consensus\n";
        out << "property uint observation_count\nproperty uchar stable\n";
        out << "property int voxel_ix\nproperty int voxel_iy\nproperty int voxel_iz\n";
        out << "end_header\n";
        out << std::setprecision(17);
        for (const auto &p : samples) {
            const auto display = label_colors ? cityscapes19_color(p.semantic_class) :
                std::array<unsigned,3>{p.rgb[0], p.rgb[1], p.rgb[2]};
            out << p.xyz[0] << ' ' << p.xyz[1] << ' ' << p.xyz[2] << ' '
                << display[0] << ' ' << display[1] << ' '
                << display[2] << ' ' << unsigned(p.semantic_class)
                << ' ' << p.consensus << ' ' << p.observation_count << ' '
                << unsigned(p.stable) << ' ' << p.voxel_ix << ' '
                << p.voxel_iy << ' ' << p.voxel_iz << '\n';
        }
        if (!out.good()) { error = "write failed: " + path; return false; }
        paths.push_back(path);
    }
    return true;
}

} // namespace xrslam_pc_dense
#endif
