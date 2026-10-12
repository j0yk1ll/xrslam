#include <xrslam/dense_semantic_voxel_map.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void require(bool v, const char *msg) {if (!v) throw std::runtime_error(msg);}
int main() {
    using namespace xrslam;
    DenseSemanticVoxelMap m(19, 0.05, 100);
    DenseSemanticSample s;
    s.session_id = 1; s.frame_id = 1; s.xyz = {1.01, 2.0, 3.0};
    s.label = 7; s.confidence = 0.9f; s.rgb = {55, 66, 77};
    require(m.observe(s), "first observation");
    require(!m.observe(s), "duplicate frame/voxel must not reweight");
    s.frame_id = 2; s.xyz[0] = 1.02;
    require(m.observe(s), "second keyframe");
    auto points = m.snapshot(true);
    require(points.size() == 1, "stable consensus after two agreeing frames");
    require(points[0].semantic_class == 7 && points[0].observation_count == 2,
            "label and observation count");
    s.frame_id = 3; s.label = 9; s.confidence = 0.99f;
    require(m.observe(s), "opposing evidence");
    require(m.snapshot(true).empty(), "conflict destabilizes a voxel");
    s.session_id = 2; s.frame_id = 1; s.label = 7;
    require(m.observe(s), "separate VIO session");
    require(m.snapshot().size() == 2, "session isolation");
    s.session_id = 1; s.frame_id = 4; s.confidence = 0.20f;
    require(!m.observe(s), "confidence gate");
    const auto world = DenseSemanticVoxelMap::camera_to_session_world(
        {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)}, {1, 2, 3}, {1, 0, 2});
    require(std::abs(world[0] - 1) < 1e-10 &&
            std::abs(world[1] - 3) < 1e-10 &&
            std::abs(world[2] - 5) < 1e-10, "camera -> session yaw rotation");
    DenseSemanticVoxelMap bounded(19, 0.05, 1);
    s.session_id=1; s.frame_id=1; s.confidence=0.9; s.xyz={0,0,1};
    require(bounded.observe(s), "bounded first");
    s.xyz={1,0,1};
    require(!bounded.observe(s) && bounded.capacity_rejections()==1,
            "voxel capacity bounded");
    std::cout << "dense semantic voxel smoke PASS\n";
}
