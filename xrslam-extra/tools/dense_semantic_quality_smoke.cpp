#include <xrslam/dense_semantic_depth_filter.h>
#include <xrslam/dense_semantic_voxel_map.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool ok, const char *description) {
    if (!ok) throw std::runtime_error(description);
}

int main() {
    using namespace xrslam;
    SemanticMetricDepthBuffer depth;
    depth.width = 5; depth.height = 5; depth.meters.assign(25, 2.0f);
    DenseSemanticQualityOptions q;
    q.enabled = true;
    q.validate();
    check(dense_semantic_depth_is_consistent(depth, 2, 2, q), "smooth depth");
    depth.meters[2*5+3] = 2.5f;
    check(!dense_semantic_depth_is_consistent(depth, 2, 2, q), "step edge");
    depth.meters[2*5+3] = std::numeric_limits<float>::quiet_NaN();
    check(!dense_semantic_depth_is_consistent(depth, 2, 2, q), "missing neighbor");
    q.reject_missing_neighbors = false;
    check(dense_semantic_depth_is_consistent(depth, 2, 2, q), "hole allowed by option");
    check(dense_semantic_depth_is_consistent(depth, 0, 0, q), "boundary allowed by option");
    q.reject_missing_neighbors = true;
    check(!dense_semantic_depth_is_consistent(depth, 0, 0, q), "boundary closed");
    q.depth_edge_abs_m = -1;
    bool threw = false;
    try { q.validate(); } catch (const std::invalid_argument &) { threw=true; }
    check(threw, "invalid config rejected");

    DenseSemanticVoxelMap vox(19, 0.05, 100);
    DenseSemanticSample s;
    s.session_id=1; s.label=3; s.confidence=0.9f; s.rgb={10,20,30};
    const auto add = [&](int x, int y, int frame) {
        s.xyz = {0.025 + x*0.05, 0.025 + y*0.05, 2.025};
        s.frame_id=frame;
        check(vox.observe(s), "unique per-voxel-frame sample");
    };
    add(0,0,1); add(1,0,1); add(0,1,1); add(5,5,1);
    check(vox.snapshot().size()==4, "0123a baseline unchanged");
    check(vox.snapshot_filtered(false, 1, 2).size()==3,
          "isolated cell removed without deleting neighbor-supported cells");
    check(vox.snapshot_filtered(false, 2, 0).empty(), "single-frame cells removed");
    add(0,0,2); add(1,0,2); add(0,1,2);
    check(vox.snapshot_filtered(false, 2, 2).size()==3,
          "multi-view plus neighbor support");
    check(vox.snapshot_filtered(true, 2, 2).size()==3,
          "semantic stability unchanged");
    s.session_id = 2; s.frame_id=1;
    s.xyz={0.025,0.025,2.025};
    check(vox.observe(s),"second session geometry");
    check(vox.snapshot_filtered(false, 2, 2).size()==3,
          "no cross-session neighborhood support");
    check(vox.snapshot_filtered(false, 1, 0).size()==5,
          "unfiltered map retains all session data");
    std::cout << "dense semantic quality smoke PASS\n";
}
