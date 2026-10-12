#include <xrslam/semantic_landmark_map.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "semantic landmark fusion FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    xrslam::SemanticLandmarkMap map(19);
    const auto vote = [](std::uint64_t session, std::uint64_t frame,
                         std::size_t track, std::uint8_t cls,
                         float conf, double x) {
        xrslam::SemanticLandmarkVote v;
        v.session_id = session; v.frame_id = frame; v.track_id = track;
        v.label = cls; v.confidence = conf;
        v.position_session_world = {x, 2.0, 3.0};
        return v;
    };
    require(!map.observe(vote(1, 1, 3, 19, 0.9f, 1.0)), "label outside taxonomy accepted");
    require(!map.observe(vote(1, 1, 3, 2, 0.1f, 1.0)), "low-confidence vote accepted");
    require(map.observe(vote(1, 101, 3, 2, 0.8f, 1.0)), "first vote rejected");
    require(!map.observe(vote(1, 101, 3, 2, 0.9f, 9.0)), "duplicate keyframe vote accepted");
    require(map.size() == 1 && map.stable_count() == 0, "premature stable landmark");
    require(map.observe(vote(1, 102, 3, 2, 0.9f, 3.0)), "second vote rejected");
    auto fused = map.find(1, 3);
    require(bool(fused), "missing fused landmark");
    require(fused->stable && fused->observation_count == 2, "two-view label not stable");
    require(fused->winning_label == 2 && fused->winning_label_views == 2, "incorrect class consensus");
    require(fused->consensus_fraction > 0.999, "incorrect consensus fraction");
    require(fused->position_session_world[0] > 2.0 &&
            fused->position_session_world[0] < 2.1, "incorrect weighted 3D position");
    require(map.observe(vote(1, 103, 3, 7, 1.0f, 5.0)), "contradictory vote rejected");
    fused = map.find(1, 3);
    require(bool(fused) && !fused->stable && fused->winning_label == 2,
            "contradictory label did not downgrade stability");
    require(map.stable_count() == 0, "stable count not downgraded");
    // Session 2 can reuse track 3 without inheriting session 1's evidence.
    require(map.observe(vote(2, 101, 3, 5, 0.9f, 50.0)), "new session vote rejected");
    require(map.observe(vote(2, 102, 3, 5, 0.9f, 52.0)), "second session view rejected");
    auto s2 = map.find(2, 3);
    require(bool(s2) && s2->stable && s2->winning_label == 5, "session labels contaminated");
    require(std::abs(s2->position_session_world[0] - 51.0) < 1e-4,
            "incorrect second-session position");
    require(map.size() == 2 && map.stable_count() == 1, "incorrect landmark counts");
    require(map.snapshot(1, true).empty(), "unstable landmark exported as stable");
    require(map.snapshot(2, true).size() == 1, "stable landmark missing from filtered snapshot");
    require(map.snapshot().size() == 2, "missing full map entries");
    require(!map.find(3, 3), "unknown session lookup succeeded");
    std::puts("semantic landmark fusion smoke PASS");
    return EXIT_SUCCESS;
}
