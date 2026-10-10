# XRSLAM Engineering Handoff — Loop Closure, Global Output, and Relocalization v1

**Prepared:** 2026-10-10  
**Repository:** `j0yk1ll/xrslam`; development checkout described as `/home/kevin/Code/xrslam` on Arch Linux.  
**Primary objective:** Preserve the validated VIO/loop-closure baseline while moving from experimental relocalization to a maintainable system. Semantic mapping comes *after* code cleanup and regression tests.

> **Evidence boundary.** This document combines reported local build/replay results, preceding implementation notes, and an audit of the public GitHub tree. The user's local working tree, uncommitted changes, final Git SHA, and compiled binary were **not** independently inspected in this handoff. Re-check them before changing code. “PASS” below is scoped to the named replay and checker, not a production-wide guarantee.

## 1. Quick-start for a new LLM or engineer

1. Read this entire handoff. Inspect `git status --short`, `git rev-parse HEAD`, `git diff --stat`, relevant headers, and the CMake cache before proposing changes. The latest public `main` may differ from the local checkout.
2. **Preserve** the frozen **0112h 4-DoF graph** (yaw + xyz) and the **0120a/b causal, output-only global pose** behavior. Do not feed the graph correction into the local VIO optimizer.
3. The validated relocalization path is built on **0121a–0121h**, with **0121f's loss-trigger timing fix** and **0121j's test-only negative control** subsequently applied. Some experiment scripts are in `~/Downloads`, not the repository. Do not assume all historical patches apply to current `main`.
4. The previous acceptance suite includes: normal no-loss V2_03, controlled-loss recovery, synthetic camera blackout recovery, and corrupted-3D-association rejection. Retain the exact logs and TUM trajectories before cleanup.
5. Do not equate *a verified PnP hypothesis* with *resumed VIO*. Accepted recovery must create a valid new estimator state, show subsequent mapped visual tracks, and preserve the right map/global coordinate relationship.
6. Deliver incremental, auditable changes; always run `git diff --check`, build the existing `build-eigenplaces-cpu` target, and replay the relevant tests. No silent change to thresholds, frame conventions, camera/IMU extrinsics, graph geometry, or baseline output semantics.

## 2. System architecture and invariants

| Area | Implementation / invariant |
|---|---|
| Local VIO | Feature tracker, `Initializer`, `SlidingWindowTracker`, sliding-window bundle adjustment and IMU preintegration. Local estimates remain independent of loop-closure global corrections. |
| Place representation | EigenPlaces ResNet50/512D descriptors; place database (USearch where enabled); ORB local descriptors for geometric matching. XFeat/LighterGlue is a retired experimental backend, **not** the accepted recovery pipeline. |
| Reference persistence | `PersistentRelocalizationMap` owns historical `KeyframeArchive` + `PlaceKeyframeStore` per lost-tracking session; reference sessions have distinct local coordinate systems. |
| Lost-frame retrieval | In initializer/recovery phase, retrieve historical places, match ORB local descriptors to archived 2D–3D observations, and verify camera pose using PnP/RANSAC and reprojection checks. |
| Temporal confirmation | `RelocalizationConsensus` requires two independent geometrically verified current frames in the **same** historical session, with bounded separation and pose disagreement. |
| Recovery handoff | `relocalization_bootstrap.h` seeds an **isolated** new sliding-window map from verified historical 3D landmarks, transfers KLT/IMU measurements to the next frame, and accepts only after agreement with independent PnP. |
| Tracking health | `relocalization_tracking_health.h`, `relocalization_auto_loss.h`; opt-in detector uses 20-frame grace period and 5 consecutive suspicious samples. Current thresholds derive primarily from one healthy EuRoC sequence; **not production-calibrated**. |
| Global pose | 0120b causal output applies the graph's yaw + translation to body pose, without altering raw VIO; 0121f retains the old-session alignment and restores it only for verified same-session recovery. |
| Loop closure | **0112h frozen**. Leave optimizer, loop weights, graph thresholds, and 4-DoF transform path unchanged unless an independently reviewed change demands it. |

**Important implementation cautions:** The historical session's `local` coordinates are not a universal world frame. A pose seed and archived landmark coordinates must refer to the same session. The recovered map must preserve camera↔body extrinsics, metric scale, gravity, velocity/bias validity, 2D–3D track IDs, and timestamps. A copied pose alone is not a relocalization. Reference descriptors persist on longer-lived `XRSLAM::Detail` structures; verify lifecycle and bounded ownership before long-duration deployment.

## 3. Acceptance history and quantitative evidence

### 0112h–0120b: loop closure and corrected output

- 0112h 4-DoF yaw + xyz place graph was accepted and frozen.
- 0120a published thread-safe global drift snapshots.
- 0120b exported `XRSLAM_RESULT_GLOBAL_BODY_POSE` and player `--global-tum`. Successful tests reported local/raw vs corrected APE RMSE (m): V2_03 **0.248064218 → 0.166484810**; V2_02 **0.115562420 → 0.114858207**; MH04 **0.285255827 → 0.284369233**; MH05 **0.683513355 → 0.681716841**. V1_03 had no closures and byte-identical local/global output. Prior validations also reported raw VIO unchanged against available baselines.

### 0121a–0121j: persistent-map recovery

| Milestone | What actually passed |
|---|---|
| 0121a | Preserved 237 archived keyframes and 237 place entries after loss; build succeeded. |
| 0121b | After forced loss, 253 probe summaries, 3,947 candidate PnP solutions, 23 geometrically verified historical hypotheses. Example: frame 660 ↔ reference 656, 29/47 inliers, RMSE 2.216 px. |
| 0121c | Independent confirmed poses at current frames **660→665**, 0.250 s apart, translation disagreement **0.463 m**, rotation **7.387°**. At that time ordinary VIO reinitialization was still zero. |
| 0121d | Diagnostics showed why the ordinary initializer failed: 1,250 SfM attempts, max 43 shared feature matches vs required 50. Do not reduce the 50-match threshold merely to force success. |
| 0121e | Archived 3D map bootstrap transferred 22 tracks, accepted handoff with 0.034358 m/3.09° disagreement, created new VIO session, achieved 20/20 good first steps and **1,256** post-recovery steps in controlled replay. |
| 0121f + timing fix | Preserved `session=1` alignment, yaw **0.012516830°**, translation **(0.117741731, −0.013021059, 0.064437394) m**; **1,256** post-recovery pose transformations matched it to 1.47e−9 m / 2.41e−6° maximum residual. This is **transform consistency**, not proof of zero physical jump. |
| 0121g | Passive healthy V2_03: 1,813 frames; nine suspicious, longest streak two; mapped median 48; reprojection median 1.9661 px; no loss. |
| 0121h | Automatic detector enabled: 1,813 healthy samples, zero false resets in that replay. |
| 0121i | Synthetic 16-frame camera blackout (indices 639–654), IMU unchanged; one automatic loss, zero forced loss, one confirmed seed/handoff/new VIO, **481** post-recovery steps and 20/20 good early steps. |
| 0121j | False-correspondence control: 639 deliberately corrupted historical sets, 521 candidate logs, zero verified candidates, zero seeds, zero handoffs, zero erroneous global alignment restorations. |

**Scope of “v1 PASS”:** a reproducible EuRoC V2_03 *experimental* recovery under synthetic degradation and one controlled incorrect-association condition. It does **not** guarantee robustness to actual camera occlusion, repeated failures, cross-session mapping, sustained high dynamic motion, photometric changes, severe IMU drift, repetitive textures, or naturally wrong place recognition.

## 4. Reproduce and retain the baseline

Checkout/build:

```bash
cd /home/kevin/Code/xrslam
git rev-parse HEAD
git status --short
git diff --check
cmake --build build-eigenplaces-cpu -j"$(nproc)"
```

Existing tests (scripts were delivered into `~/Downloads` and may need copying into a versioned `tests/relocalization` directory):

```bash
# Healthy replay with automatic gate enabled: zero false resets expected.
bash ~/Downloads/0121h_auto_loss_baseline_replay.sh

# Synthetic 16-frame visual dropout. No forced-loss hook.
bash ~/Downloads/0121i_visual_dropout_replay.sh

# Negative PnP 2D–3D association control: rejects all corrupted matches.
bash ~/Downloads/0121j_false_match_negative_replay.sh

# Historical global output continuity; provide existing baseline trajectories.
python3 ~/Downloads/0121f_global_continuity_check.py \
  --prefix /tmp/v2_03_0121f_timing_fix \
  --baseline-local /tmp/v2_03_0121e_local.tum
```

**Caution:** cleanup removes experimental test hooks, so do **not** rerun the old forced-loss or corrupt-association scripts on a stripped production target and interpret missing injection as a failed recovery. Preserve a separate test fixture or compile-time test target to reproduce them. Make reproducibility a first-class part of the cleanup.

**Record together:** exact Git SHA, `git status --porcelain`, CMake cache option summaries (without credentials), model SHA256 and location, dataset checksum/manifest, command line and effective env, full stderr/stdout, raw and global TUM, checker version/result, timestamps, system/CPU, replay throughput, peak RSS.

## 5. Present flags: classify BEFORE deleting

**Known retired XFeat/LighterGlue items** (public tree still contains them; local state must be verified): `xrslam-extra-xfeat` target, optional `onnxruntime` find/link, `XRSLAM_HAS_XFEAT_ONNX`, source/header `xfeat_local_feature_backend.*`, and `tools/*xfeat*` / `tools/*lighterglue*` benchmarks and exporters. The `XRSLAM_XFEAT_*` and `XRSLAM_LIGHTERGLUE_*` environment variables in that backend are obsolete when the backend is removed. Some old `Frame` learned recovery methods/logs remain; audit their reachability before removing them.

**Test/debug controls to relocate out of the shipping binary:** `XRSLAM_TEST_RELOCALIZATION_FORCE_LOSS_ONCE`, `XRSLAM_TEST_RELOCALIZATION_CORRUPT_PNP`, `XRSLAM_RELOCALIZATION_INIT_TRACE`, `XRSLAM_RELOCALIZATION_HEALTH_SHADOW`, 0121b high-volume probe/candidate logging, and experimental event CSV/diagnostic tracing as appropriate. Keep failure-injection in a test-only executable or compile-time guard if repeatability is needed.

**Currently behavior-bearing flags — do not blindly delete:** `XRSLAM_RELOCALIZATION_AUTO_LOSS`, `XRSLAM_RELOCALIZATION_HANDOFF`, `XRSLAM_RELOCALIZATION_PROBE`, `XRSLAM_PLACE_RETRIEVAL_SHADOW`, `XRSLAM_PLACE_DESCRIPTOR_SHADOW`, `XRSLAM_PLACE_GRAPH_4DOF_SHADOW`, `XRSLAM_ORB_PNP_SHADOW`, related ORB/archive/local-descriptor flags, `XRSLAM_EIGENPLACES_MODEL` and cache path, and optional USearch/EigenPlaces CMake switches. Several names include “SHADOW” but gate real descriptor collection, retrieval, graph operation, or recovery. Their removal without replacement could silently disable the accepted pipeline. Migrate them to a documented coherent `place_recognition/relocalization` configuration, with validation and explicit defaults; do not change their meaning while merely renaming.

## 6. Prioritized future improvements (recommended order)

### P0 — Code cleanup, reproducibility, and regression protection

1. **Archive the accepted state** in a named commit/tag and a replay manifest, then implement the conservative cleanup described in the accompanying cleanup tool. Do not delete experimental logs/checkers until they are captured in a versioned regression suite.
2. **Delete decommissioned XFeat/LighterGlue build/dependency paths** and obsolete standalone tools, then audit `Frame::recover_keypoints`, `recovery_pose_cache`, learned factor code and backend interface methods for unreachable code. Measure compile time, binary size and behavior, and remove dead code in a second reviewed patch.
3. **Separate production configuration from test injection:** migrate runtime flags to validated typed config, remove test-only branches from shipping builds, and keep testing functionality under `XRSLAM_BUILD_TESTS` or a dedicated fixture interface. Log effective configuration once per run.
4. **Reduce log volume and contention:** make PnP diagnostic detail opt-in, rate-limit logs, and move learned descriptor inference/PnP outside synchronized feature-map locks. Current prototype probes while holding a feature-map lock; evaluate frontend latency and contention.
5. Create CI tiers: cheap compile/static checks; small test fixture; full V2_03 healthy/blackout/negative; other EuRoC sequences; performance regression. Include sanitizers (ASan/UBSan/TSan where practicable).

### P1 — Real-world relocalization robustness

1. **Calibrate automatic loss detection** on multiple EuRoC MAV sequences plus true occlusion, motion blur, overexposure, low texture, abrupt rotation, and lens covering. Evaluate ROC, false losses per hour, detection latency, recovery time, unrecovered duration; five suspicious frames and 20 grace frames are provisional.
2. **Repeated/multi-session recovery:** track two or more natural failures, archive session transforms, choose the correct session unambiguously, avoid stale descriptor IDs and unbounded memory, and recover across varied coordinate origins.
3. **False-place rejection beyond scrambled correspondences:** naturally similar corridors and repeated surfaces; changed illumination/season; wrong submap with internally consistent PnP; large database with competing candidates. Validate independent temporal consistency, IMU rotation/velocity prior, scale and gravity, and cheirality/multi-hypothesis policies.
4. **Relocalization quality metrics:** time from loss to first trustworthy pose, map overlap required, inlier ratio, RMSE, reprojection heatmaps, pose covariance / uncertainty, end-to-end ATE/RPE, physical position and orientation jump at handoff, and post-handoff IMU bias/velocity transients.
5. **Fallback when no map or no match:** define deterministic `LOST → SEARCHING → BOOTSTRAP → TRACKING` transitions, timeout, restart-new-submap policy, error reporting, stale trajectory behavior, and subsequent submap alignment.

### P2 — Map, performance, and lifecycle hardening

1. Bound the persistent archives, place database, and image lifetime with eviction policies that retain spatial coverage/landmark observability; add multi-session serialization/versioning and recovery from process restart.
2. Move inference/retrieval to a budgeted worker queue; cap hypotheses and feature counts, cancel stale jobs, avoid long frontend locks, and measure worst-case frame latency on CPU and target mobile devices.
3. Make tracker/initializer ownership explicit (move-only snapshots, stable landmark IDs, immutable geometry and thread-safe readers), clarify map↔world frame semantics and stale global alignment transitions.
4. Ensure global-output correction handling is mathematically and causally consistent after new loop closures *post-recovery*, not only replay of a frozen correction. Test global pose continuity and local-pose invariant across multiple graph updates.
5. Add numeric safety gates for invalid intrinsics, nonfinite poses, insufficient positive depth, degenerate PnP configurations, stale keyframes, timestamp regression, unrealistic acceleration or angular-rate jumps.

### P3 — Further SLAM quality and research directions

- Improve robust visual front-end under drastic appearance changes without reviving obsolete XFeat/LighterGlue experiments by default. Benchmark any proposed replacement against ORB+EigenPlaces on accuracy, memory, wall time, warm-up and reproducibility.
- Compare local bundle adjustment and IMU prior weighting over recovery transients; prefer physically interpretable sensor constraints over ad hoc trajectory smoothing.
- Investigate candidate ranking, bag-of-words/HNSW tuning, vocabulary maintenance, discriminative landmarks and diversity-aware keyframe selection. Retain frozen 0112h as baseline.
- Explore uncertainty-aware position/yaw/global alignment filtering with explicit policy for discontinuities and output quality flags.

### P4 — Semantic mapping, only after P0 is green

Start with a **read-only semantic observation archive** keyed by immutable frame ID, timestamp, map-session ID, landmark/track ID, camera calibration and global body/camera pose source. Preserve per-detection confidence and uncertainty and allow old-session re-association after relocalization. Never initially mutate raw VIO, IMU estimation, PnP acceptance, or loop graph based on neural semantics. Define privacy, data retention, offline evaluation and dynamic-object policies. Only consider semantic-informed SLAM after semantics-only evaluation passes and geometry invariance is proven.

## 7. Proposed completion criteria for maintainable relocalization v1

- No references or linked targets for the retired XFeat/LighterGlue backend in the release configuration. No obsolete build artifacts masquerading as active dependencies.
- A documented enable path for the actual place-recognition/relocalization stack; coherent defaults and startup checks for model/index availability.
- No test-only forced-loss or PnP corruption env flags in the shipping binary; testing injection remains reproducible under a dedicated test path.
- Healthy baseline unchanged and auto loss produces zero false resets on previously accepted healthy replay; black-frame test still triggers automatic recovery; incorrect-match control still rejects false handoffs under test fixture.
- Raw local trajectory unchanged within appropriate controls, and corrected output matches the validated graph transform; no new physical handoff discontinuity introduced.
- Code review and `git diff --check` clean; build/replay artifact manifest, commit and rollback instructions checked in.

## 8. The first task for the next LLM

> **Do not start semantic mapping.** Inspect the local tree and apply a minimal cleanup of orphaned XFeat/LighterGlue build targets and the experimental test-only relocalization hooks. Provide a readable `git diff` and confirm no active recovery path uses the removed backend. Preserve feature-bearing environment controls until replaced with structured config, then run build + healthy + automatic-blackout + geometric negative tests. If any test requires the removed test hooks, move that injection into a dedicated test-only build, or run and archive it before removing production hooks. Do not claim production-ready relocalization from the existing EuRoC-only evidence.

---

*Sources: local test outputs supplied in this work session; patches/readmes named `0120*`, `0121a`…`0121j`; and repository source files inspected on public GitHub on 2026-10-10. Source of truth for future changes must be the actual local Git tree at the time of work.*
