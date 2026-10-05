#!/usr/bin/env python3
"""
Offline XFeat learned-landmark -> LighterGlue -> PnP viability benchmark.

This intentionally mirrors the useful identity pattern from the user's prior
ORB-SLAM3/XFeat system:

    native XFeat reference feature index -> persistent 3D landmark
    LighterGlue source index             -> same 3D landmark
    LighterGlue target index             -> current-frame native XFeat pixel
    (landmark XYZ, current pixel)         -> PnP RANSAC

No XFeat-to-KLT spatial snapping is used.

The learned landmark table is built offline from one or more earlier support
anchors using XRSLAM's estimated camera poses. Support/reference matches are
validated against the known relative pose, triangulated, checked for
cheirality/reprojection/parallax, and keyed by the *exact reference XFeat
feature index*. If multiple support anchors triangulate the same reference
feature, the geometrically strongest candidate is retained.

The reference->current LighterGlue source set contains ONLY reference features
that own a valid learned 3D landmark, capped to the highest-ranked source
features (the official XFeat sparse output is score sorted). There is no
reference->current Essential-matrix gate: PnP is the geometric verifier.

This is a behavior-neutral offline viability test. It does not modify XRSLAM.
"""

import argparse
import bisect
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ATTEMPT_RE = re.compile(
    r"\[LighterGlueDiag\]\s+attempt\s+current=(\d+)\s+reference=(\d+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", required=True,
        help="EuRoC mav0 directory, e.g. .../MH_01_easy/mav0")
    p.add_argument(
        "--trajectory", required=True,
        help="XRSLAM TUM BODY trajectory from the same diagnostic run")
    p.add_argument(
        "--sensor-config", required=True,
        help="XRSLAM EuRoC sensor YAML")
    p.add_argument(
        "--log", required=True,
        help="XRSLAM log containing recovery attempt lines")
    p.add_argument(
        "--xfeat", required=True,
        help="Path to verlab/accelerated_features checkout")

    p.add_argument(
        "--support-gaps", default="10,20",
        help=(
            "Comma-separated earlier camera-frame gaps used to triangulate "
            "learned landmarks into each reference anchor. Default: 10,20"))
    p.add_argument(
        "--top-k", type=int, default=512,
        help="Native XFeat features extracted per image")
    p.add_argument(
        "--source-features", type=int, default=128,
        help=(
            "Maximum landmark-bearing reference features passed to "
            "LighterGlue, mirroring the prior system"))
    p.add_argument(
        "--min-conf", type=float, default=0.1,
        help="LighterGlue minimum matching confidence")

    p.add_argument(
        "--support-epipolar-px", type=float, default=1.0,
        help=(
            "Known-pose Sampson-error gate for support->reference matches, "
            "expressed approximately in pixels"))
    p.add_argument(
        "--triangulation-reproj-px", type=float, default=2.0,
        help="Maximum two-view reprojection error per view")
    p.add_argument(
        "--min-triangulation-angle-deg", type=float, default=1.0,
        help="Minimum support/reference triangulation parallax")
    p.add_argument(
        "--max-landmark-distance", type=float, default=50.0,
        help="Maximum distance from either triangulating camera, metres")

    # The previous system used sqrt(5.991 * levelSigma2[0]); its XFeat
    # keypoints are level zero, giving ~2.448 px when sigma^2=1.
    p.add_argument(
        "--pnp-reproj-px", type=float, default=2.448,
        help="PnP RANSAC reprojection threshold")
    p.add_argument(
        "--pnp-iterations", type=int, default=300)
    p.add_argument(
        "--pnp-confidence", type=float, default=0.99)
    p.add_argument(
        "--pose-time-tolerance", type=float, default=0.03,
        help="Maximum TUM-pose timestamp mismatch for a camera frame")
    p.add_argument(
        "--max-attempts", type=int, default=0,
        help="0 = process every unique recovery attempt found in the log")
    p.add_argument(
        "--ground-truth", default="",
        help=(
            "Optional EuRoC state_groundtruth_estimate0/data.csv. If omitted, "
            "the benchmark automatically uses DATASET/state_groundtruth_estimate0/data.csv "
            "when it exists."))
    p.add_argument(
        "--ground-truth-time-tolerance", type=float, default=0.01,
        help="Maximum timestamp mismatch for EuRoC ground-truth lookup")
    return p.parse_args()


def parse_support_gaps(text):
    gaps = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        gap = int(token)
        if gap <= 0:
            raise ValueError("support gaps must be positive")
        if gap not in gaps:
            gaps.append(gap)
    if not gaps:
        raise ValueError("at least one support gap is required")
    return gaps


def parse_bracket_vector(text, key):
    m = re.search(rf"{re.escape(key)}\s*:\s*\[([^\]]+)\]", text)
    if not m:
        raise RuntimeError(f"cannot find {key} in sensor config")
    return np.asarray(
        [float(v.strip()) for v in m.group(1).split(",")],
        dtype=np.float64,
    )


def load_sensor_config(path):
    text = Path(path).read_text()
    intr = parse_bracket_vector(text, "intrinsics")
    dist = parse_bracket_vector(text, "distortion")[:4]
    q_bc = parse_bracket_vector(text, "q_bc")
    p_bc = parse_bracket_vector(text, "p_bc")

    if intr.size != 4:
        raise RuntimeError("expected four camera intrinsics")
    if q_bc.size != 4 or p_bc.size != 3:
        raise RuntimeError("unexpected camera extrinsic shape")

    fx, fy, cx, cy = intr
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, dist.reshape(-1, 1), q_bc, p_bc


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        raise RuntimeError("zero quaternion")
    return q / n


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_to_R(q):
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),
             2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),
             2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx),
             1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quat(R):
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return quat_normalize(np.array([qx, qy, qz, qw]))


def camera_pose_from_body(p_wb, q_wb, p_bc, q_bc):
    # XRSLAM Frame::get_pose(camera):
    # q_wc = q_wb * q_bc
    # p_wc = p_wb + R_wb * p_bc
    R_wb = quat_to_R(q_wb)
    q_wc = quat_normalize(quat_mul(q_wb, q_bc))
    p_wc = p_wb + R_wb @ p_bc
    return p_wc, q_wc


def load_tum_camera_poses(path, p_bc, q_bc):
    times = []
    poses = []
    skipped_invalid = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            values = [float(v) for v in s.split()]
            if len(values) != 8:
                continue

            t = values[0]
            p_wb = np.asarray(values[1:4], dtype=np.float64)
            q_wb = np.asarray(values[4:8], dtype=np.float64)

            # XRSLAM diagnostic trajectories may contain timestamped
            # placeholder rows with an all-zero quaternion. Such a row is
            # not a rigid-body pose, so treat it as an unavailable pose
            # sample instead of attempting to normalize it.
            if (
                not np.isfinite(t)
                or not np.all(np.isfinite(p_wb))
                or not np.all(np.isfinite(q_wb))
                or float(np.linalg.norm(q_wb)) <= 1.0e-12
            ):
                skipped_invalid += 1
                continue

            p_wc, q_wc = camera_pose_from_body(
                p_wb, q_wb, p_bc, q_bc)
            times.append(t)
            poses.append((p_wc, q_wc))

    if not times:
        raise RuntimeError(
            "trajectory contains no valid non-zero-quaternion poses")

    if skipped_invalid:
        print(
            f"# trajectory skipped_invalid_poses={skipped_invalid}",
            flush=True,
        )

    return np.asarray(times, dtype=np.float64), poses



def load_euroc_ground_truth_camera_poses(path, p_bc, q_bc):
    """Load EuRoC ground-truth IMU/body poses and convert them to camera poses.

    EuRoC state_groundtruth_estimate0/data.csv stores
      timestamp_ns, p_RS_R xyz, q_RS wxyz, ...
    For the supplied XRSLAM EuRoC config, body == IMU (identity q_bi/p_bi), so
    the same camera extrinsic q_bc/p_bc used for the XRSLAM trajectory applies.
    """
    times = []
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                continue
            try:
                t = int(row[0].strip()) * 1.0e-9
                p_wb = np.asarray(
                    [float(row[1]), float(row[2]), float(row[3])],
                    dtype=np.float64,
                )
                # EuRoC CSV is w,x,y,z; this script consistently uses x,y,z,w.
                q_wb = np.asarray(
                    [float(row[5]), float(row[6]), float(row[7]), float(row[4])],
                    dtype=np.float64,
                )
            except (ValueError, IndexError):
                continue

            if (
                not np.isfinite(t)
                or not np.all(np.isfinite(p_wb))
                or not np.all(np.isfinite(q_wb))
                or float(np.linalg.norm(q_wb)) <= 1.0e-12
            ):
                continue

            p_wc, q_wc = camera_pose_from_body(
                p_wb, q_wb, p_bc, q_bc)
            times.append(t)
            poses.append((p_wc, q_wc))

    if not times:
        raise RuntimeError(f"ground truth contains no valid poses: {path}")
    return np.asarray(times, dtype=np.float64), poses

def nearest_pose(t, pose_times, poses, tolerance):
    i = bisect.bisect_left(pose_times, t)
    candidates = []
    if i < len(pose_times):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    if not candidates:
        return None

    j = min(candidates, key=lambda k: abs(pose_times[k] - t))
    dt = abs(float(pose_times[j]) - t)
    if dt > tolerance:
        return None
    return poses[j], dt


def load_camera_index(dataset):
    csv_path = Path(dataset) / "cam0" / "data.csv"
    image_dir = Path(dataset) / "cam0" / "data"

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            timestamp_ns = int(row[0])
            filename = row[1].strip()
            rows.append((timestamp_ns * 1.0e-9, image_dir / filename))

    rows.sort(key=lambda x: x[0])
    return rows


def load_attempts(log_path):
    attempts = []
    seen = set()

    for line in Path(log_path).read_text(errors="replace").splitlines():
        m = ATTEMPT_RE.search(line)
        if not m:
            continue
        current = int(m.group(1))
        reference = int(m.group(2))
        key = (current, reference)
        if key in seen:
            continue
        seen.add(key)
        attempts.append(key)

    if not attempts:
        raise RuntimeError(
            "no '[LighterGlueDiag] attempt current=... reference=...' "
            "lines found in log")

    return attempts


def read_xrslam_image(frame_index, camera_rows, K, D):
    if frame_index < 0 or frame_index >= len(camera_rows):
        raise IndexError(frame_index)

    t, path = camera_rows[frame_index]
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot read {path}")

    # Matches EurocDatasetReader: undistort first, then grayscale.
    image = cv2.undistort(raw, K, D)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return t, image


def make_feature_dict(xfeat, image, top_k):
    out = xfeat.detectAndCompute(image, top_k=top_k)[0]
    h, w = image.shape[:2]
    out["image_size"] = (w, h)
    return out


def subset_feature_dict(features, indices):
    # LighterGlue only needs keypoints/descriptors/image_size. Preserve scores
    # when available for easier inspection/debugging.
    if not indices:
        raise ValueError("cannot build an empty feature subset")

    device = features["keypoints"].device
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)

    subset = {
        "keypoints": features["keypoints"].index_select(0, idx),
        "descriptors": features["descriptors"].index_select(0, idx),
        "image_size": features["image_size"],
    }
    if "scores" in features:
        subset["scores"] = features["scores"].index_select(0, idx)
    return subset


def normalize_pixel(p, K):
    return np.array(
        [
            (float(p[0]) - K[0, 2]) / K[0, 0],
            (float(p[1]) - K[1, 2]) / K[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )


def projection_from_pose(p_wc, q_wc):
    R_wc = quat_to_R(q_wc)
    R_cw = R_wc.T
    t_cw = -R_cw @ p_wc
    return np.column_stack((R_cw, t_cw))


def relative_essential(pose0, pose1):
    # x1^T E_10 x0 = 0, where points are normalized camera coordinates.
    p_w0, q_w0 = pose0
    p_w1, q_w1 = pose1
    R_w0 = quat_to_R(q_w0)
    R_w1 = quat_to_R(q_w1)

    R_10 = R_w1.T @ R_w0
    t_10 = R_w1.T @ (p_w0 - p_w1)

    tx = np.array(
        [
            [0.0, -t_10[2], t_10[1]],
            [t_10[2], 0.0, -t_10[0]],
            [-t_10[1], t_10[0], 0.0],
        ],
        dtype=np.float64,
    )
    return tx @ R_10


def sampson_error_sq(E, x0, x1):
    Ex0 = E @ x0
    Etx1 = E.T @ x1
    numerator = float(x1 @ Ex0)
    denominator = (
        Ex0[0] * Ex0[0] +
        Ex0[1] * Ex0[1] +
        Etx1[0] * Etx1[0] +
        Etx1[1] * Etx1[1]
    )
    if denominator <= 1.0e-16:
        return float("inf")
    return numerator * numerator / denominator


def triangulate_two_view(p0, p1, pose0, pose1, K):
    P0 = projection_from_pose(*pose0)
    P1 = projection_from_pose(*pose1)
    x0 = normalize_pixel(p0, K)
    x1 = normalize_pixel(p1, K)

    A = np.vstack(
        [
            x0[0] * P0[2] - P0[0],
            x0[1] * P0[2] - P0[1],
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
        ]
    )
    _, _, vh = np.linalg.svd(A)
    Xh = vh[-1]
    if abs(float(Xh[3])) < 1.0e-12:
        return None
    return Xh[:3] / Xh[3]


def project_world(X, pose, K):
    p_wc, q_wc = pose
    R_cw = quat_to_R(q_wc).T
    Xc = R_cw @ (X - p_wc)
    if Xc[2] <= 0.0:
        return None, float(Xc[2])

    uv = np.array(
        [
            K[0, 0] * Xc[0] / Xc[2] + K[0, 2],
            K[1, 1] * Xc[1] / Xc[2] + K[1, 2],
        ],
        dtype=np.float64,
    )
    return uv, float(Xc[2])


def triangulation_angle_deg(X, pose0, pose1):
    a = X - pose0[0]
    b = X - pose1[0]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0

    cosine = np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def rotation_error_deg(q_est_wc, q_ref_wc):
    R_delta = quat_to_R(q_est_wc).T @ quat_to_R(q_ref_wc)
    cosine = np.clip((float(np.trace(R_delta)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def pnp_pose_to_camera_world(rvec, tvec):
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    p_wc = -R_wc @ tvec.reshape(3)
    q_wc = rotation_matrix_to_quat(R_wc)
    return p_wc, q_wc


def build_reference_landmarks(
    reference,
    support_gaps,
    frame_data,
    pose_for_frame,
    K,
    xfeat,
    min_conf,
    epipolar_px,
    reproj_px,
    min_angle_deg,
    max_distance,
):
    _, _, feat_r = frame_data(reference)
    pose_r = pose_for_frame(reference)
    if pose_r is None:
        return {}, []

    # ref_index -> dict(X, angle, reproj, support)
    best = {}
    support_stats = []

    avg_focal = max(1.0, 0.5 * (K[0, 0] + K[1, 1]))
    normalized_epi_threshold_sq = (epipolar_px / avg_focal) ** 2

    for gap in support_gaps:
        support = reference - gap
        if support < 0:
            continue

        pose_s = pose_for_frame(support)
        if pose_s is None:
            support_stats.append((support, 0, 0, 0))
            continue

        _, _, feat_s = frame_data(support)
        _, _, idx_sr = xfeat.match_lighterglue(
            feat_s, feat_r, min_conf=min_conf)
        idx_sr = np.asarray(idx_sr, dtype=np.int64)

        E = relative_essential(pose_s, pose_r)
        epipolar_ok = 0
        triangulated_ok = 0

        for s_idx, r_idx in idx_sr:
            s_idx = int(s_idx)
            r_idx = int(r_idx)

            p_s = feat_s["keypoints"][s_idx].detach().cpu().numpy()
            p_r = feat_r["keypoints"][r_idx].detach().cpu().numpy()

            x_s = normalize_pixel(p_s, K)
            x_r = normalize_pixel(p_r, K)
            if sampson_error_sq(E, x_s, x_r) > normalized_epi_threshold_sq:
                continue
            epipolar_ok += 1

            X = triangulate_two_view(p_s, p_r, pose_s, pose_r, K)
            if X is None or not np.all(np.isfinite(X)):
                continue

            uv_s, z_s = project_world(X, pose_s, K)
            uv_r, z_r = project_world(X, pose_r, K)
            if uv_s is None or uv_r is None or z_s <= 0.0 or z_r <= 0.0:
                continue

            dist_s = float(np.linalg.norm(X - pose_s[0]))
            dist_r = float(np.linalg.norm(X - pose_r[0]))
            if max(dist_s, dist_r) > max_distance:
                continue

            err_s = float(np.linalg.norm(uv_s - p_s))
            err_r = float(np.linalg.norm(uv_r - p_r))
            max_reproj = max(err_s, err_r)
            if max_reproj > reproj_px:
                continue

            angle = triangulation_angle_deg(X, pose_s, pose_r)
            if angle < min_angle_deg:
                continue

            triangulated_ok += 1

            candidate = {
                "X": X,
                "angle": angle,
                "reproj": max_reproj,
                "support": support,
            }

            previous = best.get(r_idx)
            if previous is None:
                best[r_idx] = candidate
            else:
                # Prefer stronger parallax. Break near-ties with lower
                # reprojection error.
                if (
                    candidate["angle"] > previous["angle"] + 1.0e-6
                    or (
                        abs(candidate["angle"] - previous["angle"]) <= 1.0e-6
                        and candidate["reproj"] < previous["reproj"]
                    )
                ):
                    best[r_idx] = candidate

        support_stats.append(
            (support, int(len(idx_sr)), epipolar_ok, triangulated_ok)
        )

    return best, support_stats



def estimate_se3_alignment(source_points, target_points):
    """Estimate target ~= R * source + t with fixed scale 1."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("alignment points must be Nx3 arrays")
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("alignment point arrays must have matching Nx3 shape")
    if source.shape[0] < 3:
        raise RuntimeError("need at least three pose pairs for SE(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (
        target_centered.T @ source_centered / source.shape[0]
    )
    U, _, Vt = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        correction[2, 2] = -1.0

    R = U @ correction @ Vt
    t = target_mean - R @ source_mean
    return R, t


def apply_world_alignment_to_pose(pose, R_align, t_align):
    p_w, q_w = pose
    p_aligned = R_align @ p_w + t_align
    R_aligned = R_align @ quat_to_R(q_w)
    q_aligned = rotation_matrix_to_quat(R_aligned)
    return p_aligned, q_aligned


def build_ground_truth_alignment(
    pose_times,
    poses,
    gt_pose_times,
    gt_poses,
    tolerance,
):
    source_points = []
    target_points = []
    pose_pairs = []

    for t, pose in zip(pose_times, poses):
        result = nearest_pose(
            float(t), gt_pose_times, gt_poses, tolerance
        )
        if result is None:
            continue
        gt_pose, _ = result
        source_points.append(pose[0])
        target_points.append(gt_pose[0])
        pose_pairs.append((pose, gt_pose))

    if len(source_points) < 3:
        raise RuntimeError(
            "insufficient common XRSLAM/ground-truth poses for SE(3) alignment"
        )

    R_align, t_align = estimate_se3_alignment(
        source_points, target_points
    )

    translation_residuals = []
    rotation_residuals = []
    for pose, gt_pose in pose_pairs:
        aligned = apply_world_alignment_to_pose(
            pose, R_align, t_align
        )
        translation_residuals.append(
            float(np.linalg.norm(aligned[0] - gt_pose[0]))
        )
        rotation_residuals.append(
            rotation_error_deg(aligned[1], gt_pose[1])
        )

    translation_residuals = np.asarray(
        translation_residuals, dtype=np.float64
    )
    rotation_residuals = np.asarray(
        rotation_residuals, dtype=np.float64
    )

    return (
        R_align,
        t_align,
        len(pose_pairs),
        float(np.sqrt(np.mean(translation_residuals ** 2))),
        float(np.median(translation_residuals)),
        float(np.median(rotation_residuals)),
    )

def main():
    args = parse_args()
    support_gaps = parse_support_gaps(args.support_gaps)

    xfeat_root = str(Path(args.xfeat).resolve())
    if xfeat_root not in sys.path:
        sys.path.insert(0, xfeat_root)

    try:
        from modules.xfeat import XFeat
    except Exception as e:
        raise RuntimeError(
            "failed to import XFeat from accelerated_features; "
            "check --xfeat and its Python dependencies") from e

    K, D, q_bc, p_bc = load_sensor_config(args.sensor_config)
    camera_rows = load_camera_index(args.dataset)
    pose_times, poses = load_tum_camera_poses(
        args.trajectory, p_bc, q_bc)

    ground_truth_path = (
        Path(args.ground_truth)
        if args.ground_truth
        else Path(args.dataset) / "state_groundtruth_estimate0" / "data.csv"
    )
    gt_pose_times = None
    gt_poses = None
    R_gt_align = None
    t_gt_align = None

    if ground_truth_path.exists():
        gt_pose_times, gt_poses = load_euroc_ground_truth_camera_poses(
            ground_truth_path, p_bc, q_bc)
        print(f"# ground_truth={ground_truth_path}", flush=True)

        (
            R_gt_align,
            t_gt_align,
            alignment_pairs,
            alignment_translation_rmse,
            alignment_translation_median,
            alignment_rotation_median,
        ) = build_ground_truth_alignment(
            pose_times,
            poses,
            gt_pose_times,
            gt_poses,
            args.ground_truth_time_tolerance,
        )

        print(
            "# ground_truth_alignment "
            f"mode=SE3 scale=1.000000 "
            f"pairs={alignment_pairs} "
            f"trajectory_t_rmse_m={alignment_translation_rmse:.4f} "
            f"trajectory_t_median_m={alignment_translation_median:.4f} "
            f"trajectory_r_median_deg={alignment_rotation_median:.3f}",
            flush=True,
        )
    else:
        print(
            f"# ground_truth=unavailable path={ground_truth_path}",
            flush=True,
        )

    attempts = load_attempts(args.log)

    if args.max_attempts > 0:
        attempts = attempts[: args.max_attempts]

    xfeat = XFeat(top_k=args.top_k)

    feature_cache = {}
    image_cache = {}
    pose_cache = {}
    reference_landmark_cache = {}

    def frame_data(index):
        if index not in image_cache:
            image_cache[index] = read_xrslam_image(
                index, camera_rows, K, D)
        t, image = image_cache[index]

        if index not in feature_cache:
            feature_cache[index] = make_feature_dict(
                xfeat, image, args.top_k)

        return t, image, feature_cache[index]

    def pose_for_frame(index):
        if index in pose_cache:
            return pose_cache[index]

        if index < 0 or index >= len(camera_rows):
            pose_cache[index] = None
            return None

        t = camera_rows[index][0]
        result = nearest_pose(
            t, pose_times, poses, args.pose_time_tolerance)
        pose_cache[index] = None if result is None else result[0]
        return pose_cache[index]

    def ground_truth_pose_for_frame(index):
        if gt_pose_times is None or gt_poses is None:
            return None
        if index < 0 or index >= len(camera_rows):
            return None
        t = camera_rows[index][0]
        result = nearest_pose(
            t,
            gt_pose_times,
            gt_poses,
            args.ground_truth_time_tolerance,
        )
        return None if result is None else result[0]

    def learned_landmarks_for_reference(reference):
        if reference not in reference_landmark_cache:
            reference_landmark_cache[reference] = build_reference_landmarks(
                reference=reference,
                support_gaps=support_gaps,
                frame_data=frame_data,
                pose_for_frame=pose_for_frame,
                K=K,
                xfeat=xfeat,
                min_conf=args.min_conf,
                epipolar_px=args.support_epipolar_px,
                reproj_px=args.triangulation_reproj_px,
                min_angle_deg=args.min_triangulation_angle_deg,
                max_distance=args.max_landmark_distance,
            )
        return reference_landmark_cache[reference]

    accepted = 0
    pnp_attempted = 0
    pnp_solved = 0
    total_lg_matches = 0
    total_pnp_inliers = 0
    translation_errors = []
    rotation_errors = []
    pnp_gt_translation_errors = []
    pnp_gt_rotation_errors = []
    xrslam_gt_translation_errors = []
    xrslam_gt_rotation_errors = []
    pnp_better_translation = 0
    pnp_better_rotation = 0
    gt_compared = 0

    print(
        "# current reference learned_landmarks source_features "
        "lg_matches pnp_ok pnp_inliers pnp_ratio "
        "pose_t_delta_m pose_r_delta_deg "
        "pnp_gt_t_m pnp_gt_r_deg xrslam_gt_t_m xrslam_gt_r_deg supports"
    )

    for current, reference in attempts:
        if current >= len(camera_rows) or reference >= len(camera_rows):
            continue

        pose_c = pose_for_frame(current)
        pose_r = pose_for_frame(reference)
        if pose_c is None or pose_r is None:
            print(
                f"{current} {reference} SKIP missing_pose",
                flush=True,
            )
            continue

        landmarks, support_stats = learned_landmarks_for_reference(reference)
        if not landmarks:
            print(
                f"{current} {reference} 0 0 0 0 0 0.000 nan nan "
                f"supports={support_stats}",
                flush=True,
            )
            continue

        _, _, feat_r = frame_data(reference)
        _, _, feat_c = frame_data(current)

        # Official XFeat sparse output is score sorted. Sorting by original
        # reference index therefore mirrors the prior system's "first
        # MapPoint-bearing features up to source budget" policy.
        source_ref_indices = sorted(landmarks.keys())
        source_ref_indices = source_ref_indices[: args.source_features]

        if not source_ref_indices:
            continue

        source_feat = subset_feature_dict(feat_r, source_ref_indices)

        try:
            _, _, idx_rc = xfeat.match_lighterglue(
                source_feat, feat_c, min_conf=args.min_conf)
        except Exception as e:
            print(
                f"{current} {reference} ERROR lighterglue {e}",
                flush=True,
            )
            continue

        idx_rc = np.asarray(idx_rc, dtype=np.int64)
        total_lg_matches += int(len(idx_rc))

        object_points = []
        image_points = []
        used_current = set()

        # LighterGlue source index indexes source_ref_indices directly.
        for src_idx, current_idx in idx_rc:
            src_idx = int(src_idx)
            current_idx = int(current_idx)

            if (
                src_idx < 0
                or src_idx >= len(source_ref_indices)
                or current_idx < 0
                or current_idx >= int(feat_c["keypoints"].shape[0])
            ):
                continue

            if current_idx in used_current:
                continue

            ref_idx = source_ref_indices[src_idx]
            landmark = landmarks.get(ref_idx)
            if landmark is None:
                continue

            used_current.add(current_idx)
            object_points.append(landmark["X"])
            image_points.append(
                feat_c["keypoints"][current_idx].detach().cpu().numpy()
            )

        accepted += 1
        pnp_matches = len(object_points)

        pnp_ok = False
        pnp_inliers = 0
        pnp_ratio = 0.0
        t_delta = float("nan")
        r_delta = float("nan")
        pnp_gt_t = float("nan")
        pnp_gt_r = float("nan")
        xrslam_gt_t = float("nan")
        xrslam_gt_r = float("nan")

        gt_pose = ground_truth_pose_for_frame(current)
        aligned_xrslam_pose = None
        if (
            gt_pose is not None
            and R_gt_align is not None
            and t_gt_align is not None
        ):
            aligned_xrslam_pose = apply_world_alignment_to_pose(
                pose_c, R_gt_align, t_gt_align
            )
            xrslam_gt_t = float(
                np.linalg.norm(
                    aligned_xrslam_pose[0] - gt_pose[0]
                )
            )
            xrslam_gt_r = rotation_error_deg(
                aligned_xrslam_pose[1], gt_pose[1]
            )

        if pnp_matches >= 6:
            pnp_attempted += 1
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray(object_points, dtype=np.float64),
                np.asarray(image_points, dtype=np.float64),
                K,
                None,
                iterationsCount=args.pnp_iterations,
                reprojectionError=args.pnp_reproj_px,
                confidence=args.pnp_confidence,
                flags=cv2.SOLVEPNP_EPNP,
            )

            if ok and inliers is not None:
                pnp_ok = True
                pnp_solved += 1
                pnp_inliers = int(inliers.size)
                pnp_ratio = pnp_inliers / pnp_matches
                total_pnp_inliers += pnp_inliers

                p_est, q_est = pnp_pose_to_camera_world(rvec, tvec)
                t_delta = float(np.linalg.norm(p_est - pose_c[0]))
                r_delta = rotation_error_deg(q_est, pose_c[1])
                translation_errors.append(t_delta)
                rotation_errors.append(r_delta)

                if (
                    gt_pose is not None
                    and R_gt_align is not None
                    and t_gt_align is not None
                ):
                    aligned_pnp_pose = apply_world_alignment_to_pose(
                        (p_est, q_est), R_gt_align, t_gt_align
                    )
                    pnp_gt_t = float(
                        np.linalg.norm(
                            aligned_pnp_pose[0] - gt_pose[0]
                        )
                    )
                    pnp_gt_r = rotation_error_deg(
                        aligned_pnp_pose[1], gt_pose[1]
                    )
                    pnp_gt_translation_errors.append(pnp_gt_t)
                    pnp_gt_rotation_errors.append(pnp_gt_r)
                    xrslam_gt_translation_errors.append(xrslam_gt_t)
                    xrslam_gt_rotation_errors.append(xrslam_gt_r)
                    gt_compared += 1
                    if pnp_gt_t < xrslam_gt_t:
                        pnp_better_translation += 1
                    if pnp_gt_r < xrslam_gt_r:
                        pnp_better_rotation += 1

        print(
            f"{current} {reference} "
            f"{len(landmarks)} {len(source_ref_indices)} "
            f"{pnp_matches} {1 if pnp_ok else 0} "
            f"{pnp_inliers} {pnp_ratio:.3f} "
            f"{t_delta:.4f} {r_delta:.3f} "
            f"{pnp_gt_t:.4f} {pnp_gt_r:.3f} "
            f"{xrslam_gt_t:.4f} {xrslam_gt_r:.3f} "
            f"supports={support_stats}",
            flush=True,
        )

    median_t = (
        float(np.median(translation_errors))
        if translation_errors else float("nan")
    )
    median_r = (
        float(np.median(rotation_errors))
        if rotation_errors else float("nan")
    )
    aggregate_ratio = (
        total_pnp_inliers / total_lg_matches
        if total_lg_matches else 0.0
    )
    median_pnp_gt_t = (
        float(np.median(pnp_gt_translation_errors))
        if pnp_gt_translation_errors else float("nan")
    )
    median_pnp_gt_r = (
        float(np.median(pnp_gt_rotation_errors))
        if pnp_gt_rotation_errors else float("nan")
    )
    median_xrslam_gt_t = (
        float(np.median(xrslam_gt_translation_errors))
        if xrslam_gt_translation_errors else float("nan")
    )
    median_xrslam_gt_r = (
        float(np.median(xrslam_gt_rotation_errors))
        if xrslam_gt_rotation_errors else float("nan")
    )

    print(
        "# summary "
        f"recovery_attempts={accepted} "
        f"pnp_attempted={pnp_attempted} "
        f"pnp_solved={pnp_solved} "
        f"lg_2d3d_matches={total_lg_matches} "
        f"pnp_inliers={total_pnp_inliers} "
        f"aggregate_inliers_per_lg_match={aggregate_ratio:.3f} "
        f"median_pose_t_delta_m={median_t:.4f} "
        f"median_pose_r_delta_deg={median_r:.3f} "
        f"gt_compared={gt_compared} "
        f"median_pnp_gt_t_m={median_pnp_gt_t:.4f} "
        f"median_pnp_gt_r_deg={median_pnp_gt_r:.3f} "
        f"median_xrslam_gt_t_m={median_xrslam_gt_t:.4f} "
        f"median_xrslam_gt_r_deg={median_xrslam_gt_r:.3f} "
        f"pnp_better_t={pnp_better_translation}/{gt_compared} "
        f"pnp_better_r={pnp_better_rotation}/{gt_compared}",
        flush=True,
    )


if __name__ == "__main__":
    main()
