#!/usr/bin/env python3
"""Benchmark KLT vs. XFeat + LighterGlue on EuRoC frame gaps.

This is deliberately an offline experiment. It does not modify XRSLAM tracks.
For each temporal gap it reports correspondence count, essential-matrix RANSAC
inliers/inlier ratio, and pair-processing time.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


EUROC_INTRINSICS = (458.654, 457.296, 367.215, 248.375)
EUROC_DISTORTION = (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xfeat",
        type=Path,
        required=True,
        help="Path to the official verlab/accelerated_features repository.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to EuRoC mav0 directory.",
    )
    parser.add_argument(
        "--gaps",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Frame gaps to benchmark.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=40,
        help="Maximum sampled pairs per gap.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Maximum XFeat/KLT features per image.",
    )
    parser.add_argument(
        "--lighterglue-min-conf",
        type=float,
        default=0.1,
        help="LighterGlue match confidence threshold.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="Essential-matrix RANSAC threshold in pixels.",
    )
    return parser.parse_args()


def camera_matrix():
    fx, fy, cx, cy = EUROC_INTRINSICS
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def load_undistorted(path, k, distortion):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.undistort(image, k, distortion)


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def essential_stats(p0, p1, k, threshold):
    if len(p0) < 8 or len(p1) < 8:
        return 0, 0.0
    _, mask = cv2.findEssentialMat(
        np.asarray(p0, dtype=np.float32),
        np.asarray(p1, dtype=np.float32),
        k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold,
    )
    if mask is None:
        return 0, 0.0
    inliers = int(np.count_nonzero(mask))
    return inliers, inliers / max(1, len(p0))


def klt_match(image0, image1, top_k):
    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=top_k,
        qualityLevel=0.01,
        minDistance=20.0,
        blockSize=3,
    )
    if p0 is None or len(p0) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    p1, status1, _ = cv2.calcOpticalFlowPyrLK(
        gray0,
        gray1,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    if p1 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray0,
        p1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )

    p0 = p0.reshape(-1, 2)
    p1 = p1.reshape(-1, 2)
    p0_back = p0_back.reshape(-1, 2)
    valid = status1.reshape(-1).astype(bool) & status_back.reshape(-1).astype(bool)
    valid &= np.linalg.norm(p0 - p0_back, axis=1) <= 0.5

    h, w = gray1.shape
    valid &= p1[:, 0] >= 20
    valid &= p1[:, 0] < w - 20
    valid &= p1[:, 1] >= 20
    valid &= p1[:, 1] < h - 20
    return p0[valid], p1[valid]


def xfeat_lighterglue_match(xfeat, image0, image1, top_k, min_conf):
    rgb0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    d0 = xfeat.detectAndCompute(rgb0, top_k=top_k)[0]
    d1 = xfeat.detectAndCompute(rgb1, top_k=top_k)[0]
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    d0["image_size"] = (w0, h0)
    d1["image_size"] = (w1, h1)

    p0, p1, _ = xfeat.match_lighterglue(d0, d1, min_conf=min_conf)
    return np.asarray(p0, np.float32), np.asarray(p1, np.float32)


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.median(a))


def main():
    args = parse_args()
    sys.path.insert(0, str(args.xfeat.resolve()))
    try:
        from modules.xfeat import XFeat
    except Exception as exc:
        raise RuntimeError(
            "failed to import official XFeat; check --xfeat and install "
            "its dependencies (including kornia for LighterGlue)"
        ) from exc

    image_dir = args.dataset / "cam0" / "data"
    images = sorted(image_dir.glob("*.png"))
    if len(images) < 2:
        raise RuntimeError(f"no EuRoC images found under {image_dir}")

    k = camera_matrix()
    distortion = np.asarray(EUROC_DISTORTION, dtype=np.float64)
    xfeat = XFeat(top_k=args.top_k)

    device = getattr(xfeat, "dev", torch.device("cpu"))
    print(f"XFeat device: {device}")
    print(f"Images: {len(images)}")
    print(
        "gap method pairs mean_matches median_matches "
        "mean_inliers median_inliers mean_inlier_ratio "
        "median_inlier_ratio mean_ms median_ms"
    )

    # Warm up XFeat and lazy LighterGlue initialization outside measurements.
    warm0 = load_undistorted(images[0], k, distortion)
    warm1 = load_undistorted(images[1], k, distortion)
    xfeat_lighterglue_match(
        xfeat, warm0, warm1, args.top_k, args.lighterglue_min_conf
    )
    synchronize_cuda()

    for gap in args.gaps:
        if gap <= 0 or gap >= len(images):
            continue

        sample_count = min(args.pairs, len(images) - gap)
        indices = np.linspace(
            0, len(images) - gap - 1, sample_count, dtype=np.int64
        )

        results = {
            "klt": {"matches": [], "inliers": [], "ratios": [], "ms": []},
            "lighterglue": {
                "matches": [],
                "inliers": [],
                "ratios": [],
                "ms": [],
            },
        }

        for index in indices:
            image0 = load_undistorted(images[int(index)], k, distortion)
            image1 = load_undistorted(images[int(index) + gap], k, distortion)

            t0 = time.perf_counter()
            p0, p1 = klt_match(image0, image1, args.top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            inliers, ratio = essential_stats(
                p0, p1, k, args.ransac_threshold
            )
            results["klt"]["matches"].append(len(p0))
            results["klt"]["inliers"].append(inliers)
            results["klt"]["ratios"].append(ratio)
            results["klt"]["ms"].append(elapsed_ms)

            synchronize_cuda()
            t0 = time.perf_counter()
            p0, p1 = xfeat_lighterglue_match(
                xfeat,
                image0,
                image1,
                args.top_k,
                args.lighterglue_min_conf,
            )
            synchronize_cuda()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            inliers, ratio = essential_stats(
                p0, p1, k, args.ransac_threshold
            )
            results["lighterglue"]["matches"].append(len(p0))
            results["lighterglue"]["inliers"].append(inliers)
            results["lighterglue"]["ratios"].append(ratio)
            results["lighterglue"]["ms"].append(elapsed_ms)

        for method in ("klt", "lighterglue"):
            r = results[method]
            mean_matches, median_matches = summarize(r["matches"])
            mean_inliers, median_inliers = summarize(r["inliers"])
            mean_ratio, median_ratio = summarize(r["ratios"])
            mean_ms, median_ms = summarize(r["ms"])
            print(
                f"{gap:3d} {method:11s} {len(r['matches']):5d} "
                f"{mean_matches:12.1f} {median_matches:14.1f} "
                f"{mean_inliers:12.1f} {median_inliers:14.1f} "
                f"{mean_ratio:17.3f} {median_ratio:19.3f} "
                f"{mean_ms:8.2f} {median_ms:9.2f}"
            )


if __name__ == "__main__":
    main()
