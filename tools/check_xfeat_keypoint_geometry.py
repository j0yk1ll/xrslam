#!/usr/bin/env python3
"""Compare sparse XFeat ONNX keypoint geometry with official PyTorch XFeat.

Reports:
  * image/model shapes and keypoint bounds
  * out-of-bounds fractions
  * nearest-neighbor agreement between ONNX and official XFeat keypoints
  * several simple coordinate transform hypotheses (identity, swap, and
    independent scale fits) to expose systematic export geometry errors
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xfeat", type=Path, required=True,
                   help="verlab/accelerated_features checkout")
    p.add_argument("--sparse-model", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--euroc", type=Path,
                        help="EuRoC mav0 directory")
    p.add_argument("--frame-index", type=int, default=308)
    p.add_argument("--top-k", type=int, default=512)
    return p.parse_args()


def resolve_image(args):
    if args.image:
        return args.image
    rows = [
        x.strip()
        for x in (args.euroc / "cam0" / "data.csv").read_text().splitlines()
        if x.strip() and not x.startswith("#")
    ]
    filename = rows[args.frame_index].split(",", 1)[1].strip()
    return args.euroc / "cam0" / "data" / filename


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else:
        raise SystemExit(f"unsupported image shape {bgr.shape}")
    return rgb


def onnx_keypoints(model, rgb, top_k):
    image = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    sess = ort.InferenceSession(
        str(model), providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {sess.get_inputs()[0].name: image})

    keypoints = None
    scores = None
    for meta, value in zip(sess.get_outputs(), outputs):
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            keypoints = arr.reshape(-1, 2).astype(np.float32)
        elif arr.size > 0 and not (
            arr.ndim >= 2 and arr.shape[-1] == 64
        ):
            scores = arr.reshape(-1).astype(np.float32)

    if keypoints is None:
        raise SystemExit(
            "could not identify ONNX keypoint output: "
            + repr([(m.name, np.asarray(v).shape)
                    for m, v in zip(sess.get_outputs(), outputs)])
        )

    if scores is not None and len(scores) == len(keypoints):
        order = np.argsort(-scores)
        keypoints = keypoints[order]
        scores = scores[order]

    return keypoints[:top_k], None if scores is None else scores[:top_k], sess


def official_keypoints(xfeat_root, rgb, top_k):
    sys.path.insert(0, str(xfeat_root.resolve()))
    from modules.xfeat import XFeat

    model = XFeat(top_k=top_k)
    image = torch.from_numpy(
        rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    ).to(model.dev)
    result = model.detectAndCompute(image, top_k=top_k)[0]
    return (
        result["keypoints"].detach().cpu().numpy().astype(np.float32),
        result["scores"].detach().cpu().numpy().astype(np.float32),
    )


def bounds(name, pts, width, height):
    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < width) &
        (pts[:, 1] >= 0) & (pts[:, 1] < height)
    )
    print(
        f"{name}: n={len(pts)} "
        f"x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
        f"y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
        f"in_bounds={valid.mean()*100:.2f}%"
    )
    return valid


def nearest_distances(a, b):
    # Sizes are small (~512), so an explicit pairwise matrix is fine.
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1)), d2.argmin(axis=1)


def summarize_nn(name, src, dst):
    d, idx = nearest_distances(src, dst)
    q = np.quantile(d, [0, .25, .5, .75, .9, .95, .99, 1])
    print(
        f"{name}: mean={d.mean():.3f}px median={q[2]:.3f}px "
        f"p90={q[4]:.3f}px p95={q[5]:.3f}px max={q[7]:.3f}px"
    )
    print(
        "  within 1/2/3/5/10 px: "
        + " ".join(
            f"{r}px={(d <= r).mean()*100:.1f}%"
            for r in [1, 2, 3, 5, 10]
        )
    )
    return d, idx


def fit_axis_scale(src, dst):
    """Fit dst ~= src * scale + bias independently per axis."""
    X = np.c_[src, np.ones(len(src))]
    ax, *_ = np.linalg.lstsq(X[:, [0, 2]], dst[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(X[:, [1, 2]], dst[:, 1], rcond=None)
    out = np.empty_like(src)
    out[:, 0] = src[:, 0] * ax[0] + ax[1]
    out[:, 1] = src[:, 1] * ay[0] + ay[1]
    return out, ax, ay


def mutual_pairs(a, b, max_dist=20.0):
    da, ia = nearest_distances(a, b)
    db, ib = nearest_distances(b, a)
    pairs = []
    for i, j in enumerate(ia):
        if da[i] <= max_dist and ib[j] == i:
            pairs.append((i, int(j)))
    return pairs


def main():
    args = parse_args()
    path = resolve_image(args)
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]

    print(f"image: {path}")
    print(f"image size: {w}x{h}")

    onnx_pts, onnx_scores, sess = onnx_keypoints(
        args.sparse_model, rgb, args.top_k
    )
    pt_pts, pt_scores = official_keypoints(
        args.xfeat, rgb, args.top_k
    )

    inp = sess.get_inputs()[0]
    print(f"ONNX input shape metadata: {inp.shape}")

    valid_onnx = bounds("ONNX", onnx_pts, w, h)
    valid_pt = bounds("official", pt_pts, w, h)

    # Raw agreement, considering only in-bounds points.
    a = onnx_pts[valid_onnx]
    b = pt_pts[valid_pt]
    summarize_nn("ONNX -> official identity", a, b)
    summarize_nn("official -> ONNX identity", b, a)

    # Swap hypothesis.
    swapped = a[:, ::-1].copy()
    swap_valid = (
        (swapped[:, 0] >= 0) & (swapped[:, 0] < w) &
        (swapped[:, 1] >= 0) & (swapped[:, 1] < h)
    )
    if swap_valid.any():
        summarize_nn(
            "ONNX(x,y)->(y,x) -> official",
            swapped[swap_valid], b
        )

    # Use conservative mutual nearest neighbors from raw coordinates to estimate
    # a simple per-axis affine correction. If the export has a systematic scale
    # or offset issue, this will expose it.
    pairs = mutual_pairs(a, b, max_dist=20.0)
    print(f"mutual NN pairs within 20px: {len(pairs)}")
    if len(pairs) >= 8:
        src = np.asarray([a[i] for i, _ in pairs], np.float32)
        dst = np.asarray([b[j] for _, j in pairs], np.float32)
        corrected, ax, ay = fit_axis_scale(src, dst)
        print(
            "axis fit official ~= ONNX * scale + bias:"
            f"  x: scale={ax[0]:.8f} bias={ax[1]:.4f}"
            f"  y: scale={ay[0]:.8f} bias={ay[1]:.4f}"
        )
        residual = np.linalg.norm(corrected - dst, axis=1)
        print(
            f"fit residual: mean={residual.mean():.3f}px "
            f"median={np.median(residual):.3f}px "
            f"p95={np.quantile(residual,.95):.3f}px"
        )

    # Expected preprocessing ratios for reference.
    rw = w / ((w // 32) * 32)
    rh = h / ((h // 32) * 32)
    print(
        f"official preprocess scale-back: rw={rw:.8f} rh={rh:.8f}"
    )

    print("top 12 out-of-bounds ONNX keypoints:")
    bad = np.flatnonzero(~valid_onnx)
    for i in bad[:12]:
        s = float("nan") if onnx_scores is None else float(onnx_scores[i])
        print(
            f"  i={i:4d} xy=({onnx_pts[i,0]:8.3f},"
            f"{onnx_pts[i,1]:8.3f}) score={s:.6f}"
        )


if __name__ == "__main__":
    main()
