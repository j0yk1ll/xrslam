#!/usr/bin/env python3
"""Check descriptor equivalence between sparse and dense XFeat ONNX models.

This diagnostic loads the same image through:
  1. the existing sparse xfeat.onnx model, and
  2. the dense xfeat_dense.onnx model used for track-conditioned recovery.

It samples the dense descriptor map at the sparse model's own keypoints using
the official XFeat InterpolateSparse2d/grid_sample convention, then compares
those descriptors against the sparse model's descriptor outputs.

It also reproduces XRSLAM's C++ bicubic sampler in Python and compares it
against torch.grid_sample, isolating export-vs-sampler errors.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-model", type=Path, required=True)
    parser.add_argument("--dense-model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument(
        "--euroc",
        type=Path,
        help="EuRoC mav0 directory; use with --frame-index.",
    )
    parser.add_argument("--frame-index", type=int, default=308)
    parser.add_argument("--limit", type=int, default=512)
    return parser.parse_args()


def resolve_image(args):
    if args.image is not None:
        return args.image

    csv_path = args.euroc / "cam0" / "data.csv"
    rows = [
        line.strip()
        for line in csv_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.frame_index < 0 or args.frame_index >= len(rows):
        raise SystemExit(
            f"frame index {args.frame_index} outside [0,{len(rows)-1}]"
        )
    filename = rows[args.frame_index].split(",", 1)[1].strip()
    return args.euroc / "cam0" / "data" / filename


def make_input(image_path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit(f"cannot read {image_path}")

    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    else:
        raise SystemExit(f"unsupported image shape {bgr.shape}")

    tensor = (
        rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    )
    return tensor, rgb.shape[0], rgb.shape[1]


def run_session(path, image):
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0].name
    outputs = session.run(None, {inp: image})
    return session, outputs


def identify_sparse_outputs(session, outputs):
    keypoints = descriptors = scores = None
    for meta, value in zip(session.get_outputs(), outputs):
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            keypoints = arr.reshape(-1, 2)
        elif arr.ndim >= 2 and arr.shape[-1] == 64:
            descriptors = arr.reshape(-1, 64)
        elif arr.size > 0:
            scores = arr.reshape(-1)

    if keypoints is None or descriptors is None:
        shapes = [
            (m.name, np.asarray(v).shape)
            for m, v in zip(session.get_outputs(), outputs)
        ]
        raise SystemExit(f"cannot identify sparse outputs: {shapes}")

    count = min(len(keypoints), len(descriptors))
    if scores is not None:
        count = min(count, len(scores))
    return (
        keypoints[:count].astype(np.float32),
        descriptors[:count].astype(np.float32),
        None if scores is None else scores[:count].astype(np.float32),
    )


def identify_dense_output(session, outputs):
    candidates = []
    for meta, value in zip(session.get_outputs(), outputs):
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 64:
            candidates.append((meta.name, arr.astype(np.float32)))
    if len(candidates) != 1:
        shapes = [
            (m.name, np.asarray(v).shape)
            for m, v in zip(session.get_outputs(), outputs)
        ]
        raise SystemExit(f"cannot identify unique dense output: {shapes}")
    return candidates[0]


def official_sample(dense, keypoints, image_h, image_w):
    resized_h = (image_h // 32) * 32
    resized_w = (image_w // 32) * 32

    pts = keypoints.copy()
    pts[:, 0] *= resized_w / image_w
    pts[:, 1] *= resized_h / image_h

    # Official InterpolateSparse2d:
    # grid = 2 * pos / [W-1,H-1] - 1
    grid = pts.copy()
    grid[:, 0] = 2.0 * grid[:, 0] / (resized_w - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (resized_h - 1) - 1.0

    dense_t = torch.from_numpy(dense)
    grid_t = torch.from_numpy(grid).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        dense_t,
        grid_t,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.permute(0, 2, 3, 1).squeeze(0).squeeze(1)
    sampled = F.normalize(sampled, p=2, dim=-1)
    return sampled.numpy()


def cubic_weight(x):
    alpha = -0.75
    x = abs(float(x))
    if x <= 1.0:
        return (alpha + 2.0) * x**3 - (alpha + 3.0) * x**2 + 1.0
    if x < 2.0:
        return alpha * x**3 - 5.0 * alpha * x**2 + 8.0 * alpha * x - 4.0 * alpha
    return 0.0


def cpp_equivalent_sample(dense, keypoints, image_h, image_w):
    _, channels, feat_h, feat_w = dense.shape
    resized_h = (image_h // 32) * 32
    resized_w = (image_w // 32) * 32

    result = np.zeros((len(keypoints), channels), dtype=np.float32)
    for i, p in enumerate(keypoints):
        resized_x = float(p[0]) * resized_w / image_w
        resized_y = float(p[1]) * resized_h / image_h

        fx = resized_x * feat_w / (resized_w - 1) - 0.5
        fy = resized_y * feat_h / (resized_h - 1) - 0.5
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))

        out = result[i]
        for oy in range(-1, 3):
            sy = y0 + oy
            wy = cubic_weight(fy - sy)
            if wy == 0.0 or sy < 0 or sy >= feat_h:
                continue
            for ox in range(-1, 3):
                sx = x0 + ox
                wx = cubic_weight(fx - sx)
                if wx == 0.0 or sx < 0 or sx >= feat_w:
                    continue
                out += np.float32(wx * wy) * dense[0, :, sy, sx]

        norm = np.linalg.norm(out)
        if norm > 1e-12:
            out /= norm
    return result


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def summarize(name, cos):
    q = np.quantile(cos, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(
        f"{name}: n={len(cos)} mean={cos.mean():.6f} "
        f"std={cos.std():.6f}"
    )
    print(
        "  cosine "
        f"min={q[0]:.6f} p01={q[1]:.6f} p05={q[2]:.6f} "
        f"median={q[3]:.6f} p95={q[4]:.6f} "
        f"p99={q[5]:.6f} max={q[6]:.6f}"
    )
    print(
        f"  >=0.999: {(cos >= 0.999).mean()*100:.1f}%  "
        f">=0.99: {(cos >= 0.99).mean()*100:.1f}%  "
        f">=0.95: {(cos >= 0.95).mean()*100:.1f}%"
    )


def main():
    args = parse_args()
    image_path = resolve_image(args)
    image, image_h, image_w = make_input(image_path)
    print(f"image: {image_path}")
    print(f"input: {image_w}x{image_h}")

    sparse_session, sparse_outputs = run_session(
        args.sparse_model, image
    )
    keypoints, sparse_desc, scores = identify_sparse_outputs(
        sparse_session, sparse_outputs
    )

    dense_session, dense_outputs = run_session(
        args.dense_model, image
    )
    dense_name, dense = identify_dense_output(
        dense_session, dense_outputs
    )

    count = min(len(keypoints), args.limit)
    keypoints = keypoints[:count]
    sparse_desc = normalize_rows(sparse_desc[:count])

    print(f"sparse features: {len(keypoints)} used")
    print(f"dense output: {dense_name} {tuple(dense.shape)}")
    print(
        f"preprocessed image: {(image_w//32)*32}x{(image_h//32)*32}"
    )

    official = official_sample(
        dense, keypoints, image_h, image_w
    )
    cpp = cpp_equivalent_sample(
        dense, keypoints, image_h, image_w
    )

    cos_sparse_official = np.sum(sparse_desc * official, axis=1)
    cos_official_cpp = np.sum(official * cpp, axis=1)
    cos_sparse_cpp = np.sum(sparse_desc * cpp, axis=1)

    summarize("sparse ONNX vs dense+official sampling", cos_sparse_official)
    summarize("official sampling vs XRSLAM C++ equivalent", cos_official_cpp)
    summarize("sparse ONNX vs XRSLAM C++ equivalent", cos_sparse_cpp)

    worst = np.argsort(cos_sparse_cpp)[:10]
    print("worst sparse-vs-C++ points:")
    for i in worst:
        score = float("nan") if scores is None else float(scores[i])
        print(
            f"  i={i:4d} xy=({keypoints[i,0]:8.3f},"
            f"{keypoints[i,1]:8.3f}) score={score:.6f} "
            f"cos={cos_sparse_cpp[i]:.6f}"
        )


if __name__ == "__main__":
    main()
