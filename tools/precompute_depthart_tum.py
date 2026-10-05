#!/usr/bin/env python3
"""Precompute DepthART metric depth for a TUM RGB-D style rgb.txt sequence."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_rgb_index(dataset: Path):
    entries = []
    with (dataset / "rgb.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp, relative = line.split(maxsplit=1)
            entries.append((float(timestamp), relative))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depthart", type=Path, required=True,
                        help="DepthART repository root")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="TUM sequence containing rgb.txt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=("S", "B", "L"), required=True)
    parser.add_argument("--domain", choices=("indoor", "outdoor"),
                        default="indoor")
    parser.add_argument("--intrinsics", nargs=4, type=float, required=True,
                        metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metric_root = args.depthart.resolve() / "metric"
    sys.path.insert(0, str(metric_root))

    from common import make_K, preprocess
    from model import load_model
    from network import tvimblock

    extension_root = args.depthart.resolve() / "deploy/shared/selective_scan"
    sys.path.insert(0, str(extension_root))
    try:
        import depthart_selective_scan_cuda  # noqa: F401
        from depthart_selective_scan import install_depthart
        install_depthart(tvimblock)
        optimized = True
    except ImportError:
        optimized = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(args.checkpoint), args.encoder, args.domain, device)

    output_dir = args.output_dir or (args.dataset / "depthart")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_rgb_index(args.dataset)
    print(f"DepthART device={device}, frames={len(entries)}, "
          f"selective_scan={'optimized' if optimized else 'reference'}")

    for index, (_, relative_rgb) in enumerate(entries, start=1):
        rgb_path = args.dataset / relative_rgb
        output_path = output_dir / (Path(relative_rgb).stem + ".npy")
        if output_path.exists() and not args.overwrite:
            continue

        image = cv2.imread(str(rgb_path))
        if image is None:
            raise FileNotFoundError(rgb_path)

        raw_height, raw_width = image.shape[:2]
        target_height = args.height or (
            480 if args.domain == "indoor" else 448)
        target_width = args.width or (
            640 if args.domain == "indoor" else 448)

        tensor, K = preprocess(
            image, make_K(*args.intrinsics), target_width, target_height)

        with torch.inference_mode():
            prediction = model(tensor.to(device), K.to(device))[:, None]
            prediction = F.interpolate(
                prediction,
                (raw_height, raw_width),
                mode="bilinear",
                align_corners=True,
            )[0, 0]

        depth = prediction.float().cpu().numpy().astype(
            np.float32, copy=False)
        np.save(output_path, depth)

        if index == 1 or index % 100 == 0 or index == len(entries):
            print(f"[{index}/{len(entries)}] {output_path.name}")


if __name__ == "__main__":
    main()
