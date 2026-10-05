#!/usr/bin/env python3
"""Export official XFeat sparse features to a fixed-resolution ONNX model.

This exporter uses the official verlab/accelerated_features network and
weights, but rewrites two export-hostile implementation details:

  * XFeatModel._unfold2d() is expressed as reshape/permute instead of
    Tensor.unfold().
  * sparse NMS/post-processing uses fixed-size TopK tensors instead of a
    data-dependent NonZero tensor followed by reshapes.

For the configured camera resolution the resulting graph returns exactly
top_k keypoints, descriptors, and scores and has no variable-size internal
NMS tensors.

Outputs match XRSLAM's existing XFeat backend contract:
    keypoints   [top_k,2]  float32 original-image pixel coordinates
    descriptors [top_k,64] float32 L2-normalized
    scores      [top_k]    float32 official XFeat reliability score
"""

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _onnx_safe_unfold2d(self, x, ws=2):
    """Exact XFeat non-overlapping unfold without Tensor.unfold()."""
    batch, channels, height, width = x.shape
    x = x[
        ...,
        : (height // ws) * ws,
        : (width // ws) * ws,
    ]
    batch, channels, height, width = x.shape
    return (
        x.reshape(
            batch,
            channels,
            height // ws,
            ws,
            width // ws,
            ws,
        )
        .permute(0, 1, 3, 5, 2, 4)
        .flatten(1, 3)
    )


class OfficialSparseXFeatExport(torch.nn.Module):
    def __init__(
        self,
        net,
        input_height,
        input_width,
        top_k,
        candidate_k,
        detection_threshold=0.05,
    ):
        super().__init__()
        self.net = net
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.resized_height = (self.input_height // 32) * 32
        self.resized_width = (self.input_width // 32) * 32
        self.top_k = int(top_k)
        self.candidate_k = int(candidate_k)
        self.detection_threshold = float(detection_threshold)

        pixel_count = self.resized_height * self.resized_width
        if self.top_k <= 0 or self.top_k > pixel_count:
            raise ValueError("top_k outside image pixel count")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if self.candidate_k > pixel_count:
            raise ValueError("candidate_k exceeds image pixel count")

    @staticmethod
    def _sample(feature_map, points, height, width, mode):
        """Official InterpolateSparse2d coordinate convention."""
        denom = torch.tensor(
            [width - 1, height - 1],
            dtype=points.dtype,
            device=points.device,
        )
        grid = (2.0 * (points / denom) - 1.0).unsqueeze(-2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).squeeze(-2)

    @staticmethod
    def _heatmap(keypoint_logits):
        scores = F.softmax(keypoint_logits, dim=1)[:, :64]
        batch, _, height, width = scores.shape
        return (
            scores.permute(0, 2, 3, 1)
            .reshape(batch, height, width, 8, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, 1, height * 8, width * 8)
        )

    def forward(self, images):
        # Equivalent to official XFeat.preprocess_tensor(), specialized to the
        # fixed camera geometry used for this deployment experiment.
        images = F.interpolate(
            images,
            (self.resized_height, self.resized_width),
            mode="bilinear",
            align_corners=False,
        )

        dense, keypoint_logits, reliability = self.net(images)
        dense = F.normalize(dense, p=2, dim=1)

        heatmap = self._heatmap(keypoint_logits)

        # Official NMS: local maximum with kernel 5 and detection threshold.
        local_max = F.max_pool2d(
            heatmap, kernel_size=5, stride=1, padding=2
        )
        keep = (heatmap == local_max) & (
            heatmap > self.detection_threshold
        )

        # Avoid NonZero/data-dependent shapes. candidate_k is deliberately
        # larger than the expected number of NMS maxima; invalid filler pixels
        # carry a very negative preliminary score and are discarded when the
        # final official reliability score is ranked.
        prelim = torch.where(
            keep,
            heatmap,
            torch.full_like(heatmap, -1.0e6),
        ).reshape(1, -1)

        prelim_values, flat_indices = torch.topk(
            prelim, k=self.candidate_k, dim=1, largest=True, sorted=True
        )

        candidate_y = torch.div(
            flat_indices,
            self.resized_width,
            rounding_mode="floor",
        )
        candidate_x = flat_indices - candidate_y * self.resized_width
        candidates = torch.stack(
            [candidate_x, candidate_y], dim=-1
        ).to(dense.dtype)

        # Official reliability score:
        # nearest(K1 heatmap) * bilinear(reliability head)
        detector_score = self._sample(
            heatmap,
            candidates,
            self.resized_height,
            self.resized_width,
            "nearest",
        ).squeeze(-1)
        reliability_score = self._sample(
            reliability,
            candidates,
            self.resized_height,
            self.resized_width,
            "bilinear",
        ).squeeze(-1)
        scores = detector_score * reliability_score

        valid_candidate = prelim_values > -1.0e5
        scores = torch.where(
            valid_candidate,
            scores,
            torch.full_like(scores, -1.0e6),
        )

        final_scores, final_order = torch.topk(
            scores, k=self.top_k, dim=1, largest=True, sorted=True
        )
        gather_xy = final_order.unsqueeze(-1).expand(-1, -1, 2)
        keypoints = torch.gather(candidates, 1, gather_xy)

        # Official descriptor interpolation and final L2 normalization.
        descriptors = self._sample(
            dense,
            keypoints,
            self.resized_height,
            self.resized_width,
            "bicubic",
        )
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Scale keypoints from the divisible-by-32 working image back to the
        # original camera resolution, exactly as detectAndCompute().
        scale = torch.tensor(
            [
                self.input_width / self.resized_width,
                self.input_height / self.resized_height,
            ],
            dtype=keypoints.dtype,
            device=keypoints.device,
        )
        keypoints = keypoints * scale.view(1, 1, 2)

        return (
            keypoints.squeeze(0),
            descriptors.squeeze(0),
            final_scores.squeeze(0),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--xfeat",
        type=Path,
        required=True,
        help="Path to verlab/accelerated_features.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path.",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional xfeat.pt; defaults to official repository weights.",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=752)
    p.add_argument("--top-k", type=int, default=512)
    p.add_argument(
        "--candidate-k",
        type=int,
        default=16384,
        help=(
            "Fixed NMS candidate pool before official reliability reranking. "
            "Must exceed the number of local maxima on the target camera."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.xfeat.resolve()))

    from modules.xfeat import XFeat

    weights = (
        str(args.weights.resolve())
        if args.weights is not None
        else None
    )
    xfeat = (
        XFeat(weights=weights, top_k=args.top_k)
        if weights
        else XFeat(top_k=args.top_k)
    )
    xfeat.eval()

    # Replace only the export-hostile Tensor.unfold implementation. Network
    # parameters and all learned operations remain official XFeat.
    xfeat.net._unfold2d = types.MethodType(
        _onnx_safe_unfold2d, xfeat.net
    )

    model = OfficialSparseXFeatExport(
        xfeat.net,
        input_height=args.height,
        input_width=args.width,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    ).eval()

    dummy = torch.zeros(
        1, 3, args.height, args.width, dtype=torch.float32,
        device=xfeat.dev,
    )

    # Run once before export so shape/candidate mistakes fail in PyTorch rather
    # than after a successful-looking ONNX export.
    with torch.inference_mode():
        keypoints, descriptors, scores = model(dummy)
    if keypoints.shape != (args.top_k, 2):
        raise RuntimeError(
            f"unexpected keypoint shape: {tuple(keypoints.shape)}"
        )
    if descriptors.shape != (args.top_k, 64):
        raise RuntimeError(
            f"unexpected descriptor shape: {tuple(descriptors.shape)}"
        )
    if scores.shape != (args.top_k,):
        raise RuntimeError(
            f"unexpected score shape: {tuple(scores.shape)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    print(
        f"wrote {args.output} "
        f"(fixed input 1x3x{args.height}x{args.width}, "
        f"top_k={args.top_k}, candidate_k={args.candidate_k})"
    )


if __name__ == "__main__":
    main()
