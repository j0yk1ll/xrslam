#!/usr/bin/env python3
"""Export the official XFeat dense 64-D descriptor map to ONNX.

The exported model mirrors XFeat.detectAndCompute() preprocessing: the input
is first resized to dimensions divisible by 32, then the backbone descriptor
map is produced and channel-normalized. XRSLAM samples this map at the
corresponding resized coordinates of its existing track locations.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class DenseDescriptorExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        # Keep this identical to XFeat.preprocess_tensor(). Sparse XFeat uses
        # this resize before running the backbone.
        height = images.shape[-2]
        width = images.shape[-1]
        resized_height = (height // 32) * 32
        resized_width = (width // 32) * 32
        images = F.interpolate(
            images,
            (resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

        descriptors, _, _ = self.net(images)
        return F.normalize(descriptors, p=2, dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xfeat",
        type=Path,
        required=True,
        help="Path to verlab/accelerated_features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional xfeat.pt; defaults to the official repository weight.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=752)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.xfeat.resolve()))

    from modules.xfeat import XFeat

    weights = (
        str(args.weights.resolve())
        if args.weights is not None
        else None
    )
    xfeat = XFeat(weights=weights) if weights else XFeat()
    model = DenseDescriptorExport(xfeat.net).eval()

    dummy = torch.zeros(
        1, 3, args.height, args.width, dtype=torch.float32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Opset 18 avoids the failed 18->17 converter path used by current
    # PyTorch/onnxscript. Use the legacy exporter here because dynamic_axes is
    # explicit and sufficient for this small deployment model.
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["descriptors"],
        dynamic_axes={
            "images": {2: "height", 3: "width"},
            "descriptors": {
                2: "descriptor_height",
                3: "descriptor_width",
            },
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
