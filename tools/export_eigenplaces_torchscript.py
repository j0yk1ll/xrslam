#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a pretrained EigenPlaces model to TorchScript"
    )
    parser.add_argument("--eigenplaces-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backbone", default="ResNet50")
    parser.add_argument("--dim", default=512, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=752, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def validate(module, device, dim, shapes):
    with torch.inference_mode():
        for height, width in shapes:
            sample = torch.zeros(1, 3, height, width, device=device)
            descriptor = module(sample)
            if tuple(descriptor.shape) != (1, dim):
                raise RuntimeError(
                    f"unexpected output for {height}x{width}: "
                    f"{tuple(descriptor.shape)}"
                )
            norm = descriptor.norm(dim=1).item()
            if abs(norm - 1.0) > 1.0e-4:
                raise RuntimeError(
                    f"descriptor is not unit normalized: norm={norm}"
                )
            print(
                f"validated input=1x3x{height}x{width} "
                f"output={tuple(descriptor.shape)} norm={norm:.9f}"
            )


def main():
    args = parse_args()
    repo = args.eigenplaces_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not (repo / "hubconf.py").is_file():
        raise SystemExit(f"not an EigenPlaces checkout: {repo}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    device = torch.device(args.device)
    print(
        f"loading EigenPlaces backbone={args.backbone} "
        f"dim={args.dim} device={device}"
    )
    model = torch.hub.load(
        str(repo),
        "get_trained_model",
        source="local",
        backbone=args.backbone,
        fc_output_dim=args.dim,
    ).eval().to(device)

    example = torch.zeros(1, 3, args.height, args.width, device=device)

    export_method = "script"
    try:
        exported = torch.jit.script(model)
    except Exception as script_error:
        export_method = "trace"
        print(f"torch.jit.script failed, falling back to trace: {script_error}")
        exported = torch.jit.trace(model, example, strict=False)

    # Validate the EuRoC size and a second size. The second check prevents a
    # traced GeM pooling kernel from silently becoming tied to 480x752.
    validate(
        exported,
        device,
        args.dim,
        [(args.height, args.width), (320, 512)],
    )

    exported = exported.to("cpu")
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(output))

    print(f"export method: {export_method}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
