#!/usr/bin/env python3
"""Export a *segmentation-finetuned* PIDNet checkpoint for XRSLAM/LibTorch.

The upstream ImageNet-only *.pth.tar is an initialization checkpoint, NOT a
complete semantic segmentation model. This exporter refuses incomplete heads.

The exported model is traced for exactly the fixed input canvas. XRSLAM's C++
backend uses the same canvas shape and letterboxing for source frames.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def checkpoint_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def unwrap_weights(blob):
    if not isinstance(blob, dict):
        raise RuntimeError('expected PyTorch checkpoint dictionary')
    for key in ('state_dict', 'model_state_dict'):
        if key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break
    cleaned = {}
    for key, value in blob.items():
        if not isinstance(key, str):
            raise RuntimeError('non-string checkpoint key')
        while key.startswith('module.') or key.startswith('model.'):
            key = key.split('.', 1)[1]
        if key in cleaned:
            raise RuntimeError(f'duplicate normalized checkpoint key: {key}')
        cleaned[key] = value
    return cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pidnet-repo', required=True, type=Path,
                    help='Directory of upstream XuJiacong/PIDNet clone')
    ap.add_argument('--checkpoint', required=True, type=Path,
                    help='Fine-tuned segmentation checkpoint, NOT ImageNet-only')
    ap.add_argument('--output', required=True, type=Path,
                    help='TorchScript .ts output')
    ap.add_argument('--variant', choices=('s', 'm', 'l'), default='s')
    ap.add_argument('--classes', type=int, default=19)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--atol', type=float, default=5e-4)
    args = ap.parse_args()
    if args.classes < 1 or args.classes > 255:
        ap.error('--classes must be 1..255')
    if args.width < 64 or args.height < 64 or args.width % 32 or args.height % 32:
        ap.error('--width/--height must be >=64 and divisible by 32')
    if args.output.suffix not in ('.ts', '.pt'):
        ap.error('--output must end with .ts or .pt')
    if not (args.pidnet_repo/'models'/'pidnet.py').is_file():
        ap.error('upstream PIDNet source not found at --pidnet-repo')
    if not args.checkpoint.is_file():
        ap.error('checkpoint does not exist')
    sys.path.insert(0, str(args.pidnet_repo.resolve()))

    import torch
    import models.pidnet as pidnet

    torch.set_grad_enabled(False)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch CUDA is unavailable')
    # Use the inference model without auxiliary heads, per upstream API.
    model = pidnet.get_pred_model('pidnet-' + args.variant, args.classes)
    # Use weights_only to avoid arbitrary pickle code for checkpoints. Recent
    # PyTorch versions support this argument; insist on it for safe loading.
    weights = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True)
    weights = unwrap_weights(weights)
    expected = model.state_dict()
    missing = sorted(k for k in expected if k not in weights)
    bad_shapes = sorted(k for k in expected if k in weights and
                        getattr(weights[k], 'shape', None) != expected[k].shape)
    if missing or bad_shapes:
        print('missing parameters:', missing[:20], file=sys.stderr)
        print('shape mismatches:', bad_shapes[:20], file=sys.stderr)
        raise RuntimeError('checkpoint lacks a complete fine-tuned segmentation model; '
                           'ImageNet-only initialization cannot be exported as segmentation')
    # The official PIDNet training FullModel can save auxiliary segmentation
    # heads and criterion state alongside the inference model. None is read by
    # get_pred_model(..., augment=False). Keep this allowlist deliberately
    # narrow: arbitrary unexpected checkpoint weights are still rejected.
    extras = sorted(k for k in weights if k not in expected)
    auxiliary_heads = [k for k in extras if
                       k.startswith('seghead_p.') or k.startswith('seghead_d.')]
    known_loss_state = frozenset((
        'sb_loss.criterion.weight',
        'sem_loss.criterion.weight',
    ))
    training_loss_state = [k for k in extras if k in known_loss_state]
    unexpected = [k for k in extras if
                  k not in auxiliary_heads and k not in training_loss_state]
    if unexpected:
        raise RuntimeError(f'unexpected checkpoint keys: {unexpected[:20]}')
    if training_loss_state:
        print('ignoring training-only criterion state:',
              ', '.join(training_loss_state))
    model.load_state_dict({k: weights[k] for k in expected}, strict=True)
    model = model.eval().to(args.device)

    example = torch.randn(1, 3, args.height, args.width, device=args.device)
    with torch.inference_mode():
        eager = model(example)
        if not isinstance(eager, torch.Tensor):
            raise RuntimeError('PIDNet inference output must be one Tensor')
        traced = torch.jit.trace(model, example, strict=True, check_trace=True)
        traced = torch.jit.freeze(traced.eval())
        traced_output = traced(example)
        torch.testing.assert_close(eager, traced_output,
                                   rtol=args.rtol, atol=args.atol)
        expected_shape = (1, args.classes, args.height // 8, args.width // 8)
        if tuple(eager.shape) != expected_shape:
            raise RuntimeError(f'output shape {tuple(eager.shape)} != {expected_shape}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    reloaded = torch.jit.load(str(args.output), map_location=args.device).eval()
    with torch.inference_mode():
        output = reloaded(example)
        torch.testing.assert_close(eager, output, rtol=args.rtol, atol=args.atol)
    manifest = {
        'format': 'xrslam-pidnet-torchscript-v1',
        'variant': 'pidnet-' + args.variant,
        'classes': args.classes,
        'input_shape': [1, 3, args.height, args.width],
        'output_shape': list(eager.shape),
        'checkpoint_sha256': checkpoint_hash(args.checkpoint),
        'input_color': 'RGB',
        'preprocessing': 'letterbox-then-imagenet-normalization',
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'logits_upsampling': 'bilinear_align_corners_true',
        'excluded_training_heads': auxiliary_heads,
        'excluded_training_criterion_keys': training_loss_state,
        'precision': 'float32',
    }
    args.output.with_suffix(args.output.suffix + '.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    max_difference = (eager - output).abs().max().item()
    print(f'PASS export: {args.output}, classes={args.classes} '
          f'input={args.width}x{args.height}, maximum_logit_abs_diff={max_difference:.8f}')
    print(f'manifest: {args.output}.json')


if __name__ == '__main__':
    main()
