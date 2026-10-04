#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/evaluate_trajectory.sh <mode> <ground_truth.tum> <estimate.tum> [evo_ape args...]

Modes:
  monocular       visual-only RGB; Sim(3) alignment (SE3 + scale)
  monocular_imu   RGB + IMU; SE(3) alignment
  rgbd            RGB-D; SE(3) alignment
  rgbd_imu        RGB-D + IMU; SE(3) alignment

This wrapper deliberately derives alignment from observability rather than
from dataset name. Learned depth should use SE(3) only when its scale is
actually metric; otherwise evaluate it as monocular/Sim(3).
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

mode=$1
gt=$2
estimate=$3
shift 3

case "$mode" in
  monocular)
    # evo: --align estimates an SE(3) alignment; --correct_scale adds the
    # single scale degree of freedom, yielding the required Sim(3) alignment.
    alignment=(--align --correct_scale)
    ;;
  monocular_imu|rgbd|rgbd_imu)
    alignment=(--align)
    ;;
  *)
    echo "error: unsupported mode '$mode'" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -f "$gt" ]]; then
  echo "error: ground truth not found: $gt" >&2
  exit 2
fi

if [[ ! -f "$estimate" ]]; then
  echo "error: estimate not found: $estimate" >&2
  exit 2
fi

exec evo_ape tum "$gt" "$estimate" "${alignment[@]}" "$@"
