#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch


LOG_RE = re.compile(
    r"^\[PlaceKeyframeShadow\] "
    r"frame_id=(\d+) "
    r"t=([0-9eE+\-.]+) "
    r"p=([0-9eE+\-.,]+) "
    r"q=([0-9eE+\-.,]+) "
    r"active=(.*)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline EigenPlaces + USearch shadow retrieval benchmark."
    )
    parser.add_argument("--euroc-mav0", required=True, type=Path)
    parser.add_argument("--keyframe-log", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("configs/euroc_sensor.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/place-retrieval-topk.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/tmp/place-retrieval-summary.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend",
        choices=("usearch", "exact", "both"),
        default="both",
        help="Use USearch, exact NumPy L2, or both (default: both).",
    )
    return parser.parse_args()


def parse_vector(text):
    return np.asarray([float(x) for x in text.split(",")], dtype=np.float64)


def load_keyframes(path):
    records = []
    with path.open() as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            match = LOG_RE.match(line)
            if not match:
                raise RuntimeError(
                    f"{path}:{line_no}: unrecognized PlaceKeyframeShadow line"
                )
            frame_id = int(match.group(1))
            timestamp = float(match.group(2))
            p = parse_vector(match.group(3))
            q = parse_vector(match.group(4))
            active = {
                int(x) for x in match.group(5).split(";") if x.strip()
            }
            if p.shape != (3,) or q.shape != (4,):
                raise RuntimeError(
                    f"{path}:{line_no}: malformed pose dimensions"
                )
            records.append(
                {
                    "frame_id": frame_id,
                    "t": timestamp,
                    "p_est": p,
                    "q_est": q,
                    "active": active,
                }
            )

    ids = [r["frame_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("keyframe log contains duplicate frame IDs")
    if not records:
        raise RuntimeError("keyframe log is empty")
    return records



def parse_sensor_config(path):
    text = path.read_text()

    def sequence(name):
        match = re.search(
            rf"^\s*{re.escape(name)}:\s*\[([^\]]+)\]",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"cannot find '{name}' in {path}")
        return [float(x.strip()) for x in match.group(1).split(",")]

    intrinsics = sequence("intrinsics")
    distortion = sequence("distortion")
    if len(intrinsics) != 4 or len(distortion) < 4:
        raise RuntimeError("unexpected EuRoC camera calibration dimensions")

    fx, fy, cx, cy = intrinsics
    K = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    D = np.asarray(distortion[:4], dtype=np.float32)
    return K, D


def load_camera_index(path):
    items = []
    with path.open(newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        for row in reader:
            if len(row) < 2:
                continue
            items.append((int(row[0]), row[1]))
    if not items:
        raise RuntimeError(f"no camera records in {path}")
    items.sort()
    timestamps = np.asarray([x[0] for x in items], dtype=np.int64)
    filenames = [x[1] for x in items]
    return timestamps, filenames


def nearest_camera_file(timestamp_s, timestamps_ns, filenames):
    target = int(round(timestamp_s * 1e9))
    pos = int(np.searchsorted(timestamps_ns, target))
    candidates = []
    if pos < len(timestamps_ns):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best = min(candidates, key=lambda i: abs(int(timestamps_ns[i]) - target))
    delta_s = abs(int(timestamps_ns[best]) - target) * 1e-9
    if delta_s > 0.005:
        raise RuntimeError(
            f"no camera image within 5 ms of t={timestamp_s:.9f}; "
            f"nearest delta={delta_s:.6f}s"
        )
    return filenames[best], int(timestamps_ns[best]) * 1e-9


def load_ground_truth(path):
    times = []
    positions = []
    with path.open(newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        for row in reader:
            if len(row) < 4:
                continue
            times.append(int(row[0]) * 1e-9)
            positions.append([float(row[1]), float(row[2]), float(row[3])])
    if not times:
        raise RuntimeError(f"no ground-truth records in {path}")
    return np.asarray(times), np.asarray(positions, dtype=np.float64)


def interpolate_positions(query_times, gt_times, gt_positions):
    result = np.empty((len(query_times), 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(
            query_times, gt_times, gt_positions[:, axis]
        )
    return result


class EigenPlaces:
    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

    @torch.inference_mode()
    def describe(self, gray):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch.float32)
            / 255.0
        )
        tensor = (tensor - self.mean) / self.std
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        descriptor = output.reshape(-1).to(torch.float32)
        descriptor = descriptor / torch.linalg.vector_norm(descriptor)
        result = descriptor.cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (512,):
            raise RuntimeError(
                f"unexpected EigenPlaces descriptor shape {result.shape}"
            )
        return result


def make_usearch():
    try:
        from usearch.index import Index
    except ImportError as exc:
        raise RuntimeError(
            "Python USearch bindings are not installed. Install the pinned "
            "local checkout with:\n"
            "  python -m pip install /home/kevin/Code/xrslam-deps/USearch\n"
            "or run once with --backend exact to validate descriptor generation."
        ) from exc
    return Index(ndim=512, metric="l2sq", dtype="f32")


def exact_ranking(history_ids, history_desc, active_ids, query):
    if not history_ids:
        return []
    ids = np.asarray(history_ids, dtype=np.int64)
    matrix = np.stack(history_desc)
    keep = np.asarray([int(x) not in active_ids for x in ids], dtype=bool)
    ids = ids[keep]
    matrix = matrix[keep]
    if len(ids) == 0:
        return []
    distances = np.sum((matrix - query[None, :]) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")
    return [(int(ids[i]), float(distances[i])) for i in order]


def usearch_ranking(index, history_count, active_ids, query, top_k):
    if history_count == 0:
        return []
    wanted = min(history_count, top_k + len(active_ids))
    if wanted <= 0:
        return []
    matches = index.search(query, wanted)
    ranking = []
    for match in matches:
        key = int(match.key)
        if key in active_ids:
            continue
        ranking.append((key, float(match.distance)))
        if len(ranking) >= top_k:
            break
    return ranking


def overlap_at_k(a, b, k):
    a_ids = [x[0] for x in a[:k]]
    b_ids = [x[0] for x in b[:k]]
    denom = min(k, len(b_ids))
    if denom == 0:
        return math.nan
    return len(set(a_ids) & set(b_ids)) / denom


def finite_mean(values):
    values = [x for x in values if math.isfinite(x)]
    return float(np.mean(values)) if values else math.nan


def build_coactive_neighbors(records):
    """Map each frame ID to frames observed in the same active SLAM window."""
    neighbors = {}
    for record in records:
        active = record["active"]
        for frame_id in active:
            neighbors.setdefault(frame_id, set()).update(active - {frame_id})
    return neighbors


def local_exclusion_set(active_ids, coactive_neighbors):
    """Exclude current active frames plus their direct active-window neighbors."""
    excluded = set(active_ids)
    for frame_id in active_ids:
        excluded.update(coactive_neighbors.get(frame_id, ()))
    return excluded


def main():
    args = parse_args()
    if args.top_k < 1:
        raise RuntimeError("--top-k must be >= 1")

    records = load_keyframes(args.keyframe_log)
    coactive_neighbors = build_coactive_neighbors(records)
    K, D = parse_sensor_config(args.sensor_config)

    camera_csv = args.euroc_mav0 / "cam0" / "data.csv"
    camera_dir = args.euroc_mav0 / "cam0" / "data"
    gt_csv = (
        args.euroc_mav0
        / "state_groundtruth_estimate0"
        / "data.csv"
    )
    camera_times_ns, camera_filenames = load_camera_index(camera_csv)
    gt_times, gt_positions = load_ground_truth(gt_csv)

    query_times = np.asarray([r["t"] for r in records], dtype=np.float64)
    interpolated_gt = interpolate_positions(query_times, gt_times, gt_positions)
    for record, p_gt in zip(records, interpolated_gt):
        record["p_gt"] = p_gt

    logged_ids = {r["frame_id"] for r in records}
    active_ids = set().union(*(r["active"] for r in records))
    active_only_ids = sorted(active_ids - logged_ids)

    extractor = EigenPlaces(args.model, args.device)
    usearch_index = (
        make_usearch() if args.backend in ("usearch", "both") else None
    )

    print(f"keyframes logged: {len(records)}")
    print(f"active-only frame IDs not logged: {len(active_only_ids)}")
    if active_only_ids:
        print(
            "active-only sample:",
            ",".join(str(x) for x in active_only_ids[:20]),
        )

    descriptors = {}
    image_times = {}
    for i, record in enumerate(records, 1):
        filename, image_t = nearest_camera_file(
            record["t"], camera_times_ns, camera_filenames
        )
        gray = cv2.imread(str(camera_dir / filename), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read {camera_dir / filename}")
        # Match the PC-player EuRoC path: EurocDatasetReader undistorts the
        # camera image before XRSLAMPushSensorData(), and XRSLAMManager then
        # stores that already-undistorted image in OpenCvImage::raw.
        gray = cv2.undistort(gray, K, D)
        descriptor = extractor.describe(gray)
        descriptors[record["frame_id"]] = descriptor
        image_times[record["frame_id"]] = image_t
        if i == 1 or i % 50 == 0 or i == len(records):
            print(f"descriptors: {i}/{len(records)}")

    rows = []
    history_ids = []
    history_desc = []
    id_to_record = {r["frame_id"]: r for r in records}

    overlap = {1: [], 5: [], 10: [], 20: []}
    gt_nearest_hit = {1: [], 5: [], 10: [], 20: []}
    best_gt_distance_in_top_k = {1: [], 5: [], 10: [], 20: []}
    gt_distance_excess_over_oracle = {1: [], 5: [], 10: [], 20: []}
    queries_with_k_candidates = {1: 0, 5: 0, 10: 0, 20: 0}
    gt_nearest_descriptor_ranks = []
    gt_nearest_distances = []
    gt_nearest_temporal_separations = []
    local_exclusion_sizes = []
    evaluated_queries = 0

    for query_record in records:
        qid = query_record["frame_id"]
        query = descriptors[qid]
        active = query_record["active"]
        local_excluded = local_exclusion_set(active, coactive_neighbors)
        local_exclusion_sizes.append(len(local_excluded))

        exact = exact_ranking(
            history_ids, history_desc, local_excluded, query
        )
        ann = (
            usearch_ranking(
                usearch_index,
                len(history_ids),
                local_excluded,
                query,
                args.top_k,
            )
            if usearch_index is not None
            else []
        )

        primary = ann if usearch_index is not None else exact[: args.top_k]

        if exact:
            evaluated_queries += 1
            eligible_ids = [x[0] for x in exact]
            gt_distances = [
                (
                    cid,
                    float(
                        np.linalg.norm(
                            query_record["p_gt"]
                            - id_to_record[cid]["p_gt"]
                        )
                    ),
                )
                for cid in eligible_ids
            ]
            gt_nearest_id, gt_nearest_distance = min(
                gt_distances, key=lambda x: x[1]
            )
            gt_nearest_distances.append(gt_nearest_distance)
            gt_nearest_temporal_separations.append(
                query_record["t"] - id_to_record[gt_nearest_id]["t"]
            )
            exact_rank_map = {
                cid: rank + 1 for rank, (cid, _) in enumerate(exact)
            }
            gt_rank = exact_rank_map[gt_nearest_id]
            gt_nearest_descriptor_ranks.append(gt_rank)

            for k in (1, 5, 10, 20):
                if usearch_index is not None:
                    overlap[k].append(overlap_at_k(ann, exact, k))
                    gt_nearest_hit[k].append(
                        float(gt_nearest_id in [x[0] for x in ann[:k]])
                    )
                    top_k_candidates = ann[:k]
                else:
                    gt_nearest_hit[k].append(float(gt_rank <= k))
                    top_k_candidates = exact[:k]

                if len(top_k_candidates) >= k:
                    queries_with_k_candidates[k] += 1
                    best_distance = min(
                        float(
                            np.linalg.norm(
                                query_record["p_gt"]
                                - id_to_record[cid]["p_gt"]
                            )
                        )
                        for cid, _ in top_k_candidates
                    )
                    best_gt_distance_in_top_k[k].append(best_distance)
                    gt_distance_excess_over_oracle[k].append(
                        best_distance - gt_nearest_distance
                    )

        exact_rank_map = {
            cid: rank + 1 for rank, (cid, _) in enumerate(exact)
        }
        for rank, (cid, distance) in enumerate(primary[: args.top_k], 1):
            candidate = id_to_record[cid]
            rows.append(
                {
                    "query_frame_id": qid,
                    "query_t": f"{query_record['t']:.9f}",
                    "candidate_frame_id": cid,
                    "rank": rank,
                    "descriptor_l2": f"{math.sqrt(max(distance, 0.0)):.9g}",
                    "exact_descriptor_rank": exact_rank_map.get(cid, ""),
                    "temporal_separation_s": f"{query_record['t'] - candidate['t']:.9f}",
                    "estimated_spatial_separation_m": f"{np.linalg.norm(query_record['p_est'] - candidate['p_est']):.9g}",
                    "gt_spatial_separation_m": f"{np.linalg.norm(query_record['p_gt'] - candidate['p_gt']):.9g}",
                }
            )

        if usearch_index is not None:
            usearch_index.add(qid, query)
        history_ids.append(qid)
        history_desc.append(query)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = [
            "query_frame_id",
            "query_t",
            "candidate_frame_id",
            "rank",
            "descriptor_l2",
            "exact_descriptor_rank",
            "temporal_separation_s",
            "estimated_spatial_separation_m",
            "gt_spatial_separation_m",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def distribution(values):
        if not values:
            return {
                "count": 0,
                "median": math.nan,
                "p90": math.nan,
            }
        return {
            "count": len(values),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }

    summary = {
        "keyframes_logged": len(records),
        "active_only_frame_ids_not_logged": len(active_only_ids),
        "active_only_ids": active_only_ids,
        "evaluated_queries": evaluated_queries,
        "backend": args.backend,
        "index_metric": "l2sq",
        "reported_descriptor_distance": "l2",
        "image_preprocessing": "euroc_reader_undistort_then_grayscale_to_rgb_imagenet_normalization",
        "local_exclusion_policy": "current_active_plus_direct_coactive_neighbors",
        "local_exclusion_size": {
            "median": float(np.median(local_exclusion_sizes))
            if local_exclusion_sizes
            else math.nan,
            "p90": float(np.percentile(local_exclusion_sizes, 90))
            if local_exclusion_sizes
            else math.nan,
        },
        "top_k": args.top_k,
        "gt_nearest_distance_m": {
            "median": float(np.median(gt_nearest_distances))
            if gt_nearest_distances
            else math.nan,
            "p90": float(np.percentile(gt_nearest_distances, 90))
            if gt_nearest_distances
            else math.nan,
        },
        "gt_nearest_descriptor_rank": {
            "median": float(np.median(gt_nearest_descriptor_ranks))
            if gt_nearest_descriptor_ranks
            else math.nan,
            "p90": float(np.percentile(gt_nearest_descriptor_ranks, 90))
            if gt_nearest_descriptor_ranks
            else math.nan,
        },
        "gt_nearest_temporal_separation_s": {
            "median": float(np.median(gt_nearest_temporal_separations))
            if gt_nearest_temporal_separations
            else math.nan,
            "p10": float(np.percentile(gt_nearest_temporal_separations, 10))
            if gt_nearest_temporal_separations
            else math.nan,
        },
        "strict_gt_nearest_in_top_k": {
            str(k): finite_mean(gt_nearest_hit[k]) for k in (1, 5, 10, 20)
        },
        "best_gt_distance_in_top_k_m": {
            str(k): distribution(best_gt_distance_in_top_k[k])
            for k in (1, 5, 10, 20)
        },
        "gt_distance_excess_over_oracle_m": {
            str(k): distribution(gt_distance_excess_over_oracle[k])
            for k in (1, 5, 10, 20)
        },
        "queries_with_k_candidates": {
            str(k): queries_with_k_candidates[k] for k in (1, 5, 10, 20)
        },
    }
    if usearch_index is not None:
        summary["usearch_exact_overlap_at_k"] = {
            str(k): finite_mean(overlap[k]) for k in (1, 5, 10, 20)
        }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"top-K CSV: {args.output}")
    print(f"summary:   {args.summary}")
    print(
        "No descriptor-distance, GT-distance, frame-age, or time-age "
        "acceptance threshold was applied."
    )
    print(
        "best_gt_distance_in_top_k_m measures the closest GT candidate "
        "actually retrieved within each K; gt_distance_excess_over_oracle_m "
        "measures how much farther it is than the nearest eligible GT frame."
    )


if __name__ == "__main__":
    main()
