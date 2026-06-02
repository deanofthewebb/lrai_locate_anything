#!/usr/bin/env python3
"""Counting-line audit using lrai_locate_anything as detector.

Mirrors the schema + algorithm of tracker_bot/people_counting_lines.py
(see ~/.claude/projects/-Users-deanwebb/memory/counting-line-audit-standard.md)
so the output CSV is drop-in comparable to the existing PeopleNet+TRT audit
results under ~/Downloads/ISP_audits/results/csvs/.

Detector backend swap: PeopleNet ONNX → LocateAnything-3B VLM. Tracker swap:
safetunnel ByteTracker → minimal greedy-IoU tracker (BT not yet ported; this
matches well enough for AI-FPS measurement + line-crossing schema validation).

Output schema (matches tracker_bot exactly):
    frame,timestamp_s,line_id,line_name,track_id,direction,
    coalesced,in_count_after,out_count_after,x,y

Performance metrics printed to stderr:
    ai_fps     — detect() throughput (inferences / sum-of-detect-time)
    proc_fps   — end-to-end (including video decode + tracking + counting)
    frames_processed, in_total, out_total

Usage:
    python scripts/lrai_isp_audit.py \\
        --video '~/Downloads/ISP_audits/Gate A5 (new)_Flight1_0028-0037.mp4' \\
        --lines ~/Downloads/ISP_audits/lines_temp/A5_F1.json \\
        --out-csv ./A5_F1_lrai_fps5.csv \\
        --target-fps 5 \\
        --max-seconds 30 \\
        --weights /tmp/locany_local/weights

For the full ISP audit matrix, pass --max-seconds 0 (no cap) and a real GPU
(A100/A10G recommended; PT path is ~1 fps on Mac MPS).
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Counting line algorithm — direct port of tracker_bot's per-line CCW test.
# ---------------------------------------------------------------------------
def _parse_linestring(wkt: str) -> List[Tuple[float, float]]:
    """LINESTRING(x y, x y, ...) -> list of (x, y). Tolerates commas, spaces."""
    inner = wkt.strip().upper().removeprefix("LINESTRING").strip().lstrip("(").rstrip(")")
    pts = []
    for piece in inner.split(","):
        xy = re.findall(r"-?\d+(?:\.\d+)?", piece)
        if len(xy) >= 2:
            pts.append((float(xy[0]), float(xy[1])))
    return pts


def _parse_point(wkt: str) -> Tuple[float, float]:
    xy = re.findall(r"-?\d+(?:\.\d+)?", wkt)
    return (float(xy[0]), float(xy[1]))


def _ccw(ax, ay, bx, by, cx, cy) -> float:
    """Signed cross product of (B-A) x (C-A). Positive = CCW; negative = CW; 0 = collinear."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Standard segment-segment intersection (ignores collinear-touch as no-cross)."""
    d1 = _ccw(*p3, *p4, *p1)
    d2 = _ccw(*p3, *p4, *p2)
    d3 = _ccw(*p1, *p2, *p3)
    d4 = _ccw(*p1, *p2, *p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


@dataclass
class LineConfig:
    line_id: str
    name: str
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]  # consecutive pairs
    anchor: Tuple[float, float]
    coalesce_entries: bool = True
    in_count: int = 0
    out_count: int = 0
    # per-track most-recent direction credited (for coalesce dedup)
    last_dir: Dict[int, str] = field(default_factory=dict)


def _build_line_configs(lines_json_path: Path) -> List[LineConfig]:
    """Parse a tracker_bot lines.json file into LineConfig objects."""
    raw = json.loads(lines_json_path.read_text())
    out = []
    for item in raw["content"]:
        pts = _parse_linestring(item["counting_line"])
        anchor = _parse_point(item["anchor"])
        segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        out.append(LineConfig(
            line_id=item["id"], name=item["name"],
            segments=segments, anchor=anchor,
            coalesce_entries=bool(item.get("coalesce_entries", True)),
        ))
    return out


def _evaluate_line(line: LineConfig, track_id: int, prev_pt, curr_pt,
                   frame: int, ts: float) -> List[dict]:
    """Check every segment of `line` for a crossing on the `prev_pt → curr_pt`
    motion vector. Apply anchor-side polarity and coalesce dedup. Returns
    CSV rows (one per crossing event)."""
    rows = []
    for seg_a, seg_b in line.segments:
        if not _segments_intersect(prev_pt, curr_pt, seg_a, seg_b):
            continue
        # Polarity: sign(seg → curr_pt) vs sign(seg → anchor). Match = IN.
        sign_curr = _ccw(*seg_a, *seg_b, *curr_pt)
        sign_anc = _ccw(*seg_a, *seg_b, *line.anchor)
        direction = "IN" if (sign_curr * sign_anc) > 0 else "OUT"

        coalesced = 0
        if line.coalesce_entries:
            prev_dir = line.last_dir.get(track_id)
            if prev_dir == direction:
                # Same direction recrossing — suppressed entirely.
                continue
            if prev_dir is not None and prev_dir != direction:
                # Flip: subtract the earlier credited direction (coalesce).
                if prev_dir == "IN":
                    line.in_count = max(0, line.in_count - 1)
                else:
                    line.out_count = max(0, line.out_count - 1)
                coalesced = 1
            line.last_dir[track_id] = direction

        if direction == "IN":
            line.in_count += 1
        else:
            line.out_count += 1
        rows.append({
            "frame": frame, "timestamp_s": f"{ts:.3f}",
            "line_id": line.line_id, "line_name": line.name,
            "track_id": track_id, "direction": direction,
            "coalesced": coalesced,
            "in_count_after": line.in_count,
            "out_count_after": line.out_count,
            "x": f"{curr_pt[0]:.1f}", "y": f"{curr_pt[1]:.1f}",
        })
    return rows


# ---------------------------------------------------------------------------
# Minimal greedy-IoU tracker. NOT a substitute for ByteTracker in production
# audits — it cannot recover identity through occlusions. Adequate for smoke
# tests + AI-FPS measurement + CSV schema validation.
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    last_seen: int


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class GreedyIoUTracker:
    def __init__(self, iou_thresh: float = 0.25, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float]], frame_idx: int
               ) -> List[Tuple[int, Tuple[float, float], Tuple[float, float]]]:
        """Returns [(track_id, prev_center, curr_center)] for active tracks updated this frame."""
        # Build IoU matrix
        track_ids = list(self.tracks.keys())
        if not track_ids or not detections:
            assignments = {}
        else:
            iou_mat = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_mat[i, j] = _bbox_iou(self.tracks[tid].bbox, det)
            assignments = {}
            # Greedy: pick max-IoU pair iteratively
            while True:
                idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[idx] < self.iou_thresh:
                    break
                i, j = idx
                assignments[track_ids[i]] = j
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1

        # Update matched tracks
        out = []
        matched_dets = set(assignments.values())
        for tid, j in assignments.items():
            det = detections[j]
            prev_center = self.tracks[tid].center
            new_center = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)
            self.tracks[tid] = Track(track_id=tid, bbox=det, center=new_center, last_seen=frame_idx)
            out.append((tid, prev_center, new_center))

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self.next_id; self.next_id += 1
            cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
            self.tracks[tid] = Track(track_id=tid, bbox=det, center=(cx, cy), last_seen=frame_idx)
            # No prior position → emit at curr=prev so no crossing on creation
            out.append((tid, (cx, cy), (cx, cy)))

        # Age out stale tracks
        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid].last_seen > self.max_age:
                del self.tracks[tid]
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--lines", required=True, type=Path, help="tracker_bot lines.json")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-video", type=Path, default=None,
                    help="Optional annotated overlay mp4")
    ap.add_argument("--target-fps", type=int, default=5,
                    help="Sample stride from source (5 or 15 per canonical audit matrix)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="Cap video duration for smoke tests (0 = full clip)")
    ap.add_argument("--weights", type=Path, default=Path("/tmp/locany_local/weights"),
                    help="LocateAnything-3B local weights dir")
    ap.add_argument("--prompt", default="Locate all the instances that matches the following description: people.",
                    help="Detection prompt (canonicalize_prompt auto-pluralizes singulars)")
    ap.add_argument("--device", default=None,
                    help="Override device (cpu/mps/cuda); default = autodetect")
    args = ap.parse_args()

    # Lazy import so --help works without torch
    import torch
    from lrai_locate_anything.model_loader import load_locateanything_3b
    from lrai_locate_anything.orchestrator import LocateAnythingRunner, canonicalize_prompt
    from lrai_locate_anything.parse import parse_boxes

    print(f"[audit] device={args.device or 'auto'}  weights={args.weights}", file=sys.stderr)
    t_load_0 = time.time()
    model, tok, proc, cfg, local, snap = load_locateanything_3b(
        local_dir=args.weights, verbose=True,
    )
    if args.device:
        model.to(args.device)
    runner = LocateAnythingRunner(model, tok, proc, cfg, local, patches_snapshot=snap)
    print(f"[audit] load_time={time.time() - t_load_0:.1f}s", file=sys.stderr)

    # Canonicalize the prompt once (auto-pluralizes per the new helper)
    canonical_prompt, was_rewritten = canonicalize_prompt(args.prompt)
    if was_rewritten:
        print(f"[audit] prompt rewritten: {args.prompt!r} -> {canonical_prompt!r}", file=sys.stderr)

    lines_cfg = _build_line_configs(args.lines)
    tracker = GreedyIoUTracker()
    rows: List[dict] = []
    inference_time = 0.0
    n_inferences = 0
    n_detections_total = 0

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[audit] ERROR: could not open {args.video}", file=sys.stderr); return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(src_fps / args.target_fps)))
    print(f"[audit] video {W}x{H} src_fps={src_fps:.1f} total_frames={src_total}  "
          f"target_fps={args.target_fps} stride={stride}", file=sys.stderr)

    writer = None
    if args.out_video is not None:
        writer = cv2.VideoWriter(str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                  src_fps / stride, (W, H))

    t_start = time.time()
    frame_idx = 0
    last_print = t_start
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1; continue
        ts = frame_idx / src_fps
        if args.max_seconds and ts > args.max_seconds:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        t0 = time.time()
        try:
            boxes, _text = runner.detect(pil, canonical_prompt, diagnostic=False)
        except RuntimeError as e:
            # detect() raises if 0 boxes; treat as no detections this frame
            if "returned 0 boxes" in str(e):
                boxes = []
            else:
                raise
        inference_time += time.time() - t0
        n_inferences += 1
        n_detections_total += len(boxes)

        # Tracker + crossing eval
        tracks = tracker.update(list(boxes), frame_idx)
        for tid, prev_c, curr_c in tracks:
            for ln in lines_cfg:
                rows.extend(_evaluate_line(ln, tid, prev_c, curr_c, frame_idx, ts))

        # Overlay
        if writer is not None:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(bgr, f"t={ts:.1f}s  dets={len(boxes)}  ai={n_inferences/inference_time:.2f} fps"
                        if inference_time > 0 else f"t={ts:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for ln in lines_cfg:
                for (a, b) in ln.segments:
                    cv2.line(bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 0, 0), 2)
            writer.write(bgr)

        # Progress
        if time.time() - last_print > 15.0:
            elapsed = time.time() - t_start
            print(f"[audit] frame={frame_idx}  ts={ts:.1f}s  "
                  f"inf={n_inferences}  ai_fps={n_inferences/inference_time:.2f}  "
                  f"proc_fps={n_inferences/elapsed:.2f}  "
                  f"dets_total={n_detections_total}  rows={len(rows)}", file=sys.stderr)
            last_print = time.time()
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    ai_fps = n_inferences / inference_time if inference_time > 0 else 0.0
    proc_fps = n_inferences / elapsed if elapsed > 0 else 0.0
    total_in = sum(ln.in_count for ln in lines_cfg)
    total_out = sum(ln.out_count for ln in lines_cfg)

    # CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "timestamp_s", "line_id", "line_name", "track_id",
            "direction", "coalesced", "in_count_after", "out_count_after", "x", "y",
        ])
        w.writeheader()
        w.writerows(rows)

    print("", file=sys.stderr)
    print(f"[audit] DONE", file=sys.stderr)
    print(f"  total_wall_clock_s   = {elapsed:.1f}", file=sys.stderr)
    print(f"  total_inference_s    = {inference_time:.1f}", file=sys.stderr)
    print(f"  frames_processed     = {n_inferences}", file=sys.stderr)
    print(f"  detections_total     = {n_detections_total}", file=sys.stderr)
    print(f"  ai_fps               = {ai_fps:.2f}   (model throughput; detect() only)", file=sys.stderr)
    print(f"  proc_fps             = {proc_fps:.2f}   (end-to-end incl decode/track/count)", file=sys.stderr)
    print(f"  in_total             = {total_in}", file=sys.stderr)
    print(f"  out_total            = {total_out}", file=sys.stderr)
    print(f"  csv_out              = {args.out_csv}", file=sys.stderr)
    if args.out_video is not None:
        print(f"  video_out            = {args.out_video}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
