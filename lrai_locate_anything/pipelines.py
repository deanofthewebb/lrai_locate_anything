"""Image + video inference pipelines.

`run_image` — single image -> (boxes, annotated PIL).
`run_video`  — per-frame inference over an mp4, writes an annotated mp4.
`run_compare` — side-by-side multi-runtime comparison on a custom clip.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple, Callable, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None  # video pipelines require opencv-python-headless

from .orchestrator import LocateAnythingRunner
from .parse import iou


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------
def run_image(
    runner: LocateAnythingRunner,
    image,
    prompt: str = "Locate all the instances that matches the following description: objects.",
    max_new_tokens: int = 128,
    draw: bool = True,
    path: str = "pt",
) -> Tuple[List[Tuple[float, float, float, float]], Image.Image, str]:
    """Run single-image detection. Returns (boxes, annotated_image, raw_text).

    path: 'auto' picks TRT engines if loaded else PT; 'pt' forces PyTorch;
          'trt' forces TensorRT. Default 'pt' because the TRT decode engines
          currently return all-zero logits (the bf16-export regression
          captured by the MTP-decode probe). Override to 'trt' or 'auto'
          once the engines are fixed."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    boxes, text = runner.detect(image, prompt, max_new_tokens=max_new_tokens, path=path)
    if not draw:
        return boxes, image, text
    canvas = image.copy()
    drw = ImageDraw.Draw(canvas)
    for (x1, y1, x2, y2) in boxes:
        drw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    return boxes, canvas, text


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def run_video(
    runner: LocateAnythingRunner,
    input_path: Path | str,
    output_path: Path | str,
    prompt: str = "Locate all the instances that matches the following description: people</c>luggage.",
    max_frames: int = 40,
    max_new_tokens: int = 120,
    generation_mode: str = "hybrid",
    path: str = "pt",
) -> dict:
    """Run per-frame detection on a video. Writes annotated mp4 to output_path.
    Returns a dict of metrics: total frames, total time, avg latency ms.

    path: 'auto' / 'pt' / 'trt'. Default 'pt' until the TRT decode engines
          are fixed (see run_image docstring).
    """
    if cv2 is None:
        raise ImportError("run_video requires opencv-python-headless")
    input_path, output_path = Path(input_path), Path(output_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    i, t_start = 0, time.time()
    latencies = []
    while i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t0 = time.time()
        boxes, _ = runner.detect(rgb, prompt, max_new_tokens=max_new_tokens,
                                  generation_mode=generation_mode, path=path)
        latencies.append((time.time() - t0) * 1000)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(bgr, f"LocateAnything-3B (TRT)  frame {i}  {latencies[-1]:.0f}ms  ({len(boxes)} dets)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)
        i += 1
        if i % 5 == 0:
            print(f"  processed {i} frames")

    writer.release()
    cap.release()
    dt = time.time() - t_start
    return {
        "frames": i,
        "total_s": dt,
        "fps": i / max(dt, 1e-3),
        "avg_ms_per_frame": float(np.mean(latencies)) if latencies else 0.0,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Side-by-side multi-runtime comparison
# ---------------------------------------------------------------------------
def run_compare(
    runner: LocateAnythingRunner,
    input_path: Path | str,
    output_path: Path | str,
    prompt: str = "Locate all the instances that matches the following description: luggage</c>person.",
    max_frames: int = 30,
    panel_w: int = 640,
    panel_h: int = 360,
    include_pytorch: bool = True,
) -> dict:
    """Side-by-side panel video comparing TRT+MTP / TRT-AR / PyTorch (if available).

    Each frame is rendered through each runtime; panels are concatenated horizontally;
    per-runtime metrics returned.
    """
    if cv2 is None:
        raise ImportError("run_compare requires opencv-python-headless")
    input_path, output_path = Path(input_path), Path(output_path)

    paths: List[Tuple[str, Callable]] = []
    paths.append(("TRT + MTP", lambda pil: runner.detect(pil, prompt, generation_mode="hybrid")))
    paths.append(("TRT (AR)",  lambda pil: runner.detect(pil, prompt, generation_mode="slow")))
    if include_pytorch and runner.model is not None:
        # PyTorch path via canonical generate()
        def _pt(pil):
            import torch
            from .config import REF_DTYPE
            enc = runner._processor_call(pil, prompt)
            with torch.inference_mode():
                out = runner.model.generate(
                    pixel_values=enc["pixel_values"], input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], image_grid_hws=enc["image_grid_hws"],
                    tokenizer=runner.tokenizer, max_new_tokens=128, use_cache=True,
                    generation_mode="hybrid",
                    do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                    verbose=False,
                )
            ot = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(ot):
                txt = runner.tokenizer.decode(ot[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            else:
                txt = str(ot)
            from .parse import parse_boxes as _pb
            # parse_boxes returns [(bbox, label)]; strip labels for the
            # downstream rendering loop which only consumes 4-tuples.
            boxes = [bbox for (bbox, _lbl) in _pb(txt, *pil.size)]
            return boxes, txt
        paths.append(("PyTorch", _pt))

    print(f"Comparing {len(paths)} runtimes: {', '.join(n for n,_ in paths)}")

    def _panel(bgr_frame, boxes, label, ms):
        p = cv2.resize(bgr_frame, (panel_w, panel_h))
        Hf, Wf = bgr_frame.shape[:2]
        sx, sy = panel_w / Wf, panel_h / Hf
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(p, (int(x1 * sx), int(y1 * sy)),
                          (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 2)
        cv2.rectangle(p, (0, 0), (panel_w, 36), (0, 0, 0), -1)
        cv2.putText(p, f"{label}  {ms:6.0f}ms  ({len(boxes)} dets)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return p

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    out_w, out_h = panel_w * len(paths), panel_h
    vw = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    metrics: Dict[str, dict] = {name: {"lat_ms": [], "n_dets": []} for name, _ in paths}
    agree_ious: List[float] = []

    i, t0 = 0, time.time()
    while i < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        panels, boxes_per = [], []
        for name, fn in paths:
            ts = time.time()
            boxes, _ = fn(rgb)
            ms = (time.time() - ts) * 1000
            metrics[name]["lat_ms"].append(ms)
            metrics[name]["n_dets"].append(len(boxes))
            boxes_per.append(boxes)
            panels.append(_panel(bgr, boxes, name, ms))
        if len(boxes_per) >= 2 and boxes_per[0] and boxes_per[1]:
            ious_f = [max((iou(b0, b1) for b1 in boxes_per[1]), default=0) for b0 in boxes_per[0]]
            if ious_f:
                agree_ious.append(float(np.mean(ious_f)))
        vw.write(np.concatenate(panels, axis=1))
        i += 1
        if i % 5 == 0:
            print(f"  processed {i} frames")

    vw.release()
    cap.release()
    return {
        "frames": i,
        "total_s": time.time() - t0,
        "output_path": str(output_path),
        "per_runtime": {
            name: {
                "avg_ms": float(np.mean(m["lat_ms"])) if m["lat_ms"] else 0.0,
                "p50_ms": float(np.percentile(m["lat_ms"], 50)) if m["lat_ms"] else 0.0,
                "p99_ms": float(np.percentile(m["lat_ms"], 99)) if m["lat_ms"] else 0.0,
                "avg_dets": float(np.mean(m["n_dets"])) if m["n_dets"] else 0.0,
                "total_dets": int(np.sum(m["n_dets"])) if m["n_dets"] else 0,
            }
            for name, m in metrics.items()
        },
        "agreement_iou_first_two": float(np.mean(agree_ious)) if agree_ious else None,
    }
