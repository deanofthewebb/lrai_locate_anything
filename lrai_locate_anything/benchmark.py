"""Side-by-side runtime benchmark with detailed metrics.

Three benchmarks are exposed:

  - `bench_text_runtimes(...)`: text-only AR latency comparison across PyTorch,
    plain-TRT (your installed engines), and TRT-LLM. Measures prefill + decode/tok
    separately. Useful for isolating runtime gains from algorithmic ones.

  - `bench_image(runner, image, prompt)`: timed inference on a single image
    through the orchestrator's TRT engines. Returns the boxes plus latency breakdown.

  - `bench_video_compare(runner, video, output)`: side-by-side multi-runtime video
    panels (TRT+MTP / TRT-AR / PyTorch) with per-frame metrics + cross-runtime IoU
    agreement. The replacement for the original notebook's §16/§17 cells.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .orchestrator import LocateAnythingRunner
from .pipelines import run_compare
from .parse import parse_boxes


# ---------------------------------------------------------------------------
# Text-only runtime comparison
# ---------------------------------------------------------------------------
def _ar_pt(model, ids: torch.Tensor, max_new: int) -> torch.Tensor:
    """Manual AR loop using Qwen2Model + lm_head. Qwen2ForCausalLM in the vendored
    code doesn't inherit GenerationMixin, so .generate() doesn't exist on it."""
    lm = model.language_model
    out_ids = ids.clone()
    with torch.inference_mode():
        past = None
        for _ in range(max_new):
            if past is None:
                o = lm.model(input_ids=out_ids, use_cache=True, return_dict=True)
            else:
                o = lm.model(input_ids=out_ids[:, -1:], past_key_values=past, use_cache=True, return_dict=True)
            past = o.past_key_values
            logits = lm.lm_head(o.last_hidden_state[:, -1, :])
            next_id = logits.argmax(-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_id], dim=1)
    return out_ids


def _ar_plain_trt(runner: LocateAnythingRunner, ids_np: np.ndarray, max_new: int):
    """Run plain-TRT generate in AR mode (generation_mode='slow' disables MTP).
    Vision input is a dummy zero tensor at the engine-baked size — only the LM is timed."""
    L = runner.grid_h * runner.grid_w
    px_dummy = np.zeros((L, 3, 14, 14), dtype=np.float16)
    _, toks = runner._gen.generate(
        px_dummy, ids_np, max_new_tokens=max_new,
        generation_mode="slow", temperature=0.0,
    )
    return toks


def _ar_trtllm(trtllm_runner, ids: torch.Tensor, max_new: int):
    return trtllm_runner.generate(ids, max_new_tokens=max_new)


def bench_text_runtimes(
    runner: LocateAnythingRunner,
    trtllm_runner=None,
    prompt: Optional[str] = None,
    iters: int = 5,
    warmup: int = 2,
    decode_probe_tokens: int = 32,
) -> Dict[str, dict]:
    """Apples-to-apples text-only AR benchmark across runtimes.

    Method: run each runtime at max_new=1 and max_new=K, subtract to get decode/tok.
        prefill_ms ≈ t_1 - decode_ms
        decode_ms  = (t_K - t_1) / (K - 1)

    Skips the PyTorch row if `model` was freed (e.g. by an OOM-retry path).
    Skips TRT-LLM if `trtllm_runner` is None.
    """
    if prompt is None:
        prompt = "Describe the scene in detail. " * 30
    ids = runner.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    ids_np = ids.cpu().numpy().astype(np.int64)
    S = ids.shape[1]

    def derive(fn: Callable, max_new: int) -> Optional[float]:
        try:
            for _ in range(warmup):
                fn(max_new)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = []
            for _ in range(iters):
                t0 = time.time()
                fn(max_new)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t.append(time.time() - t0)
            return min(t)
        except Exception as e:
            print(f"  bench step failed: {e}")
            return None

    results: Dict[str, dict] = {}

    # PyTorch
    if runner.model is not None:
        def _pt(n): return _ar_pt(runner.model, ids, n)
        t1 = derive(_pt, 1)
        tK = derive(_pt, decode_probe_tokens) if t1 is not None else None
        if t1 and tK:
            dec = (tK - t1) / max(1, decode_probe_tokens - 1) * 1000
            pre = t1 * 1000 - dec
            results["PyTorch"] = {"prefill_ms": pre, "decode_ms_per_tok": dec,
                                   "fps_ar": 1000/dec if dec > 0 else 0,
                                   "bps_ar_6tok_per_box": (1000/dec/6) if dec > 0 else 0}
    else:
        results["PyTorch"] = {"status": "skipped (model freed)"}

    # Plain TRT (AR)
    def _trt(n): return _ar_plain_trt(runner, ids_np, n)
    t1 = derive(_trt, 1)
    tK = derive(_trt, decode_probe_tokens) if t1 is not None else None
    if t1 and tK:
        dec = (tK - t1) / max(1, decode_probe_tokens - 1) * 1000
        pre = t1 * 1000 - dec
        results["Plain TRT (AR)"] = {"prefill_ms": pre, "decode_ms_per_tok": dec,
                                       "fps_ar": 1000/dec if dec > 0 else 0,
                                       "bps_ar_6tok_per_box": (1000/dec/6) if dec > 0 else 0}

    # TRT-LLM
    if trtllm_runner is not None:
        def _tll(n): return _ar_trtllm(trtllm_runner, ids, n)
        t1 = derive(_tll, 1)
        tK = derive(_tll, decode_probe_tokens) if t1 is not None else None
        if t1 and tK:
            dec = (tK - t1) / max(1, decode_probe_tokens - 1) * 1000
            pre = t1 * 1000 - dec
            results["TRT-LLM (AR)"] = {"prefill_ms": pre, "decode_ms_per_tok": dec,
                                        "fps_ar": 1000/dec if dec > 0 else 0,
                                        "bps_ar_6tok_per_box": (1000/dec/6) if dec > 0 else 0}
    else:
        results["TRT-LLM (AR)"] = {"status": "skipped (no trtllm_runner provided)"}

    # Plain TRT + MTP — context row (the actual fastest configuration in the package)
    def _trt_mtp(n):
        L = runner.grid_h * runner.grid_w
        px_dummy = np.zeros((L, 3, 14, 14), dtype=np.float16)
        return runner._gen.generate(px_dummy, ids_np, max_new_tokens=n,
                                     generation_mode="hybrid", temperature=0.0)
    try:
        for _ in range(warmup): _trt_mtp(decode_probe_tokens)
        t = []
        n_box = 0
        for _ in range(iters):
            t0 = time.time()
            _, toks = _trt_mtp(decode_probe_tokens * 4)
            t.append(time.time() - t0)
            n_box += len(parse_boxes(runner.tokenizer.decode(toks, skip_special_tokens=False)))
        dt = float(np.mean(t))
        results["Plain TRT + MTP (context)"] = {
            "wall_s_per_request": dt,
            "boxes_per_request_avg": n_box / iters,
            "bps_end_to_end": n_box / iters / dt,
        }
    except Exception as e:
        results["Plain TRT + MTP (context)"] = {"status": f"failed: {e}"}

    return results


def print_text_table(results: Dict[str, dict]) -> None:
    """Pretty-print the bench_text_runtimes output as a single table."""
    print()
    print(f'{"Runtime":<28}  {"Prefill":>10}  {"Decode/tok":>12}  {"AR fps":>8}  {"AR BPS":>9}')
    print("-" * 78)
    for name, r in results.items():
        if "status" in r:
            print(f"{name:<28}  {r['status']:>10}")
            continue
        if "prefill_ms" in r:
            print(f"{name:<28}  {r['prefill_ms']:7.1f} ms  {r['decode_ms_per_tok']:9.2f} ms  "
                  f"{r['fps_ar']:6.1f}  {r['bps_ar_6tok_per_box']:6.2f}")
        elif "bps_end_to_end" in r:
            print(f"{name:<28}  end-to-end BPS = {r['bps_end_to_end']:.2f} "
                  f"({r['boxes_per_request_avg']:.1f} boxes/request, {r['wall_s_per_request']:.2f}s)")


# ---------------------------------------------------------------------------
# Image timing
# ---------------------------------------------------------------------------
def bench_image(runner: LocateAnythingRunner, image, prompt: str, iters: int = 3) -> dict:
    """Time a single image through .detect(). Returns per-iter ms + box stats."""
    times = []
    n_boxes = []
    for _ in range(iters):
        t0 = time.time()
        boxes, _ = runner.detect(image, prompt)
        times.append((time.time() - t0) * 1000)
        n_boxes.append(len(boxes))
    return {
        "iters": iters,
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "avg_boxes": float(np.mean(n_boxes)),
    }


# ---------------------------------------------------------------------------
# Video side-by-side (alias for pipelines.run_compare with richer metrics)
# ---------------------------------------------------------------------------
def bench_video_compare(*args, **kwargs) -> dict:
    """Alias for pipelines.run_compare — kept here so the benchmark namespace is
    a single import."""
    return run_compare(*args, **kwargs)
