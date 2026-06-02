"""ONNXRuntime CPU parity test: fused vision_proj.onnx vs 3-stage chain.

Runs the same synthetic input through both:
  (a) vision.onnx -> python_patch_merger (numpy) -> projector.onnx
  (b) vision_proj.onnx (fused)
and asserts max_abs_diff < 1e-2 on the fp16 output. Loose tolerance
because LayerNorm + GELU rounding differs once ops are fused into a
single graph for tactic selection.

Not invoked by the workflow — meant for the operator to run on learn02
once the Phase 1 export completes. CPU-only; no GPU required.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

from lrai_locate_anything.parse import python_patch_merger


def _run_chained(vision_onnx: Path, projector_onnx: Path,
                 pixel_values: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    vit = ort.InferenceSession(str(vision_onnx), providers=providers)
    prj = ort.InferenceSession(str(projector_onnx), providers=providers)

    vit_feats = vit.run(["vit_feats"], {"pixel_values": pixel_values})[0]
    gh = np.array([[grid_h, grid_w]], dtype=np.int32)
    merged = python_patch_merger(vit_feats, gh, kh=2, kw=2).astype(np.float16)
    return prj.run(["proj_feats"], {"vit_feats_4x": merged})[0]


def _run_fused(vision_proj_onnx: Path, pixel_values: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(vision_proj_onnx), providers=["CPUExecutionProvider"])
    return sess.run(["visual_features"], {"pixel_values": pixel_values})[0]


def parity(vision_onnx: Path, projector_onnx: Path, vision_proj_onnx: Path,
           grid_h: int = 36, grid_w: int = 46, tol: float = 1e-2,
           seed: int = 0) -> bool:
    L_pre = grid_h * grid_w
    rng = np.random.default_rng(seed)
    px = rng.standard_normal((L_pre, 3, 14, 14), dtype=np.float32).astype(np.float16)

    out_chained = _run_chained(vision_onnx, projector_onnx, px, grid_h, grid_w)
    out_fused = _run_fused(vision_proj_onnx, px)

    if out_chained.shape != out_fused.shape:
        print(f"FAIL: shape mismatch  chained={out_chained.shape}  fused={out_fused.shape}")
        return False

    diff = np.abs(out_fused.astype(np.float32) - out_chained.astype(np.float32))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ok = max_abs < tol
    tag = "PASS" if ok else "FAIL"
    print(f"{tag}: max_abs_diff={max_abs:.4e}  mean_abs_diff={mean_abs:.4e}  tol={tol:.0e}  shape={out_fused.shape}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-onnx", type=Path, required=True)
    ap.add_argument("--projector-onnx", type=Path, required=True)
    ap.add_argument("--vision-proj-onnx", type=Path, required=True)
    ap.add_argument("--grid-h", type=int, default=36)
    ap.add_argument("--grid-w", type=int, default=46)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok = parity(
        args.vision_onnx, args.projector_onnx, args.vision_proj_onnx,
        grid_h=args.grid_h, grid_w=args.grid_w, tol=args.tol, seed=args.seed,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
