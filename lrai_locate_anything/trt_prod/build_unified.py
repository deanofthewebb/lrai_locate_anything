"""TRT builders for the fused production engines.

Phase 1: build_vision_proj_engine — one engine for vision + merger + mlp1.
Phase 3: build_llm_unified_engine — one engine for prefill + decode via
         TRT 10's IIfConditional (ONNX If node + use_cache_branch BOOL).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import tensorrt as trt

from lrai_locate_anything.trt.build import build_engine
from lrai_locate_anything.trt.engine import get_trt_logger


def build_vision_proj_engine(
    onnx_path: Path,
    engine_path: Path,
    L_pre: int,
    pixel_shape_tail: tuple = (3, 14, 14),
    workspace_gb: int = 4,
    logger: Optional[trt.Logger] = None,
) -> Path:
    """Build the fused vision_proj engine.

    Fixed shape (min=opt=max) on pixel_values because pos_emb_baked
    inside VisionForExport is resolution-locked. STRONGLY_TYPED + FP16
    so the ONNX fp16 dtype is honored exactly — no silent tactic-driven
    precision drift on LayerNorm / GELU inside mlp1.
    """
    shape = (L_pre,) + pixel_shape_tail
    profile_spec = {"pixel_values": (shape, shape, shape)}
    return build_engine(
        onnx_path,
        profile_spec,
        engine_path,
        fp16=True,
        bf16=False,
        strongly_typed=True,
        workspace_gb=workspace_gb,
        name="vision_proj",
        logger=logger,
    )


def build_llm_unified_engine(
    onnx_path: Path,
    engine_path: Path,
    hidden_size: int = 2048,
    n_layers: int = 36,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    logger: trt.Logger | None = None,
) -> Path:
    """Build the unified prefill+decode LLM engine from llm_unified.onnx.

    STRONGLY_TYPED bf16 (matches the existing decode engines). Single
    optimization profile spanning prefill (S in [16, 1024, 4096], P=0,
    use_cache_branch=False) and decode (S=1, P in [0, 1024, 4096],
    use_cache_branch=True). TRT 10's IIfConditional importer maps the
    ONNX If node + use_cache_branch BOOL input directly to its native
    conditional layer.
    """
    builder = trt.Builder(logger or get_trt_logger())
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.BF16)
    # CRITICAL: do NOT also set FP16 — STRONGLY_TYPED makes that an error.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    parser = trt.OnnxParser(network, logger or get_trt_logger())
    success = parser.parse_from_file(str(onnx_path))
    if not success:
        for e in range(parser.num_errors):
            print(parser.get_error(e))
        raise RuntimeError("OnnxParser failed")

    # Single optimization profile covering BOTH prefill and decode regimes.
    # input_ids: (1, S) — S min=1 opt=1024 max=4096
    # position_ids: (1, S) — same as input_ids
    # attention_mask: (1, S) — same (prefill uses S, decode uses 1; both fit in [1, 4096])
    # visual_features: (Lpost, hidden_size) — min=(0,2048) opt=(414,2048) max=(2048,2048)
    # past_k_i / past_v_i: (1, n_kv_heads, P, head_dim) — P min=0 opt=1024 max=4096
    # use_cache_branch: () BOOL scalar — no profile needed (no dynamic dims)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",      (1, 1), (1, 1024), (1, 4096))
    profile.set_shape("position_ids",   (1, 1), (1, 1024), (1, 4096))
    profile.set_shape("attention_mask", (1, 1), (1, 1024), (1, 4096))
    profile.set_shape(
        "visual_features",
        (0, hidden_size),
        (414, hidden_size),
        (2048, hidden_size),
    )
    for i in range(n_layers):
        profile.set_shape(
            f"past_k_{i}",
            (1, n_kv_heads, 0, head_dim),
            (1, n_kv_heads, 1024, head_dim),
            (1, n_kv_heads, 4096, head_dim),
        )
        profile.set_shape(
            f"past_v_{i}",
            (1, n_kv_heads, 0, head_dim),
            (1, n_kv_heads, 1024, head_dim),
            (1, n_kv_heads, 4096, head_dim),
        )
    config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return engine_path


def _cli() -> None:
    p = argparse.ArgumentParser(description=build_llm_unified_engine.__doc__)
    p.add_argument("--onnx", type=Path, required=True, help="Path to llm_unified.onnx")
    p.add_argument("--engine", type=Path, required=True, help="Output .engine path")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=36)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    args = p.parse_args()
    out = build_llm_unified_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )
    print(f"[trt_prod] built unified LLM engine -> {out}")


if __name__ == "__main__":
    _cli()
