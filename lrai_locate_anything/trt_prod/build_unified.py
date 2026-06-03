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
    # STRONGLY_TYPED networks derive precision from ONNX dtypes; explicit BF16/FP16
    # builder flags are forbidden here (TRT 10 Error Code 3).
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    parser = trt.OnnxParser(network, logger or get_trt_logger())
    success = parser.parse_from_file(str(onnx_path))
    if not success:
        for e in range(parser.num_errors):
            print(parser.get_error(e))
        raise RuntimeError("OnnxParser failed")

    # Profile 0: PREFILL regime (use_cache_branch=False)
    # S = prompt length (variable); P = 0 (no past KV); visual_features active.
    profile_prefill = builder.create_optimization_profile()
    profile_prefill.set_shape("input_ids",      (1, 1), (1, 1024), (1, 4096))
    profile_prefill.set_shape("position_ids",   (1, 1), (1, 1024), (1, 4096))
    profile_prefill.set_shape("attention_mask", (1, 1), (1, 1024), (1, 4096))
    profile_prefill.set_shape("visual_features", (0, hidden_size), (414, hidden_size), (2048, hidden_size))
    for i in range(n_layers):
        profile_prefill.set_shape(f"past_k_{i}", (1, n_kv_heads, 0, head_dim),
                                                  (1, n_kv_heads, 0, head_dim),
                                                  (1, n_kv_heads, 0, head_dim))
        profile_prefill.set_shape(f"past_v_{i}", (1, n_kv_heads, 0, head_dim),
                                                  (1, n_kv_heads, 0, head_dim),
                                                  (1, n_kv_heads, 0, head_dim))
    config.add_optimization_profile(profile_prefill)

    # Profile 1: DECODE regime (use_cache_branch=True)
    # S = 1 (single token per step); P = past length (grows during generation); visual_features unused (zero-length).
    profile_decode = builder.create_optimization_profile()
    profile_decode.set_shape("input_ids",      (1, 1), (1, 1), (1, 1))
    profile_decode.set_shape("position_ids",   (1, 1), (1, 1), (1, 1))
    profile_decode.set_shape("attention_mask", (1, 2), (1, 1025), (1, 4097))  # P + S; min=2 ensures P>=1 since S>=1
    profile_decode.set_shape("visual_features", (0, hidden_size), (0, hidden_size), (0, hidden_size))
    for i in range(n_layers):
        profile_decode.set_shape(f"past_k_{i}", (1, n_kv_heads, 1, head_dim),
                                                 (1, n_kv_heads, 1024, head_dim),
                                                 (1, n_kv_heads, 4096, head_dim))
        profile_decode.set_shape(f"past_v_{i}", (1, n_kv_heads, 1, head_dim),
                                                 (1, n_kv_heads, 1024, head_dim),
                                                 (1, n_kv_heads, 4096, head_dim))
    config.add_optimization_profile(profile_decode)

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
