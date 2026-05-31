"""TRT engine build: generic build_engine + per-graph profile builders.

Two architectural invariants this module enforces:

1. **Use parse_from_file, not parse(bytes)**. The parser needs the .onnx file path
   to locate side-car external-data .bin files (LLM exports use them).

2. **Profile keys MUST match ONNX input_names exactly**. TRT's
   build_serialized_network returns None silently on mismatch with no log entry at
   any verbosity level. A profile key typo is the most expensive bug class here.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import tensorrt as trt
from .engine import _DEFAULT_LOGGER


def build_engine(
    onnx_path: Path | str,
    profile_spec: Dict[str, Tuple[tuple, tuple, tuple]],
    out_engine: Path,
    fp16: bool = True,
    workspace_gb: int = 4,
    name: str = "engine",
    logger: trt.Logger | None = None,
) -> Path:
    """Build a single TRT engine from ONNX with one optimisation profile.

    profile_spec maps input_name -> (min_shape, opt_shape, max_shape).
    Idempotent: returns the cached engine file if it exists.
    """
    out_engine = Path(out_engine)
    if out_engine.exists():
        print(f"[trt] cached: {out_engine}")
        return out_engine

    logger = logger or _DEFAULT_LOGGER
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    # parse_from_file (not parse(bytes)) so the parser can locate side-car external-data
    # files (.bin) that sit next to the .onnx (FP16 LLM + INT4 exports use them).
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)

    prof = builder.create_optimization_profile()
    for n, (lo, opt, hi) in profile_spec.items():
        prof.set_shape(n, lo, opt, hi)
    cfg.add_optimization_profile(prof)

    print(f"[trt] building {name} (fp16={fp16}, workspace={workspace_gb} GB) ...")
    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError(
            f"TRT build_serialized_network returned None for {name}. "
            f"Most common cause: profile_spec keys don't match the ONNX input_names. "
            f"Verify with: `import onnx; print([i.name for i in onnx.load('{onnx_path}', load_external_data=False).graph.input])`"
        )
    out_engine.write_bytes(plan)
    print(f"[trt]   built in {time.time()-t0:.1f}s -> {out_engine}  ({out_engine.stat().st_size/1e9:.2f} GB)")
    return out_engine


# ---------------------------------------------------------------------------
# Per-graph profile builders
# ---------------------------------------------------------------------------
def build_vision(onnx_path: Path, out_engine: Path, l_pre_fixed: int,
                 pixel_shape_tail: tuple = (3, 14, 14), **kw) -> Path:
    """Fixed-resolution vision engine. L_pre is fixed because pos_emb is baked."""
    prof = {
        "pixel_values": (
            (l_pre_fixed,) + pixel_shape_tail,
            (l_pre_fixed,) + pixel_shape_tail,
            (l_pre_fixed,) + pixel_shape_tail,
        ),
    }
    return build_engine(onnx_path, prof, out_engine, name="vision", **kw)


def build_projector(onnx_path: Path, out_engine: Path, l_post_fixed: int,
                    proj_dim_in: int = 4608, **kw) -> Path:
    """Projector engine. L_post == L_pre_fixed / 4 (after the 2x2 merger)."""
    prof = {
        "vit_feats_4x": (
            (l_post_fixed, proj_dim_in),
            (l_post_fixed, proj_dim_in),
            (l_post_fixed, proj_dim_in),
        ),
    }
    return build_engine(onnx_path, prof, out_engine, name="projector", **kw)


def build_llm(prefill_onnx: Path, decode_onnx: Path,
              prefill_engine: Path, decode_engine: Path,
              hidden_size: int, n_layers: int, n_kv_heads: int, head_dim: int,
              s_min: int = 16, s_opt: int = 1024, s_max: int = 4096,
              p_min: int = 0, p_opt: int = 1024, p_max: int = 4096,
              workspace_gb: int = 8, **kw) -> Tuple[Path, Path]:
    """Build prefill + decode LLM engines. Profile keys here MUST match LLMPrefill /
    LLMDecode's forward parameter names exported via input_names."""
    H = hidden_size
    pref_prof = {
        "input_ids":       ((1, s_min), (1, s_opt), (1, s_max)),
        # visual_features is dynamic on dim 0; the engine accepts any image-token count
        # up to the typical max L_post for a frame. We expose it as L_post in [1, s_max].
        "visual_features": ((1, H),     (1024, H),  (s_max, H)),
        "position_ids":    ((1, s_min), (1, s_opt), (1, s_max)),
        "attention_mask":  ((1, s_min), (1, s_opt), (1, s_max)),
    }
    pre = build_engine(prefill_onnx, pref_prof, prefill_engine,
                       workspace_gb=workspace_gb, name="llm_prefill", **kw)

    dec_prof = {
        "input_ids":      ((1, 1), (1, 6),    (1, 6)),
        "position_ids":   ((1, 1), (1, 6),    (1, 6)),
        "attention_mask": ((1, 1), (1, p_opt + 6), (1, p_max + 6)),
    }
    for i in range(n_layers):
        dec_prof[f"past_k_{i}"] = ((1, n_kv_heads, p_min, head_dim),
                                    (1, n_kv_heads, p_opt, head_dim),
                                    (1, n_kv_heads, p_max, head_dim))
        dec_prof[f"past_v_{i}"] = ((1, n_kv_heads, p_min, head_dim),
                                    (1, n_kv_heads, p_opt, head_dim),
                                    (1, n_kv_heads, p_max, head_dim))
    dec = build_engine(decode_onnx, dec_prof, decode_engine,
                       workspace_gb=workspace_gb, name="llm_decode", **kw)
    return pre, dec
