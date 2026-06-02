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
from typing import Dict, Tuple, Any, Optional

import tensorrt as trt
from .engine import _DEFAULT_LOGGER


class _CapturingLogger(trt.ILogger):
    """TRT logger that forwards to the default logger AND captures ERROR-severity
    messages so we can surface them in Python exceptions.

    build_serialized_network returns None on failure with NO Python exception —
    the actual error (hw-precision rejection, ONNX op unsupported, etc.) is
    only emitted to the C++ logger. Without capturing, the only signal is the
    None return, and our caller would raise a misleading "profile_spec keys
    don't match" message that hides the real cause.
    """
    def __init__(self, inner: trt.ILogger):
        trt.ILogger.__init__(self)
        self.inner = inner
        self.errors: list[str] = []

    def log(self, severity, msg):
        try:
            self.inner.log(severity, msg)
        except Exception:
            pass
        if severity in (trt.ILogger.ERROR, trt.ILogger.INTERNAL_ERROR):
            self.errors.append(str(msg))


def build_engine(
    onnx_path: Path | str,
    profile_spec: Dict[str, Tuple[tuple, tuple, tuple]],
    out_engine: Path,
    fp16: bool = True,
    bf16: bool = False,
    workspace_gb: int = 4,
    name: str = "engine",
    logger: trt.Logger | None = None,
    strongly_typed: bool = False,
) -> Path:
    """Build a single TRT engine from ONNX with one optimisation profile.

    profile_spec maps input_name -> (min_shape, opt_shape, max_shape).
    Idempotent: returns the cached engine file if it exists.

    Precision flags:
      - fp16=True: allow FP16 throughout. Required for the numpy I/O boundary
        (numpy lacks a native bf16 dtype, so the engine bindings stay FP16).
      - bf16=True: ADDITIONALLY allow BF16 for layers that benefit (Qwen2
        attention scores, RMSNorm). TRT picks per-layer between FP16 and BF16;
        BF16's fp32 exponent range avoids the underflow that drove our PT
        REF_DTYPE choice (bfloat16) and is the leading suspect for the TRT
        decode divergence (prefill cos_sim 0.945 vs PT bf16). A100 has BF16
        tensor cores; TRT 10.x supports BuilderFlag.BF16.
      - strongly_typed=True: build with NetworkDefinitionCreationFlag.STRONGLY_TYPED.
        ONNX dtypes are HONORED EXACTLY — TRT cannot promote/demote layer
        precisions for tactic selection. With BuilderFlag.BF16 alone, TRT may
        silently demote BF16 ops to FP16 if it judges a faster tactic exists
        (verified via polygraphy inspect --show layers). STRONGLY_TYPED removes
        that freedom: an ONNX op declared bf16 stays bf16. fp16/bf16 builder
        flags are then ignored (the ONNX is the source of truth).
        Required to keep LLM prefill/decode cos_sim above 0.99 vs PT bf16 —
        without it, prefill_cos saturates at ~0.945 (the empirical FP16
        precision-loss envelope on Qwen2 attention scores).
    """
    out_engine = Path(out_engine)
    if out_engine.exists():
        print(f"[trt] cached: {out_engine}")
        return out_engine

    inner_logger = logger or _DEFAULT_LOGGER
    cap_logger = _CapturingLogger(inner_logger)
    builder = trt.Builder(cap_logger)
    # Compose network flags. EXPLICIT_BATCH is required for ONNX-parsed networks
    # in TRT 10.x; STRONGLY_TYPED additionally locks layer precisions to ONNX.
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        if not hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            raise RuntimeError(
                f"NetworkDefinitionCreationFlag.STRONGLY_TYPED not available in "
                f"tensorrt=={trt.__version__}. Upgrade TRT to >=9.0 for STRONGLY_TYPED."
            )
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    net = builder.create_network(net_flags)
    parser = trt.OnnxParser(net, cap_logger)
    # parse_from_file (not parse(bytes)) so the parser can locate side-car external-data
    # files (.bin) that sit next to the .onnx (FP16 LLM + INT4 exports use them).
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    # In STRONGLY_TYPED mode, builder precision flags are ignored — ONNX dtypes
    # are the source of truth. Set them anyway for non-strongly_typed callers.
    if not strongly_typed:
        if fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
        if bf16:
            # TRT 10.x: BuilderFlag.BF16 allows BF16 compute on layers that support it.
            if not hasattr(trt.BuilderFlag, "BF16"):
                raise RuntimeError(
                    f"BuilderFlag.BF16 not available in tensorrt=={trt.__version__}. "
                    f"Upgrade TRT to >=10.0 for BF16 support."
                )
            cfg.set_flag(trt.BuilderFlag.BF16)

    prof = builder.create_optimization_profile()
    for n, (lo, opt, hi) in profile_spec.items():
        prof.set_shape(n, lo, opt, hi)
    cfg.add_optimization_profile(prof)

    print(f"[trt] building {name} (fp16={fp16}, bf16={bf16}, workspace={workspace_gb} GB) ...")
    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        captured = "\n  ".join(cap_logger.errors) if cap_logger.errors else "(no TRT ERROR logs captured)"
        raise RuntimeError(
            f"TRT build_serialized_network returned None for {name}.\n"
            f"Captured TRT ERROR(s):\n  {captured}\n\n"
            f"If captured says 'BF16 precision require hardware with BF16 support': "
            f"you're on pre-Ampere hardware (sm<80, e.g. Turing 2080 Ti). BF16 needs "
            f"sm_80+ (A100/H100/Ada). Either run on Ampere+ or pass bf16=False.\n"
            f"If captured is empty, the most common cause is profile_spec keys not "
            f"matching the ONNX input_names. Verify with: "
            f"`import onnx; print([i.name for i in onnx.load('{onnx_path}', load_external_data=False).graph.input])`"
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


def _decode_profile(n_layers: int, n_kv_heads: int, head_dim: int,
                     p_min: int, p_opt: int, p_max: int) -> Dict[str, Tuple[tuple, tuple, tuple]]:
    """Shared profile for both decode engines (MTP-branch and AR-branch).
    Both engines have identical I/O shapes; only the constant-folded attention
    branch inside the LM body differs."""
    prof: Dict[str, Tuple[tuple, tuple, tuple]] = {
        "input_ids":      ((1, 1), (1, 6),    (1, 6)),
        "position_ids":   ((1, 1), (1, 6),    (1, 6)),
        "attention_mask": ((1, 1), (1, p_opt + 6), (1, p_max + 6)),
    }
    for i in range(n_layers):
        prof[f"past_k_{i}"] = ((1, n_kv_heads, p_min, head_dim),
                                (1, n_kv_heads, p_opt, head_dim),
                                (1, n_kv_heads, p_max, head_dim))
        prof[f"past_v_{i}"] = ((1, n_kv_heads, p_min, head_dim),
                                (1, n_kv_heads, p_opt, head_dim),
                                (1, n_kv_heads, p_max, head_dim))
    return prof


def build_llm(prefill_onnx: Path, decode_onnx: Path,
              prefill_engine: Path, decode_engine: Path,
              hidden_size: int, n_layers: int, n_kv_heads: int, head_dim: int,
              decode_ar_onnx: Optional[Path] = None, decode_ar_engine: Optional[Path] = None,
              s_min: int = 16, s_opt: int = 1024, s_max: int = 4096,
              p_min: int = 0, p_opt: int = 1024, p_max: int = 4096,
              workspace_gb: int = 8, **kw) -> Tuple[Path, Path, Optional[Path]]:
    """Build prefill + (MTP-branch) decode + optional (AR-branch) decode engines.

    The two decode engines have identical I/O profiles; they differ only in the
    constant-folded SDLM block-mask branch inside the LM body. The MTP engine
    runs the block-mask attention path (correct for input `[last, mask×5]`);
    the AR engine runs the canonical AR attention path (correct for inputs with
    no text_mask_token_id at the last position, i.e. AR steps and KV-rebuild).

    Profile keys MUST match LLMPrefill / LLMDecode's forward parameter names.
    """
    H = hidden_size
    pref_prof = {
        "input_ids":       ((1, s_min), (1, s_opt), (1, s_max)),
        "visual_features": ((1, H),     (1024, H),  (s_max, H)),
        "position_ids":    ((1, s_min), (1, s_opt), (1, s_max)),
        "attention_mask":  ((1, s_min), (1, s_opt), (1, s_max)),
    }
    # STRONGLY_TYPED build: ONNX dtypes are honored exactly. Our bf16 ONNX
    # (bf16 initializers + bf16 KV-cache I/O + fp32 logits output) is the
    # canonical precision contract — STRONGLY_TYPED prevents TRT from silently
    # demoting bf16 attention/MatMul/Softmax/RMSNorm to fp16 during tactic
    # selection. Polygraphy inspect confirms this is what was happening with
    # the prior BuilderFlag.BF16-alone path (prefill cos_sim stuck at 0.945).
    # fp16/bf16 builder flags are ignored under STRONGLY_TYPED.
    llm_fp16 = kw.pop("fp16", True)
    llm_bf16 = kw.pop("bf16", True)
    llm_strongly_typed = kw.pop("strongly_typed", True)

    pre = build_engine(prefill_onnx, pref_prof, prefill_engine,
                       workspace_gb=workspace_gb, name="llm_prefill",
                       fp16=llm_fp16, bf16=llm_bf16,
                       strongly_typed=llm_strongly_typed, **kw)

    dec_prof = _decode_profile(n_layers, n_kv_heads, head_dim, p_min, p_opt, p_max)
    dec = build_engine(decode_onnx, dec_prof, decode_engine,
                       workspace_gb=workspace_gb, name="llm_decode_mtp",
                       fp16=llm_fp16, bf16=llm_bf16,
                       strongly_typed=llm_strongly_typed, **kw)

    dec_ar = None
    if decode_ar_onnx is not None and decode_ar_engine is not None:
        dec_ar = build_engine(decode_ar_onnx, dec_prof, decode_ar_engine,
                              workspace_gb=workspace_gb, name="llm_decode_ar",
                              fp16=llm_fp16, bf16=llm_bf16,
                              strongly_typed=llm_strongly_typed, **kw)

    return pre, dec, dec_ar
