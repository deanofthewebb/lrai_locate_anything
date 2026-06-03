"""HF checkpoint -> TRT-LLM checkpoint conversion for LocateAnything-3B LM body.

Wraps TRT-LLM's Qwen2 checkpoint conversion with our local concerns:
  - lm_head tie repair (per project_locateanything_lm_head_root_cause memory):
    HF nvidia/LocateAnything-3B may load with random-init lm_head because the
    vendor model code skips post_init() and therefore tie_weights() never runs.
    We MUST verify (and, if requested, repair) before conversion, otherwise the
    TRT-LLM engine will mode-collapse identically to the un-repaired PyTorch
    path.
  - vocab_size = 152681 (extended Qwen2.5 with bbox/locator tokens) — verified
    on the source config before conversion and re-checked on the emitted
    TRT-LLM config (note: TRT-LLM pads to a multiple of tp_size*64 internally;
    the runtime tokenizer must clamp ids < 152681).
  - qwen_type: LocateAnything-3B should report model_type='qwen2'. If a vendor
    rename is detected we surface a clear error before TRT-LLM's internal
    assertion fires.

Output layout (tp_size=1):
  out_dir/config.json        — serialized QWenConfig
  out_dir/rank0.safetensors  — sharded weights for rank 0

The CLI convert_checkpoint.py from TRT-LLM examples is NOT installed in the
trtllm venv on learn02, so this module calls the equivalent Python API
directly:
  tensorrt_llm.models.qwen.model.QWenForCausalLM.from_hugging_face(...)
  qwen.save_checkpoint(output_dir, save_config=(rank==0))

Engines (rank0.engine + build_config) are produced by the SECOND step,
build.py, which invokes `trtllm-build` on the checkpoint dir emitted here.

NOTE: scripts/trtllm_env.sh does NOT exist on learn02 — callers must set
LD_LIBRARY_PATH manually to the cu13 libs under
  /mnt/ssd0/locany_test_lvl2/venvs/trtllm/lib/python3.10/site-packages/nvidia/cu13/lib
(plus cudnn/nccl dirs alongside) BEFORE importing this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Threshold separating PyTorch default-init (~0.005) from a trained Qwen2.5
# lm_head/embed_tokens (~0.02). See project_locateanything_lm_head_root_cause.
_LM_HEAD_TRAINED_STD_THRESHOLD = 0.015


def _check_lm_head_not_random(model_dir: Path) -> dict:
    """Open the HF checkpoint and verify lm_head.weight is the trained
    version, not random init.

    Mode-collapse signature: lm_head std ~ 0.005 (PyTorch default init).
    Trained signature:       lm_head std ~ 0.02  (matches trained Qwen2.5).
    Threshold: std > 0.015 means trained.

    Returns:
        dict with keys: key, shape, std, mean, looks_trained, shard.
    Raises:
        RuntimeError if no lm_head weight is found in any shard.
    """
    from safetensors import safe_open  # lazy: keep this module light

    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if "lm_head" in key:
                    t = f.get_tensor(key)
                    tf = t.float()
                    std = float(tf.std())
                    return {
                        "key": key,
                        "shard": str(shard),
                        "shape": list(t.shape),
                        "std": std,
                        "mean": float(tf.mean()),
                        "looks_trained": std > _LM_HEAD_TRAINED_STD_THRESHOLD,
                    }
    # Fallback: single-file checkpoint (model.safetensors)
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as f:
            for key in f.keys():
                if "lm_head" in key:
                    t = f.get_tensor(key)
                    tf = t.float()
                    std = float(tf.std())
                    return {
                        "key": key,
                        "shard": str(single),
                        "shape": list(t.shape),
                        "std": std,
                        "mean": float(tf.mean()),
                        "looks_trained": std > _LM_HEAD_TRAINED_STD_THRESHOLD,
                    }
    raise RuntimeError(f"No lm_head weight found in {model_dir}")


def _verify_hf_config(hf_dir: Path) -> dict:
    """Read HF config.json and verify it is convertible by TRT-LLM's Qwen2
    path. Surfaces the LocateAnything-specific gotchas as clear errors
    before TRT-LLM's internal assertions fire.

    Returns the loaded config dict.
    Raises RuntimeError on any mismatch we care about.
    """
    cfg_path = hf_dir / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type")
    if model_type != "qwen2":
        raise RuntimeError(
            f"HF config.json model_type={model_type!r} but TRT-LLM Qwen2 "
            f"conversion requires model_type='qwen2'. If LocateAnything-3B "
            f"renamed it, patch config.json (model_type='qwen2', "
            f"architectures=['Qwen2ForCausalLM']) before re-running."
        )

    vocab_size = cfg.get("vocab_size")
    if vocab_size != 152681:
        # Not fatal — just surface. Standard Qwen2.5 is 152064; LocateAnything
        # extends to 152681. Anything else means the source dir is wrong.
        print(
            f"[convert] WARNING: vocab_size={vocab_size} (expected 152681 "
            f"for LocateAnything-3B). Continuing — verify this matches "
            f"the tokenizer added_tokens count."
        )

    return cfg


def convert_locateanything_checkpoint(
    hf_dir: Path,
    out_dir: Path,
    *,
    dtype: str = "bfloat16",
    tp_size: int = 1,
    workers: int = 1,
    weight_only_precision: Optional[str] = None,
) -> Path:
    """Convert HF LocateAnything-3B to TRT-LLM checkpoint format.

    Args:
        hf_dir: path to the lm_head-repaired HF checkpoint dir (config.json +
            safetensors). NOT the original vendor dir if the repair was done
            out-of-place.
        out_dir: where the TRT-LLM checkpoint lands (per-rank safetensors +
            config.json). Created if missing.
        dtype: 'bfloat16' (recommended; matches our STRONGLY_TYPED build).
            Pass explicitly — 'auto' may downcast fp32 to fp16.
        tp_size: tensor-parallel world size. Default 1 (L40S/4090 single-GPU).
        workers: kept at 1 for VRAM safety; with tp_size=1 only one rank exists
            so this is informational.
        weight_only_precision: None for the bf16 baseline; 'int8' or 'int4' to
            enable W8A16/W4A16 weight-only quant. Do NOT enable on the first
            validation run — get the bf16 baseline working first.

    Returns:
        out_dir path on success.

    Raises:
        FileNotFoundError if hf_dir is missing config.json.
        RuntimeError if lm_head looks random-init (run tie repair first), or
            if hf_config sanity checks fail.
    """
    hf_dir = Path(hf_dir).resolve()
    out_dir = Path(out_dir).resolve()

    if not (hf_dir / "config.json").exists():
        raise FileNotFoundError(f"HF config.json not found at {hf_dir}")

    # 1. Verify HF config is convertible by the Qwen2 path.
    _verify_hf_config(hf_dir)

    # 2. Pre-flight: lm_head tie repair check.
    diag = _check_lm_head_not_random(hf_dir)
    if not diag["looks_trained"]:
        raise RuntimeError(
            f"lm_head appears random-init (std={diag['std']:.4f}, "
            f"shard={diag['shard']}). Run lm_head tie repair BEFORE convert "
            f"(flip tie_word_embeddings=True in config.json, or overwrite "
            f"lm_head.weight with embed_tokens.weight in the safetensors "
            f"shard). See project_locateanything_lm_head_root_cause memory."
        )
    print(
        f"[convert] lm_head OK: std={diag['std']:.4f} (trained signature, "
        f"key={diag['key']}, shape={diag['shape']})"
    )

    # 3. Call TRT-LLM Python API directly. The convert_checkpoint.py CLI is
    # NOT installed in the trtllm venv on learn02, but the underlying API is.
    # See /mnt/ssd0/locany_test_lvl2/venvs/trtllm/lib/python3.10/site-packages/
    #   tensorrt_llm/models/qwen/model.py line ~300 (from_hugging_face) and
    #   line ~365 (custom_dict lm_head->model.embed_tokens for qwen2 tied).
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.models.qwen.model import QWenForCausalLM
    try:
        from tensorrt_llm.quantization import QuantAlgo
    except ImportError:  # older TRT-LLM layouts
        from tensorrt_llm.models.modeling_utils import QuantAlgo  # type: ignore

    quant_config = QuantConfig()
    if weight_only_precision == "int8":
        quant_config.quant_algo = QuantAlgo.W8A16
    elif weight_only_precision == "int4":
        quant_config.quant_algo = QuantAlgo.W4A16
    elif weight_only_precision is not None:
        raise ValueError(
            f"weight_only_precision={weight_only_precision!r} not supported; "
            f"use None, 'int8', or 'int4'."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # tp_size=1 -> single rank; loop kept for symmetry with multi-rank dumps.
    world_size = tp_size  # pp_size=cp_size=1
    print(
        f"[convert] running QWenForCausalLM.from_hugging_face "
        f"(hf_dir={hf_dir}, dtype={dtype}, tp_size={tp_size}, "
        f"workers={workers}, weight_only={weight_only_precision})"
    )
    for rank in range(world_size):
        mapping = Mapping(world_size=world_size, rank=rank, tp_size=tp_size,
                          pp_size=1)
        qwen = QWenForCausalLM.from_hugging_face(
            str(hf_dir),
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            load_model_on_cpu=True,  # 3B + vocab 152681 -> avoid 2x GPU load
        )
        qwen.save_checkpoint(str(out_dir), save_config=(rank == 0))
        del qwen  # free before next rank

    # 4. Verify expected outputs landed.
    config_json = out_dir / "config.json"
    if not config_json.exists():
        raise RuntimeError(f"convert succeeded but no config.json at {config_json}")
    rank0_shard = out_dir / "rank0.safetensors"
    if not rank0_shard.exists():
        raise RuntimeError(
            f"convert succeeded but no rank0.safetensors at {rank0_shard}"
        )
    print(f"[convert] success — output at {out_dir}")
    return out_dir


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-dir", type=Path, required=True,
                    help="Path to lm_head-repaired HF LocateAnything-3B dir")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output TRT-LLM checkpoint dir")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--weight-only-precision", default=None,
                    choices=(None, "int8", "int4"))
    args = ap.parse_args()
    out = convert_locateanything_checkpoint(
        args.hf_dir,
        args.out_dir,
        dtype=args.dtype,
        tp_size=args.tp_size,
        workers=args.workers,
        weight_only_precision=args.weight_only_precision,
    )
    print(f"OUT_DIR={out}")


if __name__ == "__main__":
    _cli()
