"""Checkpoint conversion + engine build for TRT-LLM.

The LocateAnything model wraps Qwen2.5-3B inside a custom outer class. TRT-LLM's
convert_checkpoint understands only stock Qwen2ForCausalLM, so we dump the inner
LM to a standalone HF checkpoint first, then convert + build from there.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

from ..config import WORK


def _sh(cmd: str, check: bool = False) -> int:
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True)
    if r.returncode and check:
        raise RuntimeError(f"command failed: {cmd}")
    return r.returncode


def dump_qwen2_lm_only(model, tokenizer, config, target_dir: Path | None = None) -> Path:
    """Save the inner Qwen2 LM (model.language_model) as a standalone HF checkpoint.

    config.text_config is dumped as the model's config.json with architectures=['Qwen2ForCausalLM'].
    Tokenizer is copied. Result is consumable by TRT-LLM's convert_checkpoint.
    """
    target_dir = Path(target_dir) if target_dir else (WORK / "qwen2_lm_only")
    if (target_dir / "config.json").exists():
        print(f"[trtllm] cached: {target_dir}")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    cfg_dict = config.text_config.to_dict()
    cfg_dict["architectures"] = ["Qwen2ForCausalLM"]
    with open(target_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[trtllm] saved standalone Qwen2 LM → {target_dir}")
    return target_dir


def convert_and_build(
    qwen_hf_dir: Path,
    trtllm_ckpt_dir: Path | None = None,
    trtllm_engine_dir: Path | None = None,
    dtype: str = "float16",
    tp_size: int = 1,
    max_batch_size: int = 1,
    max_input_len: int = 4096,
    max_seq_len: int = 4608,
    verbose: bool = True,
) -> Path:
    """Convert HF → TRT-LLM checkpoint, then run trtllm-build.

    Idempotent: returns the existing engine directory if one is present.
    Falls back to the Python API if the CLI conversion fails (the CLI signature has
    changed twice across recent TRT-LLM releases).
    """
    trtllm_ckpt_dir = Path(trtllm_ckpt_dir) if trtllm_ckpt_dir else (WORK / "trtllm_ckpt")
    trtllm_engine_dir = Path(trtllm_engine_dir) if trtllm_engine_dir else (WORK / "trtllm_engine")

    if list(trtllm_engine_dir.glob("*.engine")) or (trtllm_engine_dir / "rank0.engine").exists():
        if verbose:
            print(f"[trtllm] cached engine: {trtllm_engine_dir}")
        return trtllm_engine_dir

    if not (trtllm_ckpt_dir / "config.json").exists():
        if verbose:
            print("[trtllm] converting HF → TRT-LLM checkpoint ...")
        rc = _sh(
            f'python -c "from tensorrt_llm.commands.convert_checkpoint import main; main()" '
            f'--model_dir {qwen_hf_dir} '
            f'--output_dir {trtllm_ckpt_dir} '
            f'--dtype {dtype} '
            f'--tp_size {tp_size}'
        )
        if rc != 0 or not (trtllm_ckpt_dir / "config.json").exists():
            if verbose:
                print("[trtllm] CLI conversion failed; trying Python API fallback ...")
            try:
                from tensorrt_llm.models import Qwen2ForCausalLM as _Q
                m = _Q.from_hugging_face(str(qwen_hf_dir), dtype=dtype)
                m.save_checkpoint(str(trtllm_ckpt_dir))
                if verbose:
                    print("[trtllm]  Python-API conversion OK")
            except Exception as e:
                raise RuntimeError(f"both convert paths failed: {e}")
    elif verbose:
        print(f"[trtllm] cached checkpoint: {trtllm_ckpt_dir}")

    if verbose:
        print("[trtllm] building engine (10-20 min) ...")
    t0 = time.time()
    rc = _sh(
        f"trtllm-build "
        f"--checkpoint_dir {trtllm_ckpt_dir} "
        f"--output_dir {trtllm_engine_dir} "
        f"--gemm_plugin {dtype} "
        f"--gpt_attention_plugin {dtype} "
        f"--max_batch_size {max_batch_size} "
        f"--max_input_len {max_input_len} "
        f"--max_seq_len {max_seq_len} "
        f"--use_paged_context_fmha enable"
    )
    if rc != 0 or not (list(trtllm_engine_dir.glob("*.engine")) or (trtllm_engine_dir / "rank0.engine").exists()):
        raise RuntimeError("trtllm-build did not produce an engine; see logs above")
    if verbose:
        print(f"[trtllm]  built in {time.time()-t0:.0f}s → {trtllm_engine_dir}")
    return trtllm_engine_dir
