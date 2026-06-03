#!/usr/bin/env python3
"""Patch nvidia/LocateAnything-3B HF config.json for TRT-LLM Qwen2 conversion.

The vendor checkpoint has model_type='locateanything' at root with the actual
Qwen2 LM body params nested in text_config. TRT-LLM's QWenConfig.from_hugging_face
requires model_type='qwen2' at root.

This script creates a sibling directory containing:
- Symlinks for all original files EXCEPT config.json
- A patched config.json with text_config hoisted to root + model_type=qwen2

Usage:
    python scripts/patch_locateanything_config_for_trtllm.py \
        --src /mnt/ssd0/locany_test_lvl2/weights \
        --dst /mnt/ssd0/locany_test_lvl2/weights_qwen2patched
"""
import argparse, json, os
from pathlib import Path


def patch(src: Path, dst: Path) -> Path:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if dst.exists():
        raise FileExistsError(f"{dst} exists — remove it first")
    dst.mkdir(parents=True)

    # Symlink everything except config.json
    for entry in src.iterdir():
        if entry.name == "config.json":
            continue
        (dst / entry.name).symlink_to(entry)

    # Build patched config.json
    src_cfg = json.loads((src / "config.json").read_text())
    text_cfg = src_cfg.get("text_config", {})
    if not text_cfg:
        raise RuntimeError(f"src config.json has no text_config block")

    patched = {**text_cfg}
    patched["model_type"] = "qwen2"
    patched["architectures"] = ["Qwen2ForCausalLM"]
    if "vocab_size" not in patched and "vocab_size" in src_cfg:
        patched["vocab_size"] = src_cfg["vocab_size"]
    # Force tie_word_embeddings=True (per project_locateanything_lm_head_root_cause
    # memory: vendor skips post_init -> lm_head random in safetensors; tied path
    # is the safe one)
    patched["tie_word_embeddings"] = True

    (dst / "config.json").write_text(json.dumps(patched, indent=2))

    # Sanity print
    print(f"[patch] wrote {dst}/config.json")
    for k in ["model_type", "architectures", "vocab_size", "hidden_size",
              "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "tie_word_embeddings", "max_position_embeddings",
              "rope_theta", "torch_dtype"]:
        if k in patched:
            print(f"  {k}: {patched[k]}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    patch(args.src, args.dst)
