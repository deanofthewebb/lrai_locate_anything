"""HF -> TRT-LLM checkpoint conversion for LocateAnything-3B.

Distinct from `lrai_locate_anything/trtllm/convert.py`, which dumps the
inner Qwen2 LM for text-only benchmarking. This module converts the
*full* multimodal model: the Qwen2 weights are emitted in TRT-LLM's
per-rank shard format, and the SDLM block-mask metadata + visual
prompt-table dimensions are encoded into config.json so the build
step (build.py) can wire the prompt_embedding_table input.
"""
from __future__ import annotations
from pathlib import Path


def convert_locateanything_checkpoint(
    hf_dir: Path,
    out_dir: Path,
    dtype: str = "bfloat16",
    tp_size: int = 1,
) -> Path:
    """Convert a HF LocateAnything-3B checkpoint to TRT-LLM format.

    Reads from `hf_dir` (the canonical nvidia/LocateAnything-3B layout):
      - config.json                       (full LocateAnything config, incl. text_config)
      - model-*.safetensors               (sharded HF weights for vision + projector + LM)
      - tokenizer.json + tokenizer_config.json + special_tokens_map.json
      - generation_config.json

    Writes to `out_dir`:
      - config.json                       (TRT-LLM Qwen2 schema + LocateAnything extras:
                                            prompt_vocab_size, vision_hidden, sdlm_block_mask_shape)
      - rank{0..tp_size-1}.safetensors    (per-rank sharded LM weights)
      - tokenizer/                        (copied through unchanged)

    The inner Qwen2 LM is run through TRT-LLM's stock convert utility
    (tensorrt_llm.commands.convert_checkpoint or the Python-API
    Qwen2ForCausalLM.from_hugging_face fallback — see trtllm/convert.py
    for the precedent). Vision tower + projector are NOT converted here;
    they are handled by the existing export_prod path and consumed by
    MoonViTAdapter at runtime.

    Args:
        hf_dir:   Path to the HF checkpoint directory.
        out_dir:  Destination for the TRT-LLM checkpoint shards + config.
        dtype:    Weight dtype ("bfloat16" | "float16" | "float32").
                  Default bf16 to match the export_prod vision_proj engine.
        tp_size:  Tensor-parallel world size. Default 1 (single GPU).

    Returns:
        Path to `out_dir` once conversion is complete.

    Raises:
        NotImplementedError: scaffolding only.
    """
    raise NotImplementedError("trtllm_prod.convert is scaffolding only")
