"""TRT-LLM engine build for LocateAnything-3B.

Thin wrapper around `trtllm-build` (or its Python equivalent
tensorrt_llm.commands.build.main) that consumes the checkpoint
emitted by convert.convert_locateanything_checkpoint.

The resulting engine handles BOTH prefill and decode in one graph
via paged-KV attention, collapsing the current three-engine
(prefill / decode_ar / decode_mtp) layout in lrai_locate_anything/trt/.
"""
from __future__ import annotations
from pathlib import Path


def build_llm_engine(
    checkpoint_dir: Path,
    engine_path: Path,
    max_input_len: int = 4096,
    max_output_len: int = 128,
    max_batch_size: int = 1,
    dtype: str = "bfloat16",
) -> Path:
    """Build a single LocateAnything LLM engine via trtllm-build.

    Equivalent CLI:
        trtllm-build \
            --checkpoint_dir <checkpoint_dir> \
            --output_dir <engine_path.parent> \
            --gemm_plugin <dtype> \
            --gpt_attention_plugin <dtype> \
            --max_batch_size <max_batch_size> \
            --max_input_len <max_input_len> \
            --max_seq_len <max_input_len + max_output_len> \
            --use_paged_context_fmha enable \
            --max_prompt_embedding_table_size <prompt_vocab_size from config.json>

    The last flag is what differentiates this from the text-only build in
    trtllm/convert.py: it reserves a virtual-token slot for the visual
    features that MoonViTAdapter will inject at runtime.

    Args:
        checkpoint_dir:  Output of convert_locateanything_checkpoint.
        engine_path:     Target path for the .engine file. Parent directory
                         also receives config.json + rank-shard metadata.
        max_input_len:   Max prompt-side tokens (text + virtual vision tokens).
        max_output_len:  Max generated tokens. max_seq_len = sum of the two.
        max_batch_size:  Concurrency upper bound. Default 1 (single-image inference).
        dtype:           Plugin dtype. Must match the checkpoint's weight dtype.

    Returns:
        Path to the built engine (engine_path).

    Raises:
        NotImplementedError: scaffolding only.
    """
    raise NotImplementedError("trtllm_prod.build is scaffolding only")
