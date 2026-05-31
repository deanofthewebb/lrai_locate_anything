"""Minimal wrapper around tensorrt_llm.runtime.ModelRunner.

Sufficient for runtime benchmarking (text-only AR). For multimodal grounding via
TRT-LLM you'd extend with a prompt_embedding_table parameter that injects vision
features as virtual prompt tokens; that's an additional ~200 LOC including PBD
support and is intentionally out of scope here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


class TRTLLMRunner:
    """Wraps tensorrt_llm.runtime.ModelRunner for AR generation."""

    def __init__(self, engine_dir: Path | str, tokenizer, eos_id: Optional[int] = None, pad_id: int = 0):
        from tensorrt_llm.runtime import ModelRunner
        self.runner = ModelRunner.from_dir(
            engine_dir=str(engine_dir),
            rank=0,
            debug_mode=False,
        )
        self.tokenizer = tokenizer
        self.eos_id = int(eos_id if eos_id is not None else tokenizer.eos_token_id)
        self.pad_id = int(pad_id)

    def generate(self, input_ids_torch, max_new_tokens: int = 128,
                 temperature: float = 0.0, top_p: float = 1.0):
        """Generate from torch input_ids of shape (1, S). Returns the runner's dict output."""
        return self.runner.generate(
            batch_input_ids=[input_ids_torch[0]],
            max_new_tokens=max_new_tokens,
            end_id=self.eos_id,
            pad_id=self.pad_id,
            temperature=temperature,
            top_p=top_p,
            output_sequence_lengths=True,
            return_dict=True,
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 128, **kw) -> str:
        """Convenience: tokenise prompt, run generate, decode the new tokens."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        out = self.generate(ids, max_new_tokens=max_new_tokens, **kw)
        # out['output_ids']: (batch, beam, seq); strip the prompt tokens.
        try:
            seq = out["output_ids"][0][0]
            new = seq[ids.shape[1]:]
            return self.tokenizer.decode(new, skip_special_tokens=False)
        except Exception:
            return str(out)
