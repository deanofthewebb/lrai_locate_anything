"""End-to-end single-image inference for LocateAnything-3B on TRT-LLM.

Replaces the orchestrator's three-engine prefill+decode loop with a
single ModelRunner.generate call that takes (text_tokens, vision_prompt_table)
and returns the full output sequence.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from PIL import Image


class LocateAnythingTRTLLMRunner:
    """Single-image LocateAnything detection on TRT-LLM.

    Inference flow:
        1. tokenizer(prompt)                  -> input_ids
        2. MoonViTAdapter.forward(image)      -> prompt_table (1, L_post, hidden)
        3. tensorrt_llm.runtime.ModelRunner.generate(
               batch_input_ids=[input_ids],
               prompt_table=prompt_table,
               prompt_tasks=[0],
               max_new_tokens=...,
               ...)                            -> output_ids
        4. tokenizer.decode(output_ids)        -> raw text
        5. parse.parse_boxes_with_labels(text) -> List[(bbox, label)]
    """

    def __init__(
        self,
        llm_engine_path: Path,
        vision_proj_engine_path: Path,
        hf_dir: Path,
    ):
        """Wire up the engines + tokenizer.

        Args:
            llm_engine_path:         The .engine built by build.build_llm_engine.
            vision_proj_engine_path: The export_prod vision_proj.engine path.
            hf_dir:                  HF checkpoint dir (for tokenizer + generation_config).

        Raises:
            NotImplementedError: scaffolding only.
        """
        raise NotImplementedError("LocateAnythingTRTLLMRunner is scaffolding only")

    def detect(
        self,
        image: "Image.Image",
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[List[Tuple[tuple, str]], str]:
        """Run a single grounding query against the image.

        Args:
            image:          RGB PIL image.
            prompt:         User instruction (e.g. "locate the red car").
            max_new_tokens: Decode-side cap. Must be <= the engine's max_output_len.
            temperature:    0.0 = greedy. Anything >0 enables sampling.
            top_p:          Nucleus sampling cap. Ignored when temperature == 0.

        Returns:
            (detections, raw_text) where:
              detections is a list of ((x1, y1, x2, y2), label) tuples in
              the original image's pixel coordinates, and
              raw_text is the un-parsed decoded string (for debug / logging).

        Raises:
            NotImplementedError: scaffolding only.
        """
        raise NotImplementedError("LocateAnythingTRTLLMRunner.detect is scaffolding only")
