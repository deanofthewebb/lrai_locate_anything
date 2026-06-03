"""Adapter from our MoonViT/projector output to TRT-LLM's prompt-embedding input.

The vision_proj engine (built by export_prod) emits visual features in
LocateAnything's native layout: (L_post, hidden), where L_post is the
post-projector token count for a single image and hidden matches the
Qwen2 text-embedding dim.

TRT-LLM's ModelRunner.generate accepts a `prompt_table` argument of
shape (batch=1, L_post, hidden) plus a `prompt_tasks` index and a
`prompt_vocab_size` configured at build time. This adapter reshapes
and positions the features so they land in the virtual-token slots
reserved by build.build_llm_engine.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image


class MoonViTAdapter:
    """Bridges our existing vision_proj.engine output to TRT-LLM prompt tables.

    Format conversion:
        ours:     vision_proj.engine -> Tensor[L_post, hidden]   (bf16, CUDA)
        theirs:   prompt_table       -> Tensor[1, L_post, hidden] with the
                  same dtype as the LLM engine and an additive positional
                  offset baked in (TRT-LLM expects raw token embeddings,
                  not embeddings + RoPE — RoPE is applied inside the engine).
    """

    def __init__(self, vision_proj_engine: Path):
        """Load the prebuilt vision_proj.engine.

        Args:
            vision_proj_engine: Path to the export_prod-built TRT engine
                that fuses MoonViT + projector into a single graph.

        Raises:
            NotImplementedError: scaffolding only.
        """
        raise NotImplementedError("MoonViTAdapter is scaffolding only")

    def forward(self, image: "Image.Image") -> "torch.Tensor":
        """Run vision_proj on a single PIL image and reshape for TRT-LLM.

        Pipeline:
            PIL.Image
              -> letterbox + normalize (matches export_prod's input contract)
              -> vision_proj.engine.execute() -> (L_post, hidden)
              -> unsqueeze(0)                 -> (1, L_post, hidden)
              -> contiguous, dtype-aligned to the LLM engine

        Args:
            image: A single RGB PIL image.

        Returns:
            torch.Tensor of shape (1, L_post, hidden), on CUDA, in the
            LLM engine's plugin dtype.

        Raises:
            NotImplementedError: scaffolding only.
        """
        raise NotImplementedError("MoonViTAdapter.forward is scaffolding only")
