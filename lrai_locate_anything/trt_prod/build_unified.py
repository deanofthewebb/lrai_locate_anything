"""TRT builders for the fused production engines.

Phase 1: build_vision_proj_engine — one engine for vision + merger + mlp1.
Phase 3 (TODO): build_llm_unified_engine — one engine for prefill + decode.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import tensorrt as trt

from lrai_locate_anything.trt.build import build_engine


def build_vision_proj_engine(
    onnx_path: Path,
    engine_path: Path,
    L_pre: int,
    pixel_shape_tail: tuple = (3, 14, 14),
    workspace_gb: int = 4,
    logger: Optional[trt.Logger] = None,
) -> Path:
    """Build the fused vision_proj engine.

    Fixed shape (min=opt=max) on pixel_values because pos_emb_baked
    inside VisionForExport is resolution-locked. STRONGLY_TYPED + FP16
    so the ONNX fp16 dtype is honored exactly — no silent tactic-driven
    precision drift on LayerNorm / GELU inside mlp1.
    """
    shape = (L_pre,) + pixel_shape_tail
    profile_spec = {"pixel_values": (shape, shape, shape)}
    return build_engine(
        onnx_path,
        profile_spec,
        engine_path,
        fp16=True,
        bf16=False,
        strongly_typed=True,
        workspace_gb=workspace_gb,
        name="vision_proj",
        logger=logger,
    )
