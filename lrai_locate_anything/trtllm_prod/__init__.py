"""TRT-LLM-based production deployment for LocateAnything-3B.

This submodule wraps NVIDIA TensorRT-LLM as the LLM runtime, replacing
the prefill + decode_ar + decode_mtp engines from lrai_locate_anything/trt/
with a single trtllm-built engine that handles both regimes via paged
attention. See docs/trtllm_adoption_design.md for full design.

Status: SCAFFOLDING ONLY. No functional implementations yet.

Public API (all NotImplementedError at this stage):
    convert:  HF nvidia/LocateAnything-3B checkpoint -> TRT-LLM format
    build:    trtllm-build -> single llm.engine
    runner:   single-image inference via vision_proj.engine + llm.engine
    moonvit:  adapter between our MoonViT output and TRT-LLM's input expectations
"""
from .convert import convert_locateanything_checkpoint
from .build import build_llm_engine
from .moonvit_adapter import MoonViTAdapter
from .runner import LocateAnythingTRTLLMRunner

__all__ = [
    "convert_locateanything_checkpoint",
    "build_llm_engine",
    "MoonViTAdapter",
    "LocateAnythingTRTLLMRunner",
]
