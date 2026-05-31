"""TRT-LLM parallel path: install, checkpoint conversion, engine build, runtime wrapper.

This is an OPTIONAL accelerator. The base package (vision + projector + LLM via plain
TRT engines) is fully functional without it. Use TRT-LLM when you want:

  - Paged-KV cache (lower memory at long contexts)
  - Fused FlashAttention-2/3 kernels
  - Native INT4 AWQ that actually builds (vs the modelopt-export route that hits
    TRT 10's `kBLOCKED requires kINT4` rejection — see export/int4.py docstring)
  - In-flight batching (only relevant at concurrency >1)

Limitation: TRT-LLM's stock Qwen2 doesn't carry the LocateAnything `visual_features`
injection. For multimodal inference you'd port the SDLM block-mask as a custom op;
for **text-only LM benchmarking** (the canonical apples-to-apples runtime comparison)
this path is direct.
"""
from .install import install_trtllm, probe_compatible_versions
from .convert import dump_qwen2_lm_only, convert_and_build
from .runner import TRTLLMRunner

__all__ = [
    "install_trtllm",
    "probe_compatible_versions",
    "dump_qwen2_lm_only",
    "convert_and_build",
    "TRTLLMRunner",
]
