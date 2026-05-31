"""lrai_locate_anything — modular runtime for NVIDIA LocateAnything-3B.

Public API
----------
LocateAnythingRunner     high-level: load, export, build engines, run inference
run_image, run_video     pipeline helpers for single-image / per-frame inference
run_compare              side-by-side multi-runtime video comparison
parse_boxes, iou         output post-processing

Optional submodules
-------------------
benchmark                bench_text_runtimes, bench_image, bench_video_compare
trtllm                   install_trtllm, dump_qwen2_lm_only, convert_and_build, TRTLLMRunner
trt.plugins              source + build instructions for the packed-varlen attention plugin
export.int4              optional INT4 AWQ via NVIDIA Model Optimizer (see caveats)
"""
from .config import WORK, MODEL_ID, REF_DTYPE, ONNX_DIR, TRT_DIR, ensure_nvidia_stack, ensure_runtime_deps
from .orchestrator import LocateAnythingRunner
from .pipelines import run_image, run_video, run_compare
from .parse import parse_boxes, iou, python_patch_merger

__version__ = "0.2.0"
__all__ = [
    "LocateAnythingRunner",
    "run_image",
    "run_video",
    "run_compare",
    "parse_boxes",
    "iou",
    "python_patch_merger",
    "WORK",
    "MODEL_ID",
    "REF_DTYPE",
    "ONNX_DIR",
    "TRT_DIR",
    "ensure_nvidia_stack",
    "ensure_runtime_deps",
]
