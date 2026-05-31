"""lrai_locate_anything — modular runtime for NVIDIA LocateAnything-3B.

Public API
----------
LocateAnythingRunner   high-level: load model, export, build engines, run inference
run_image, run_video   pipeline helpers for single-image / per-frame inference
parse_boxes, iou       output post-processing
"""
from .config import WORK, MODEL_ID, REF_DTYPE, ONNX_DIR, TRT_DIR, ensure_nvidia_stack
from .orchestrator import LocateAnythingRunner
from .pipelines import run_image, run_video
from .parse import parse_boxes, iou, python_patch_merger

__version__ = "0.1.0"
__all__ = [
    "LocateAnythingRunner",
    "run_image",
    "run_video",
    "parse_boxes",
    "iou",
    "python_patch_merger",
    "WORK",
    "MODEL_ID",
    "REF_DTYPE",
    "ONNX_DIR",
    "TRT_DIR",
    "ensure_nvidia_stack",
]
