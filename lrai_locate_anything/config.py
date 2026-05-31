"""Global configuration: workdir detection, model constants, NVIDIA-stack install probe.

Anything that touches the filesystem layout or the model identity lives here so the
rest of the package doesn't pin hard-coded paths.
"""
from __future__ import annotations
import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import Optional

import torch

MODEL_ID = "nvidia/LocateAnything-3B"
REF_DTYPE = torch.float16

# MoonViT structural constants (do not change unless the upstream model does)
ENG_PATCH_SIZE = 14
ENG_MERGE_KH = 2
ENG_MERGE_KW = 2


def _on_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _default_workdir() -> Path:
    if _on_colab():
        return Path("/content/locany")
    env = os.environ.get("LRAI_WORKDIR")
    if env:
        return Path(env)
    return Path.home() / "locany"


WORK = _default_workdir()
WORK.mkdir(parents=True, exist_ok=True)

ONNX_DIR = WORK / "onnx"
TRT_DIR = WORK / "engines"
WEIGHTS_DIR = WORK / "weights"
for _d in (ONNX_DIR, TRT_DIR, WEIGHTS_DIR):
    _d.mkdir(exist_ok=True)


def _sh(cmd: str, check: bool = True) -> None:
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode and check:
        print(r.stderr[-3000:])
        raise RuntimeError(f"command failed: {cmd}")


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def ensure_nvidia_stack(verbose: bool = True) -> None:
    """Install the NVIDIA wheel stack if missing. Idempotent.

    - tensorrt 10.7.* + cuda-python 12.6+: required for engine build/run
    - onnxruntime-gpu: required for the parity test path
    - polygraphy: diagnostics
    The mapping below pins versions known to work with CUDA 12 drivers as of May 2026.
    """
    pkgs: list[str] = []
    if not _have("tensorrt"):
        pkgs.append('"tensorrt==10.7.*"')
        pkgs.append('"tensorrt-cu12==10.7.*"')
    if not _have("cuda.bindings"):
        pkgs.append('"cuda-python>=12.6,<13"')
    if not _have("polygraphy"):
        pkgs.append("polygraphy")
    # onnxruntime-gpu detection has to probe the provider list, not the module name —
    # Colab pre-installs CPU onnxruntime which would skip the GPU install.
    try:
        import onnxruntime as _ort
        if "CUDAExecutionProvider" not in _ort.get_available_providers():
            _sh("pip -q uninstall -y onnxruntime onnxruntime-gpu", check=False)
            pkgs.append("onnxruntime-gpu")
    except ImportError:
        pkgs.append("onnxruntime-gpu")

    if pkgs:
        if verbose:
            print(f"[lrai] installing: {' '.join(pkgs)}")
        _sh(f"pip -q install --no-input {' '.join(pkgs)}")
        # Purge stale module entries so re-imports pick up new wheels.
        for k in [k for k in list(sys.modules) if k.split(".")[0] in {"tensorrt", "cuda", "onnxruntime"}]:
            sys.modules.pop(k, None)
        importlib.invalidate_caches()

    # Hard-assert the exact submodule the orchestrator uses; surface ABI breakage early.
    from cuda.bindings import runtime as _cudart  # noqa: F401
    if verbose:
        import tensorrt as trt
        print(f"[lrai] tensorrt {trt.__version__}, cuda.bindings OK")


def gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def enable_llm_trt() -> bool:
    """Return True if VRAM is sufficient for the full FP16 LLM TRT engine build (>=22 GB).
    Below that, the orchestrator falls back to PyTorch for the LLM portion.
    """
    return gpu_vram_gb() >= 22.0
