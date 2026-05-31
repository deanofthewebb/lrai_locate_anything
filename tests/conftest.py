"""Pytest fixtures + markers.

Markers:
  gpu       — needs an NVIDIA GPU; skipped on CPU-only hosts
  trt       — needs TensorRT installed
  heavy     — needs the full 3B model loaded (downloads weights on first run)
  trtllm    — needs tensorrt-llm installed (per torch-version match)
"""
import os
import sys
import pytest


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAVE_TORCH = _have("torch")
HAVE_CUDA = HAVE_TORCH and __import__("torch").cuda.is_available()
HAVE_TRT = _have("tensorrt")
HAVE_TRT_LLM = _have("tensorrt_llm")
HAVE_MODELOPT = _have("modelopt")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA-capable GPU")
    config.addinivalue_line("markers", "trt: requires tensorrt installed")
    config.addinivalue_line("markers", "trtllm: requires tensorrt_llm installed")
    config.addinivalue_line("markers", "heavy: requires full LocateAnything-3B model loaded")


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    skip_trt = pytest.mark.skip(reason="tensorrt not installed")
    skip_trtllm = pytest.mark.skip(reason="tensorrt_llm not installed")
    skip_heavy = pytest.mark.skip(reason="skipping heavy test (set LRAI_RUN_HEAVY=1 to enable)")
    run_heavy = os.environ.get("LRAI_RUN_HEAVY") == "1"
    for item in items:
        if "gpu" in item.keywords and not HAVE_CUDA:
            item.add_marker(skip_gpu)
        if "trt" in item.keywords and not HAVE_TRT:
            item.add_marker(skip_trt)
        if "trtllm" in item.keywords and not HAVE_TRT_LLM:
            item.add_marker(skip_trtllm)
        if "heavy" in item.keywords and not run_heavy:
            item.add_marker(skip_heavy)


@pytest.fixture(scope="session")
def torch_module():
    if not HAVE_TORCH:
        pytest.skip("torch not available")
    import torch
    return torch
