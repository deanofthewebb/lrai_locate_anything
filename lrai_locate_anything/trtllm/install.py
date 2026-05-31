"""TRT-LLM install: torch-version-aware version probing.

TRT-LLM ships precompiled bindings linked against a specific libtorch ABI. A version
mismatch surfaces at IMPORT time as `undefined symbol: _ZN3c105ErrorC2ENS_...`. Each
TRT-LLM minor pins to a narrow PyTorch range; we probe to pick a wheel that loads.

Empirical compatibility (NVIDIA wheel index, mid-2026):
  torch 2.4  → tensorrt-llm 0.16–0.17
  torch 2.5  → tensorrt-llm 0.18–0.19
  torch 2.6  → tensorrt-llm 0.20–0.21
  torch 2.7+ → tensorrt-llm 0.22+

Each wheel is ~1.9 GB; we probe at most 2 candidates per torch version.
"""
from __future__ import annotations
import importlib
import subprocess
import sys
from typing import List, Optional


def _sh(cmd: str, check: bool = False) -> int:
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-2000:])
    if r.returncode and check:
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {cmd}")
    return r.returncode


def probe_compatible_versions(torch_version: Optional[str] = None) -> List[str]:
    """Return the list of TRT-LLM versions to try, in preferred-first order."""
    if torch_version is None:
        import torch
        torch_version = torch.__version__
    tv = tuple(int(x) for x in torch_version.split("+")[0].split(".")[:2])
    if tv >= (2, 7):
        return ["0.22.0", "0.21.0"]
    if tv == (2, 6):
        return ["0.21.0", "0.20.0"]
    if tv == (2, 5):
        return ["0.19.0", "0.18.0"]
    if tv == (2, 4):
        return ["0.17.0", "0.16.0"]
    return ["0.17.0"]


def _import_test() -> object:
    """Re-import after a pip change, purging any half-loaded module from sys.modules."""
    for k in list(sys.modules):
        if k == "tensorrt_llm" or k.startswith("tensorrt_llm."):
            del sys.modules[k]
    importlib.invalidate_caches()
    try:
        import tensorrt_llm
        return tensorrt_llm
    except Exception as e:
        return e


def install_trtllm(verbose: bool = True) -> bool:
    """Install + import-test a TRT-LLM wheel compatible with the active PyTorch.

    Returns True if a working wheel was installed and imports cleanly.
    """
    # 1) Already installed?
    try:
        importlib.import_module("tensorrt_llm")
        r = _import_test()
        if not isinstance(r, Exception):
            if verbose:
                print(f"tensorrt-llm {r.__version__} already installed and importable")
            return True
        if verbose:
            print(f"tensorrt-llm raises at import: {type(r).__name__}: {str(r)[:120]}")
            print("  uninstalling and probing a compatible version ...")
        _sh("pip -q uninstall -y tensorrt-llm")
    except ImportError:
        pass

    candidates = probe_compatible_versions()
    if verbose:
        import torch
        print(f"torch={torch.__version__}  candidates={candidates}")

    for ver in candidates:
        if verbose:
            print(f"trying tensorrt-llm=={ver} ...")
        # --no-deps so pip doesn't upgrade torch under us (which would break the rest
        # of the package). TRT-LLM's hard deps (tensorrt, cuda-python) come from our extras.
        _sh(
            f'pip -q install --force-reinstall --no-deps '
            f'"tensorrt-llm=={ver}" --extra-index-url https://pypi.nvidia.com'
        )
        r = _import_test()
        if not isinstance(r, Exception):
            if verbose:
                print(f"  tensorrt-llm {r.__version__} loaded OK")
            return True
        if verbose:
            print(f"  {ver}: {type(r).__name__}: {str(r)[:160]}")

    if verbose:
        print()
        print("Could not find a tensorrt-llm wheel matching this PyTorch.")
        print("Workaround: pin torch to a version listed at")
        print("  https://nvidia.github.io/TensorRT-LLM/installation/linux.html")
    return False
