"""TRTEngine: a thin Python wrapper around tensorrt.IExecutionContext.

Uses cuda-python (cuda.bindings.runtime) for memory + streams. set_input_shape's
return value IS checked — TRT 10 returns False (not an exception) when a shape
falls outside the engine's optimisation profile, which would otherwise lead to
silent stale-state inference.

BF16 binding support:
  numpy has no native bfloat16 dtype, so for BF16-typed TRT bindings we use
  uint16 as the byte-storage proxy. uint16 has the same 2-byte width as bf16;
  TRT reads raw bytes and interprets according to its declared dtype. Callers
  pass bf16 inputs either as np.uint16 buffers (already in bf16 byte layout)
  or as torch.bfloat16 tensors (we view-as-uint16 internally). Outputs come
  back as np.uint16; the caller views as torch.bfloat16 via
      torch.from_numpy(arr).view(torch.bfloat16)
  to recover the semantic dtype.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Union

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

try:  # torch is optional for the engine itself, but bf16 callers need it
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def get_trt_logger(verbosity: str = "INFO") -> trt.Logger:
    level = getattr(trt.Logger, verbosity.upper(), trt.Logger.INFO)
    return trt.Logger(level)


_DEFAULT_LOGGER = get_trt_logger("INFO")


def _check(ret):
    err = ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1]}")
    return ret[1] if len(ret) > 1 else None


def _trt_dtype_to_np(td) -> "np.dtype":
    """Map a TRT data type to the numpy dtype we use for I/O storage. BF16
    has no native numpy dtype so we map it to uint16 (same byte layout)."""
    if td == trt.DataType.BF16:
        return np.dtype(np.uint16)
    return np.dtype(trt.nptype(td))


class TRTEngine:
    """Load + run a TensorRT engine with named feed/output dicts."""

    def __init__(self, path: Path | str, logger: trt.Logger | None = None):
        rt = trt.Runtime(logger or _DEFAULT_LOGGER)
        self.engine = rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.io = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_in = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.io}
        # Per-binding semantic TRT dtype + numpy storage dtype. BF16 storage is uint16.
        self.trt_dtype = {n: self.engine.get_tensor_dtype(n) for n in self.io}
        self.dtype = {n: _trt_dtype_to_np(self.trt_dtype[n]) for n in self.io}
        self.is_bf16 = {n: (self.trt_dtype[n] == trt.DataType.BF16) for n in self.io}

    def __call__(self, feed: Dict[str, Union[np.ndarray, "torch.Tensor"]]) -> Dict[str, np.ndarray]:
        bufs = []
        outs: Dict[str, tuple] = {}
        # Bind inputs
        for n, val in feed.items():
            arr = self._coerce_input(n, val)
            dptr = _check(cudart.cudaMalloc(arr.nbytes))
            _check(cudart.cudaMemcpy(dptr, arr.ctypes.data, arr.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            self.ctx.set_tensor_address(n, int(dptr))
            ok = self.ctx.set_input_shape(n, arr.shape)
            if not ok:
                # TRT 10 returns False (not an exception) for out-of-profile shapes;
                # silently running with stale state produces wrong output. Surface it.
                raise RuntimeError(
                    f"set_input_shape({n}, {arr.shape}) rejected — outside engine optimisation profile"
                )
            bufs.append(dptr)
        # Allocate outputs
        for n in self.io:
            if self.is_in[n]:
                continue
            shape = tuple(self.ctx.get_tensor_shape(n))
            arr = np.empty(shape, dtype=self.dtype[n])  # uint16 for BF16 bindings
            dptr = _check(cudart.cudaMalloc(arr.nbytes))
            self.ctx.set_tensor_address(n, int(dptr))
            outs[n] = (dptr, arr)
            bufs.append(dptr)
        # Execute
        stream = _check(cudart.cudaStreamCreate())
        self.ctx.execute_async_v3(int(stream))
        _check(cudart.cudaStreamSynchronize(stream))
        # Read outputs
        result: Dict[str, np.ndarray] = {}
        for n, (dptr, arr) in outs.items():
            _check(cudart.cudaMemcpy(arr.ctypes.data, dptr, arr.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            result[n] = arr
        for d in bufs:
            cudart.cudaFree(d)
        cudart.cudaStreamDestroy(stream)
        return result

    def _coerce_input(self, name: str, val) -> np.ndarray:
        """Convert a caller-supplied input into the right numpy byte layout for
        this binding. For BF16 bindings:
          - torch.bfloat16 tensor -> view as uint16 (zero-copy bytes)
          - numpy uint16 buffer   -> accept as-is (caller already encoded)
          - any other input       -> error (no implicit conversion; would
                                      silently corrupt by reinterpreting fp16
                                      bytes as bf16 bytes etc.)
        For non-BF16 bindings: cast via numpy as before.
        """
        if self.is_bf16[name]:
            if _HAS_TORCH and torch.is_tensor(val):
                if val.dtype != torch.bfloat16:
                    raise TypeError(
                        f"BF16 binding {name!r} requires torch.bfloat16 tensor "
                        f"or np.uint16 buffer; got torch.{val.dtype}"
                    )
                # .view(torch.uint16) is a zero-copy bit-reinterpretation. We
                # then materialise on CPU for the cudaMemcpy.
                v = val.detach().contiguous().view(torch.uint16).cpu().numpy()
                return np.ascontiguousarray(v)
            if isinstance(val, np.ndarray):
                if val.dtype != np.uint16:
                    raise TypeError(
                        f"BF16 binding {name!r}: pass torch.bfloat16 tensor "
                        f"or np.uint16 buffer; got numpy {val.dtype}"
                    )
                return np.ascontiguousarray(val)
            raise TypeError(
                f"BF16 binding {name!r}: unsupported input type {type(val).__name__}"
            )
        # Non-BF16: regular numpy cast
        if _HAS_TORCH and torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        return np.ascontiguousarray(val.astype(self.dtype[name]))
