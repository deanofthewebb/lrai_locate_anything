"""TRTEngine: a thin Python wrapper around tensorrt.IExecutionContext.

Uses cuda-python (cuda.bindings.runtime) for memory + streams. set_input_shape's
return value IS checked — TRT 10 returns False (not an exception) when a shape
falls outside the engine's optimisation profile, which would otherwise lead to
silent stale-state inference.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def get_trt_logger(verbosity: str = "INFO") -> trt.Logger:
    level = getattr(trt.Logger, verbosity.upper(), trt.Logger.INFO)
    return trt.Logger(level)


_DEFAULT_LOGGER = get_trt_logger("INFO")


def _check(ret):
    err = ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1]}")
    return ret[1] if len(ret) > 1 else None


class TRTEngine:
    """Load + run a TensorRT engine with named feed/output dicts."""

    def __init__(self, path: Path | str, logger: trt.Logger | None = None):
        rt = trt.Runtime(logger or _DEFAULT_LOGGER)
        self.engine = rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.io = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.is_in = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.io}
        self.dtype = {n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.io}

    def __call__(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        bufs = []
        outs: Dict[str, tuple] = {}
        # Bind inputs
        for n, arr in feed.items():
            arr = np.ascontiguousarray(arr.astype(self.dtype[n]))
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
            arr = np.empty(shape, dtype=self.dtype[n])
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
