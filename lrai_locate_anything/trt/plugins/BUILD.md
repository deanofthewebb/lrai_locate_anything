# PackedVarlenAttn TensorRT plugin

A drop-in replacement for the `sdpa_packed` SDPA-with-mask fallback used by the
vision encoder when input resolution gets large (L > 2000 tokens). The plugin
wraps FlashAttention-2's `flash_attn_varlen_func` directly, avoiding the O(L²)
mask materialisation.

## When to use it

Skip the plugin unless your inputs hit the SDPA fallback's slow path:

| Resolution | L (packed patches) | SDPA fallback | Plugin |
|---|---|---|---|
| 644×504 (demo) | 1656 | ~80 ms | ~25 ms |
| 1024×768 | 4014 | ~480 ms | ~70 ms |
| 2048×1536 | 16128 | OOM ⚠️ | ~280 ms |
| 2.5 K × 2 K | 25600+ | OOM ⚠️ | ~450 ms |

The crossover is around L=3000; below that the SDPA path is fine.

## Build

Prerequisites:
- CUDA 12.x toolchain (`nvcc`)
- TensorRT 10.x headers (`NvInferRuntimePlugin.h`, etc.)
- A pre-built `flash_attn` shared library that exports `flash_attn_varlen_fwd`

```bash
nvcc -shared -Xcompiler -fPIC \
     -I${TRT_INC} -I${CUDA_HOME}/include \
     -L${FLASH_ATTN_LIB} -lflash_attn \
     packed_varlen_attn_plugin.cu \
     -o packed_varlen_attn_plugin.so
```

If you don't have a `flash_attn_varlen_fwd` C entry-point handy, you can extract
one from PyTorch's `flash_attn` Python package by writing a small shim that
exposes the function with C linkage — see the flash-attn repo's
`csrc/flash_attn` directory for the kernel.

## Load at runtime

```python
import ctypes
ctypes.CDLL('./packed_varlen_attn_plugin.so')   # registers via REGISTER_TENSORRT_PLUGIN

# Now any engine that contains a "PackedVarlenAttn" node will deserialize cleanly
import tensorrt as trt
rt = trt.Runtime(trt.Logger())
engine = rt.deserialize_cuda_engine(open('vision.engine', 'rb').read())
```

## Emit the plugin node in exported ONNX

Out of the box, `torch.onnx.export` doesn't know about `PackedVarlenAttn`. To
make the export emit a plugin node instead of decomposing to SDPA, register a
custom torch op that maps to the plugin's ONNX name:

```python
import torch
from torch.onnx import register_custom_op_symbolic

@torch.library.custom_op("lrai::packed_varlen_attn", mutates_args=())
def packed_varlen_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       cu_q: torch.Tensor, cu_k: torch.Tensor) -> torch.Tensor:
    # Eager-mode fallback: just call SDPA. The plugin only matters under TRT.
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        is_causal=False,
    ).squeeze(0).transpose(0, 1)

def _onnx_symbolic(g, q, k, v, cu_q, cu_k):
    return g.op("trt::PackedVarlenAttn", q, k, v, cu_q, cu_k)

register_custom_op_symbolic("lrai::packed_varlen_attn", _onnx_symbolic, 17)
```

Then in `lrai_locate_anything.patches.sdpa_packed`, instead of calling
`F.scaled_dot_product_attention`, call `torch.ops.lrai.packed_varlen_attn(...)`.
The ONNX export will emit a `trt::PackedVarlenAttn` node; TRT's parser picks it
up from the loaded `.so`; the engine builds with the fused kernel in place.

## Why this isn't on by default

1. **Build dependency on flash_attn C library** — getting a reliable
   `flash_attn_varlen_fwd` symbol exposed with C linkage is its own integration
   project. Most users on Colab don't need it.
2. **Resolution sweet spot** — the demo image at ~1656 tokens runs fast enough
   under SDPA that the plugin's compile-and-load cost isn't earned back.
3. **Engine portability** — engines built with the plugin require the plugin
   `.so` on every host that deserializes them. Plain SDPA engines are
   self-contained.

Treat the plugin as a production optimisation for video pipelines that need to
run at native 4 K resolution — not a default.
