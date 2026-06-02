# =============================================================================
# Colab cell: inspect the already-built TRT engines for the zero-logit bug
# =============================================================================
# Paste into a single Colab cell AFTER your existing notebook has loaded the
# model and built/loaded the TRT engines (i.e. after runner = ...).
#
# What it does (read-only — does not modify engines or push code):
#   1. Lists each TRT engine's bindings + dtypes (vit, proj, prefill, decode_mtp, decode_ar)
#   2. Runs each engine on RANDOM input (no model dependency) and reports
#      output stats. If decode_engine outputs are zeros for ALL inputs, the
#      engine itself is structurally broken (vs the past_kv we fed last time
#      being malformed). Definitively isolates input-data bug vs engine bug.
#   3. Runs polygraphy `--show-layer-precisions` against each engine and
#      counts FP16 vs BF16 layer placements. This answers "did TRT actually
#      use BF16 internally or silently demote everything to FP16" without
#      depending on layer-by-layer cos_sim numbers.
#
# Output is printed inline and saved to:
#   /content/locany/trt_engine_inspect.txt
# Download via Files panel or `!cat /content/locany/trt_engine_inspect.txt`.
# =============================================================================

import sys, subprocess, os
from pathlib import Path

# Path your engines live (matches the orchestrator's TRT_DIR)
TRT_DIR = Path("/content/locany/engines")
OUT_LOG = Path("/content/locany/trt_engine_inspect.txt")
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

# Install polygraphy if needed
try:
    import polygraphy  # noqa
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                           "polygraphy>=0.49", "colored"])

import tensorrt as trt
import numpy as np
from cuda.bindings import runtime as cudart


# ------------------------ 1. binding inventory ------------------------------
def _check(ret):
    err = ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1]}")
    return ret[1] if len(ret) > 1 else None

def inspect_bindings(eng_path: Path) -> str:
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(eng_path.read_bytes())
    lines = [f"\n=== {eng_path.name} ==="]
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "IN " if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUT"
        dt = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        lines.append(f"  {mode} {name:24s}  dtype={dt}  shape={tuple(shape)}")
    return "\n".join(lines)


# ------------------------ 2. random-input sanity ---------------------------
def random_input_probe(eng_path: Path) -> str:
    """Run the engine with random fp32 inputs (cast to binding dtype) and
    report output stats. Pure pass-through test — if outputs are all zero
    regardless of input, the engine is structurally broken."""
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(eng_path.read_bytes())
    ctx = engine.create_execution_context()
    lines = [f"\n=== RANDOM PROBE: {eng_path.name} ==="]

    bufs = []
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dt = engine.get_tensor_dtype(name)
        # Use the OPT shape from the profile (idx 0) for runtime shape
        try:
            shape = tuple(engine.get_tensor_profile_shape(name, 0)[1])
        except Exception:
            shape = tuple(engine.get_tensor_shape(name))
        # Storage dtype (numpy)
        if dt == trt.DataType.BF16:
            np_dt = np.uint16
        else:
            np_dt = trt.nptype(dt)
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
        dptr = _check(cudart.cudaMalloc(nbytes))
        ctx.set_tensor_address(name, int(dptr))
        if is_in:
            if dt == trt.DataType.INT64 or dt == trt.DataType.INT32:
                buf = np.random.randint(0, 100, size=shape).astype(np_dt)
            elif dt == trt.DataType.BF16:
                # Random small bf16 values via fp32 -> bf16 via torch (uint16 view)
                import torch
                t = (torch.randn(shape) * 0.1).to(torch.bfloat16)
                buf = t.view(torch.uint16).numpy()
            else:
                buf = (np.random.randn(*shape) * 0.1).astype(np_dt)
            buf = np.ascontiguousarray(buf)
            _check(cudart.cudaMemcpy(dptr, buf.ctypes.data, buf.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            ctx.set_input_shape(name, shape)
        else:
            outs[name] = (dptr, shape, np_dt, dt)
        bufs.append(dptr)

    stream = _check(cudart.cudaStreamCreate())
    ctx.execute_async_v3(int(stream))
    _check(cudart.cudaStreamSynchronize(stream))

    for name, (dptr, shape, np_dt, dt) in outs.items():
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
        out = np.empty(shape, dtype=np_dt)
        _check(cudart.cudaMemcpy(out.ctypes.data, dptr, nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        # Interpret BF16 storage as bf16 via torch view
        if dt == trt.DataType.BF16:
            import torch
            t = torch.from_numpy(out).view(torch.bfloat16).float()
            lines.append(f"  OUT {name:20s}  bf16 shape={shape}  "
                         f"mean={t.mean().item():+.5f}  std={t.std().item():.5f}  "
                         f"min={t.min().item():+.4f}  max={t.max().item():+.4f}  "
                         f"NaN={bool(torch.isnan(t).any())}")
        else:
            lines.append(f"  OUT {name:20s}  {np_dt} shape={shape}  "
                         f"mean={out.mean():+.5f}  std={out.std():.5f}  "
                         f"min={out.min():+.4f}  max={out.max():+.4f}  "
                         f"NaN={bool(np.isnan(out.astype(np.float32)).any())}")
    for d in bufs:
        cudart.cudaFree(d)
    cudart.cudaStreamDestroy(stream)
    return "\n".join(lines)


# ------------------------ 3. polygraphy layer precisions -------------------
def poly_layer_precisions(eng_path: Path) -> str:
    """Use polygraphy to introspect the per-layer precision TRT chose."""
    cmd = ["polygraphy", "inspect", "model", str(eng_path),
            "--show-layer-precisions", "--show", "layers", "attrs", "weights"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        text = out.stdout + ("\n[stderr]\n" + out.stderr if out.stderr else "")
    except Exception as e:
        return f"\n=== polygraphy {eng_path.name}: FAILED: {e!r} ==="
    # Count FP16 / BF16 / FP32 / INT8 layer placements
    fp16 = text.count("FP16")
    bf16 = text.count("BF16")
    fp32 = text.count("FP32")
    int8 = text.count("INT8")
    head = (f"\n=== polygraphy layer-precisions: {eng_path.name} ===\n"
            f"  FP16 mentions: {fp16}\n"
            f"  BF16 mentions: {bf16}\n"
            f"  FP32 mentions: {fp32}\n"
            f"  INT8 mentions: {int8}\n")
    # Truncate full output to 200 lines (it can be huge)
    truncated = "\n".join(text.splitlines()[:200])
    return head + "\n--- first 200 lines ---\n" + truncated


# ------------------------ Run everything -----------------------------------
report = []
for engine_name in ["vision.engine", "projector.engine",
                     "llm_prefill.engine", "llm_decode.engine", "llm_decode_ar.engine"]:
    p = TRT_DIR / engine_name
    if not p.exists():
        report.append(f"\n=== MISSING: {p} ===")
        continue
    report.append(inspect_bindings(p))

print("\n".join(report))
print("\n" + "="*78)
print("RANDOM-INPUT PROBE (does the engine produce nonzero output for ANY input?)")
print("="*78)
random_probes = []
for engine_name in ["llm_prefill.engine", "llm_decode.engine", "llm_decode_ar.engine"]:
    p = TRT_DIR / engine_name
    if p.exists():
        try:
            random_probes.append(random_input_probe(p))
        except Exception as e:
            random_probes.append(f"\n=== RANDOM PROBE FAILED on {engine_name}: {e!r} ===")
print("\n".join(random_probes))

print("\n" + "="*78)
print("POLYGRAPHY LAYER PRECISIONS (was BF16 actually used internally?)")
print("="*78)
poly_reports = []
for engine_name in ["llm_prefill.engine", "llm_decode.engine", "llm_decode_ar.engine"]:
    p = TRT_DIR / engine_name
    if p.exists():
        poly_reports.append(poly_layer_precisions(p))
print("\n".join(poly_reports))

# Save full report
full = "\n".join(report + ["\n" + "="*78 + "\nRANDOM-INPUT PROBE\n" + "="*78]
                  + random_probes
                  + ["\n" + "="*78 + "\nPOLYGRAPHY LAYER PRECISIONS\n" + "="*78]
                  + poly_reports)
OUT_LOG.write_text(full)
print(f"\n[saved] full report -> {OUT_LOG}")

# Print interpretation guide
print("""

================================================================================
INTERPRETATION GUIDE
================================================================================

A. RANDOM PROBE: llm_decode.engine outputs are all-zero / all-NaN
   => engine is structurally broken (TRT compiled to no-op or signal-lost).
   Most likely the bf16 ONNX export contains an op TRT couldn't lower correctly.
   Action: revert export/llm.py decode dummies to fp16 (matches the working
   prefill engine which still produces valid bf16 logits).

B. RANDOM PROBE: outputs are non-zero finite values
   => engine is functional. Last session's all-zero observation came from
   how we fed past_kv (not from the engine itself). Action: inspect the
   prefill output present_kv -> decode input past_kv dtype/shape/stride flow.

C. POLYGRAPHY: BF16 mentions are ~0 across all decode layers
   => TRT silently demoted to FP16 despite our BF16 flag.
   Action: rebuild with NetworkDefinitionCreationFlag.STRONGLY_TYPED in
   trt/build.py::build_engine.

D. POLYGRAPHY: BF16 mentions are high, but RANDOM PROBE shows zeros
   => TRT uses BF16 but something else is wrong (likely an op in the bf16
   path that produces -inf / underflows; visible as NaN in the random probe).
   Action: drop bf16 LLM export entirely and accept the fp16 0.945 cos_sim
   drift while investigating further.
""")
