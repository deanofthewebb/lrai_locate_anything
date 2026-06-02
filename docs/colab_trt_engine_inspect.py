# =============================================================================
# Colab cell: inspect the already-built TRT engines for the zero-logit bug
# =============================================================================
# Paste into a single Colab cell AFTER your existing notebook has loaded the
# model and built/loaded the TRT engines (i.e. after runner = ...).
#
# What it does (read-only — does not modify engines or push code):
#   1. Lists each TRT engine's I/O bindings + declared dtypes (vit, proj,
#      prefill, decode_mtp, decode_ar)
#   2. Runs each engine on RANDOM input (no model dependency) and reports
#      output stats. If decode_engine outputs are zeros for ALL inputs, the
#      engine itself is structurally broken (vs the past_kv we fed last time
#      being malformed). Definitively isolates input-data bug vs engine bug.
#   3. Uses trt.IEngineInspector (built into TensorRT 10.x — no polygraphy
#      version dependency) to enumerate the actual per-layer precision TRT
#      chose. Counts FP16/BF16/FP32 layer placements. Answers "did TRT
#      actually use BF16 internally or silently demote to FP16."
#
# Output is printed inline and saved to:
#   /content/locany/trt_engine_inspect.txt
# Download via Files panel or `!cat /content/locany/trt_engine_inspect.txt`.
# =============================================================================

import sys, json, os
from pathlib import Path

TRT_DIR = Path("/content/locany/engines")
OUT_LOG = Path("/content/locany/trt_engine_inspect.txt")
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

import tensorrt as trt
import numpy as np
from cuda.bindings import runtime as cudart


# ------------------------ helpers ------------------------------
def _check(ret):
    err = ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1]}")
    return ret[1] if len(ret) > 1 else None


def _np_dtype_for(td):
    if td == trt.DataType.BF16:
        return np.uint16
    return trt.nptype(td)


def _resolve_shape(engine, ctx, name, profile_idx=0):
    """Return a CONCRETE shape for a binding. Inputs: use opt-profile shape.
    Outputs: must be queried AFTER ctx has all input shapes bound (TRT 10
    resolves output dims from inputs in the execution context).
    """
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        return tuple(engine.get_tensor_profile_shape(name, profile_idx)[1])
    # output — read from the execution context, which now has concrete dims
    shape = tuple(ctx.get_tensor_shape(name))
    # Belt-and-suspenders: any remaining -1 means an input wasn't bound. Bail loudly.
    if any(d < 0 for d in shape):
        raise RuntimeError(
            f"output {name!r} shape is still dynamic {shape} — bind all inputs first"
        )
    return shape


# ------------------------ 1. binding inventory ------------------------------
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


# ------------------------ 2. random-input probe -----------------------------
def random_input_probe(eng_path: Path) -> str:
    """Run the engine with random inputs (cast to binding dtype) and report
    output stats. Pure pass-through test — if outputs are all zero regardless
    of input, the engine is structurally broken."""
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(eng_path.read_bytes())
    ctx = engine.create_execution_context()
    lines = [f"\n=== RANDOM PROBE: {eng_path.name} ==="]

    bufs = []
    outs = {}

    # PASS 1: bind every INPUT first so the context can resolve dynamic outputs.
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        dt = engine.get_tensor_dtype(name)
        shape = _resolve_shape(engine, ctx, name)
        np_dt = _np_dtype_for(dt)
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
        if nbytes <= 0:
            raise RuntimeError(f"input {name!r} resolved to non-positive nbytes={nbytes} shape={shape}")
        dptr = _check(cudart.cudaMalloc(nbytes))
        ctx.set_tensor_address(name, int(dptr))
        # Fill with random values appropriate to the dtype
        if dt == trt.DataType.INT64 or dt == trt.DataType.INT32:
            # attention_mask wants 0/1; position_ids want 0..S-1; input_ids want
            # 0..vocab. Random ints in [0, 100) work for all three.
            buf = np.random.randint(0, 100, size=shape).astype(np_dt)
        elif dt == trt.DataType.BF16:
            import torch
            t = (torch.randn(shape) * 0.1).to(torch.bfloat16)
            buf = t.contiguous().view(torch.uint16).numpy()
        else:
            buf = (np.random.randn(*shape) * 0.1).astype(np_dt)
        buf = np.ascontiguousarray(buf)
        _check(cudart.cudaMemcpy(dptr, buf.ctypes.data, buf.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        ok = ctx.set_input_shape(name, shape)
        if not ok:
            raise RuntimeError(f"set_input_shape({name}, {shape}) rejected")
        bufs.append(dptr)

    # PASS 2: outputs now have concrete shapes via ctx.get_tensor_shape.
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            continue
        dt = engine.get_tensor_dtype(name)
        shape = _resolve_shape(engine, ctx, name)
        np_dt = _np_dtype_for(dt)
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
        dptr = _check(cudart.cudaMalloc(nbytes))
        ctx.set_tensor_address(name, int(dptr))
        outs[name] = (dptr, shape, np_dt, dt)
        bufs.append(dptr)

    # Execute
    stream = _check(cudart.cudaStreamCreate())
    ctx.execute_async_v3(int(stream))
    _check(cudart.cudaStreamSynchronize(stream))

    # Read outputs (limit reporting to the first 4 outputs + logits, to keep
    # the report scannable on a 36-layer KV cache).
    interesting = []
    for name in outs:
        if name == "logits" or len(interesting) < 4:
            interesting.append(name)
    for name in interesting:
        dptr, shape, np_dt, dt = outs[name]
        nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
        out = np.empty(shape, dtype=np_dt)
        _check(cudart.cudaMemcpy(out.ctypes.data, dptr, nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        if dt == trt.DataType.BF16:
            import torch
            t = torch.from_numpy(out).view(torch.bfloat16).float()
            lines.append(
                f"  OUT {name:20s}  bf16 shape={shape}  "
                f"mean={t.mean().item():+.5f}  std={t.std().item():.5f}  "
                f"min={t.min().item():+.4f}  max={t.max().item():+.4f}  "
                f"NaN={bool(torch.isnan(t).any())}"
            )
        else:
            o32 = out.astype(np.float32)
            lines.append(
                f"  OUT {name:20s}  {np.dtype(np_dt).name} shape={shape}  "
                f"mean={o32.mean():+.5f}  std={o32.std():.5f}  "
                f"min={o32.min():+.4f}  max={o32.max():+.4f}  "
                f"NaN={bool(np.isnan(o32).any())}"
            )
    if len(outs) > len(interesting):
        lines.append(f"  ... +{len(outs)-len(interesting)} more outputs not shown")

    for d in bufs:
        cudart.cudaFree(d)
    cudart.cudaStreamDestroy(stream)
    return "\n".join(lines)


# ------------------------ 3. EngineInspector layer precisions ---------------
def engine_inspector_precisions(eng_path: Path) -> str:
    """Use trt.IEngineInspector (built into TRT 10.x) to enumerate per-layer
    precision. No polygraphy version dependency."""
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(eng_path.read_bytes())
    ctx = engine.create_execution_context()  # required by inspector

    inspector = engine.create_engine_inspector()
    inspector.execution_context = ctx
    fmt = trt.LayerInformationFormat.JSON

    # Get aggregate engine info (contains all layers)
    raw = inspector.get_engine_information(fmt)
    try:
        info = json.loads(raw)
    except Exception:
        return (f"\n=== EngineInspector: {eng_path.name} ===\n"
                f"  failed to parse JSON (head 200 chars): {raw[:200]!r}")

    layers = info.get("Layers", [])
    prec_counts = {"FP32": 0, "FP16": 0, "BF16": 0, "INT8": 0, "INT32": 0,
                   "INT64": 0, "BOOL": 0, "OTHER": 0}

    # Helper: extract per-tensor dtypes from a layer's "Inputs"/"Outputs" lists.
    def _stamp_layer(layer):
        # Some TRT versions expose a layer-level "Precision" / "ComputePrecision"
        # field directly; check those first.
        for key in ("ComputePrecision", "Precision", "LayerPrecision"):
            p = layer.get(key)
            if p:
                p = str(p).upper()
                for known in prec_counts:
                    if known in p:
                        prec_counts[known] += 1
                        return
                prec_counts["OTHER"] += 1
                return
        # Fallback: tally output tensor dtypes
        outs = layer.get("Outputs", []) or layer.get("OutputTensors", [])
        for t in outs:
            d = str(t.get("Type", "") or t.get("DataType", "")).upper()
            for known in prec_counts:
                if known in d:
                    prec_counts[known] += 1
                    return
        prec_counts["OTHER"] += 1

    for layer in layers:
        _stamp_layer(layer)

    lines = [f"\n=== EngineInspector: {eng_path.name} ===",
             f"  total layers: {len(layers)}"]
    for k, v in prec_counts.items():
        if v:
            lines.append(f"    {k:6s}: {v}")
    # Dump first 5 layer JSON entries verbatim for spot-checking
    if layers:
        lines.append("  --- first 3 layers (raw JSON) ---")
        for L in layers[:3]:
            lines.append("  " + json.dumps(L)[:240])
    return "\n".join(lines)


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
            random_probes.append(f"\n=== RANDOM PROBE FAILED on {engine_name}: {type(e).__name__}: {e} ===")
print("\n".join(random_probes))

print("\n" + "="*78)
print("ENGINE INSPECTOR (was BF16 actually used internally?)")
print("="*78)
inspector_reports = []
for engine_name in ["llm_prefill.engine", "llm_decode.engine", "llm_decode_ar.engine"]:
    p = TRT_DIR / engine_name
    if p.exists():
        try:
            inspector_reports.append(engine_inspector_precisions(p))
        except Exception as e:
            inspector_reports.append(f"\n=== INSPECTOR FAILED on {engine_name}: {type(e).__name__}: {e} ===")
print("\n".join(inspector_reports))

full = "\n".join(report
                  + ["\n" + "="*78 + "\nRANDOM-INPUT PROBE\n" + "="*78]
                  + random_probes
                  + ["\n" + "="*78 + "\nENGINE INSPECTOR LAYER PRECISIONS\n" + "="*78]
                  + inspector_reports)
OUT_LOG.write_text(full)
print(f"\n[saved] full report -> {OUT_LOG}")

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

C. ENGINE INSPECTOR: BF16 count ~0 across all decode layers
   => TRT silently demoted to FP16 despite our BF16 flag.
   Action: rebuild with NetworkDefinitionCreationFlag.STRONGLY_TYPED in
   trt/build.py::build_engine (already shipped in commit 7fd1449).

D. ENGINE INSPECTOR: BF16 count high, but RANDOM PROBE shows zeros
   => TRT uses BF16 but something else is wrong (likely an op in the bf16
   path that produces -inf / underflows; visible as NaN in the random probe).
   Action: drop bf16 LLM export entirely and accept the fp16 0.945 cos_sim
   drift while investigating further.

E. ENGINE INSPECTOR: only "OTHER" populated, all known precisions = 0
   => the TRT version's JSON schema for engine_inspector uses field names
   this script doesn't recognise. Fall back to reading the first-3-layers raw
   JSON dump above and manually inspecting the "Precision"/"Type" key.
""")
