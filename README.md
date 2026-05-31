# lrai_locate_anything

**Modular runtime for NVIDIA LocateAnything-3B** — a Vision-Language Grounding model that emits bounding boxes via Parallel Box Decoding (PBD). This repo refactors the original Colab notebook into a navigable Python package: ONNX export, TensorRT engine build, an orchestrator that mirrors the canonical `generate()`, and image/video inference pipelines.

```
                  pixel_values (1,3,H,W)         input_ids (1,S)
                          │                              │
        ┌─────────────────▼──────────────────┐           │
        │ MoonViT-SO-400M  (vision engine)   │           │
        │   patch=14 · 27L · d=1152 · 16H    │           │
        │   2-D RoPE (real)                  │           │
        │   flash_attn → SDPA + block mask    │           │
        └─────────────────┬──────────────────┘           │
                          │ (L_post, 4608)               │
        ┌─────────────────▼──────────────────┐           │
        │ MLP projector  (LN→Lin→GELU→Lin)   │           │
        └─────────────────┬──────────────────┘           │
                          │ scatter @image_token_index  │
                          ▼                              ▼
        ┌──────────────────────────────────────────────────┐
        │ Qwen2.5-3B-Instruct LLM   (prefill + decode)     │
        │ 36L · d2048 · GQA 16/2 · vocab 152681 · PBD       │
        └──────────────────┬───────────────────────────────┘
                           │ logits (1, K, V)
                           ▼
       ┌──────────────────────────────────────────────┐
       │ PBD orchestrator (Python)                    │
       │   MTP-fast │ MTP+AR hybrid │ pure AR slow     │
       │   parse <box><x1><y1><x2><y2></box>           │
       └──────────────────────────────────────────────┘
```

## Quick start

```bash
git clone https://github.com/deanofthewebb/lrai_locate_anything
cd lrai_locate_anything
pip install -e .
```

```python
from lrai_locate_anything import LocateAnythingRunner

runner = LocateAnythingRunner.from_pretrained(
    model_id='nvidia/LocateAnything-3B',
    workdir='/content/locany',
    auto_export=True,    # exports ONNX + builds TRT engines on first run
)
boxes = runner.detect(
    image='cats.jpg',
    prompt='Detect all cats. Return bounding boxes.',
)
```

```python
# Video inference
from lrai_locate_anything.pipelines import run_video
run_video(runner, 'demo.mp4', 'demo_boxed.mp4',
          prompt='Detect people and luggage.', max_frames=60)
```

## What's in the package

| Module | Responsibility |
|---|---|
| `lrai_locate_anything.config` | Workdir/env detection, model constants |
| `lrai_locate_anything.shims` | transformers ≥ 4.55 compat shims (DynamicCache, `_tied_weights_keys`, `mark_tied_weights_as_initialized`) |
| `lrai_locate_anything.patches` | ONNX-hostile op replacements: `sdpa_packed`, `apply_rope_real`, `Rope2DReal` + the apply-to-live-model helper |
| `lrai_locate_anything.model_loader` | Download from HF, rehydrate config, install shims, load model, apply patches |
| `lrai_locate_anything.parse` | `parse_boxes`, IoU, `python_patch_merger` |
| `lrai_locate_anything.export.vision` | `VisionForExport` (fixed-resolution, baked pos_emb + grid_hws) |
| `lrai_locate_anything.export.projector` | `ProjectorForExport` |
| `lrai_locate_anything.export.llm` | `LLMPrefill` / `LLMDecode` (canonical `input_ids + visual_features` contract) + `export_with_external_data` |
| `lrai_locate_anything.export.int4` | Optional INT4 AWQ via NVIDIA modelopt |
| `lrai_locate_anything.trt.engine` | `TRTEngine` wrapper around `tensorrt.IExecutionContext` |
| `lrai_locate_anything.trt.build` | `build_engine` + per-graph optimization profiles |
| `lrai_locate_anything.orchestrator` | `LocateAnythingRunner` — the public API; mirrors the canonical `generate()` with MTP↔AR hybrid |
| `lrai_locate_anything.pipelines` | `run_image`, `run_video`, side-by-side multi-runtime comparison |

## Architecture invariants (what's load-bearing)

These are documented inline in the source but are worth knowing up-front because they crashed the build many times during the original notebook iteration:

1. **`Qwen2Model.forward` rejects both `input_ids` and `inputs_embeds`** (line 1200 of vendored `modeling_qwen2.py`). The LM wrappers pass `input_ids + visual_features + image_token_index`; vision features get scattered internally via `image_processing()`.
2. **The vision engine is fixed-resolution** by design (pos_emb + grid_hws baked at construction). Video frames must be resized to `(ENG_IMG_W, ENG_IMG_H) = (grid_w*14, grid_h*14)` before the processor sees them, AND `processor.image_processor.min_pixels = max_pixels = ENG_IMG_W * ENG_IMG_H` must be set so the processor's smart_resize doesn't undo the resize.
3. **`flash_attn_varlen_func` returns `(L, H, D)`** — the canonical caller does `.flatten(start_dim=-2)`. Our `sdpa_packed` replacement bakes that flatten into the return value.
4. **2-D RoPE `apply_rope`** unsqueezes `freqs_cis` at dim −2 so `(L, head_dim/2)` broadcasts against `(L, num_heads, head_dim/2)`. The real-valued replacement does the same.
5. **TRT optimization profile keys must match ONNX `input_names` exactly.** `build_serialized_network` returns `None` silently on mismatch with no log entry at any verbosity level.
6. **modelopt INT4 ONNX uses INT8-packed INT4 layout**, which TRT 10's `DequantizeLinear` rejects (`kBLOCKED requires kINT4`). Use `modelopt.onnx.quantization.quantize_static` or TRT-LLM for real INT4 deployment.

## Installation

The pip install fetches Python-side deps. NVIDIA stack (TensorRT, cuda-python, optionally modelopt, optionally tensorrt-llm) is installed on first import via `lrai_locate_anything.config.ensure_nvidia_stack()` — torch-version-aware so the right wheels are pulled.

GPU requirements:
- **≥ 22 GB VRAM** (L4, A100, H100): full FP16 LLM TRT engine builds cleanly
- **16 GB VRAM** (T4, V100-16): vision + projector TRT engines build; LLM stays on PyTorch (orchestrator's fallback path)
- **CPU-only**: not supported

## Original notebook

The notebook that produced this package is preserved at [`docs/LocateAnything_3B_TensorRT.ipynb`](docs/LocateAnything_3B_TensorRT.ipynb) for reference. Every iteration log + audit trail lives in the git history of this repo.

## License

MIT for this wrapper. The underlying model is `nvidia/LocateAnything-3B` (NVIDIA license — non-commercial); MoonViT-SO-400M (MIT); Qwen2.5-3B-Instruct (Qwen Research License).
