# TRT-LLM Adoption Design — LocateAnything-3B

**Status:** Draft v1
**Owner:** dean.webb@livereach.ai
**Date:** 2026-06-03
**Predecessor:** `single_file_trt_design.md` (single-engine IIfConditional approach, failed against TRT 10.16 myelin/codeGenerator.cpp:3811)

---

## Executive Summary

We are migrating the LocateAnything-3B autoregressive LLM body off our hand-rolled TensorRT `IIfConditional` prefill/decode unification (which is structurally blocked by TRT 10 — see `single_file_trt_design.md` post-mortem) and onto **NVIDIA TensorRT-LLM v1.2.x**, which solves the same problem at the *runtime + paged-KV* layer instead of in the engine graph. The deliverable shrinks our 5-engine Phase-1 pipeline to a clean 2-engine artifact set: `vision_proj.engine` (~2 GB, unchanged from Phase 1) and `llm.engine` (~6 GB, built by `trtllm-build`), totaling ~8 GB which matches the original Phase-2 target. We get prefill+decode unification, paged KV cache, in-flight batching, and a 1.5–3x latency speedup on L4/3080 Ti; we lose our custom MTP head (deferred behind TRT-LLM Medusa) and we accept a forced TRT downgrade (10.16 → 10.9/10.11) inside an isolated venv. Timeline: 7 phases, ~17–22 engineer-days of careful, gated work.

---

## Why TRT-LLM (vs our DIY IIfConditional that just failed)

### The structural TRT 10 block
Our Phase-1 attempt to fuse prefill and decode into a single TensorRT engine via `IIfConditional` died inside `myelin/codeGenerator.cpp:3811` with a data-dependent-shape error. This is **not a bug we can route around** — it is a documented TRT 10 limitation: the myelin code generator cannot resolve dynamic shapes whose extent is conditioned on `IIfConditional` branch selection. Specifically, the KV-cache `seq_len` axis transitions from `prompt_len` (prefill, then-branch) to `prompt_len + step` (decode, else-branch), and the codegen cannot fuse those into a single shape tensor.

We have three full days of evidence in `learn02_polygraphy_llm_decode.txt`, `learn02_polygraphy_llm_prefill.txt`, and `learn02_build_strongly_typed.log` confirming the failure reproduces deterministically across:
- bf16 vs fp16
- explicit-shape vs strongly-typed networks
- with vs without `gpt_attention_plugin`

### How TRT-LLM sidesteps it
TRT-LLM's prefill+decode unification is a **runtime concern, not a graph concern**:

1. The engine itself is built as a single forward pass with a *fused attention plugin* (`gpt_attention_plugin auto`) that exposes a unified `[B, S, H]` -> `[B, S, V]` interface.
2. The KV cache is **paged** — managed by a host-side block manager (`tensorrt_llm/runtime/kv_cache_manager.py`) — and chunks are bound to TRT input tensors at runtime.
3. The prefill/decode dispatch happens in `ModelRunner.generate()` Python, which calls the same engine twice with different effective `S` and different paged-KV block lists.

There is **no `IIfConditional` in the TRT-LLM engine graph**. The codegen-level blocker disappears because the conditional logic was never expressed at the engine level to begin with. This is the industry pattern (vLLM, TGI, SGLang all use this same architectural split).

### Secondary wins
- **Paged KV** removes the contiguous-`max_seq_len` allocation that wastes ~80% of VRAM on short prompts.
- **In-flight batching** is supported out of the box (relevant if we ever serve multiple cameras concurrently).
- **Speculative decoding (Medusa, Eagle, MTP)** is a first-class TRT-LLM feature — we can revisit MTP later as a TRT-LLM speculative head instead of a hand-rolled engine.
- **Logits post-processors** let us inject box-token grammar constraints at the kernel level (future optimization).

---

## Final Artifact Set

Canonical TRT-LLM multimodal layout — **three engines, ALL TRT-LLM 10.9** (no separate `vision_proj.engine`, no TRT 10.16 dependency):

```
lrai_locate_anything_trtllm/
├── vision/
│   └── vision_encoder.engine     ~2.0 GB   TRT-LLM 10.9, MoonViT ported via build_multimodal_engine.py
└── llm/
    ├── llm.bf16.engine           ~6.0 GB   TRT-LLM 10.9, Qwen2.5-3B bf16, via trtllm-build
    ├── llm.int4.engine           ~2.0 GB   TRT-LLM 10.9, Qwen2.5-3B int4 weight-only,
    │                                       via trtllm-build --use_weight_only --weight_only_precision int4
    ├── config.json               ~ 8 KB    TRT-LLM engine config (paged_kv, dtype, plugins)
    └── rank0.engine -> llm.<dtype>.engine  symlink chosen at deploy time
```

Optional: a separate `visual_features` projection engine if the ported MoonViT template doesn't expose a built-in projector to LLM hidden dim (decided in D.1/D.2).

**Total: ~10 GB combined across both quantization variants** (the bf16 and int4 LLM engines coexist on disk; only one is loaded per process). Single TRT-LLM 10.9 runtime end-to-end.

Distribution: all files uploaded to `s3://data-labeling.livereachmedia.com/datasets/safetunnel/models/locany_trtllm/` (mirroring the SAM3 pattern — do NOT use `s3://trackerbot/`, AccessDenied).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Host Process                                    │
│                                                                              │
│   image (PIL.Image)                                                          │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────┐                                                   │
│   │ preprocess           │  448x448, normalize, dtype=bf16                   │
│   │ (CPU, NumPy)         │                                                   │
│   └──────────┬───────────┘                                                   │
│              │                                                               │
│              ▼   pixel_values [1, 3, 448, 448]                               │
│   ┌──────────────────────┐                                                   │
│   │ vision_proj.engine   │  TRT 10.16 runtime                                │
│   │ (MoonViT + projector)│  GPU resident, ~2 GB                              │
│   └──────────┬───────────┘                                                   │
│              │ visual_features [1, N=256, H=2048]                            │
│              ▼                                                               │
│   ┌──────────────────────┐                                                   │
│   │ moonvit_adapter      │  Flatten -> prompt_embedding_table                │
│   │ (Python, ~5 lines)   │  shape [N, H] with table_id starting at vocab_max │
│   └──────────┬───────────┘                                                   │
│              │ prompt_embedding_table [256, 2048]                            │
│              │ + input_ids (text + table_token placeholders)                 │
│              ▼                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ LocateAnythingTRTLLMRunner                                           │  │
│   │  └─ tensorrt_llm.runtime.ModelRunner (or ModelRunnerCpp)             │  │
│   │       ├─ prefill: 1 engine call, S=prompt_len                        │  │
│   │       ├─ decode:  K engine calls, S=1, paged KV reused               │  │
│   │       └─ stop on </box> or max_new_tokens=128                        │  │
│   │      llm.engine, GPU resident, ~6 GB + paged KV (~0.5 GB working)    │  │
│   └──────────┬───────────────────────────────────────────────────────────┘  │
│              │ token_ids [1, output_len]                                     │
│              ▼                                                               │
│   ┌──────────────────────┐                                                   │
│   │ parse_boxes_with_    │  Existing host parser (Phase 1, unchanged)        │
│   │ labels()             │  Regex + tokenizer.decode                         │
│   └──────────┬───────────┘                                                   │
│              │                                                               │
│              ▼                                                               │
│   List[Box] = [{cls, x0, y0, x1, y1, label_text}, ...]                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Engine-to-engine contract
- `vision_proj.engine` output: `[1, 256, 2048]` bf16 (already projected into LLM hidden dim — projection is baked into Phase-1 fusion).
- `llm.engine` input: `prompt_embedding_table` of shape `[N_total, 2048]` where `N_total = num_visual_tokens` (256), bound at runtime via TRT-LLM `runtime_input_dict`.
- `max_multimodal_len` (build flag) = 256, matching `num_visual_features`. This is the **same parameter** as `max_prompt_embedding_table_size` — see TRT-LLM issue #2104.

---

## Phase Plan (careful design, extensive testing — per user directive)

### Phase A — Environment + dependency setup (1 day, **small**, risk: medium)

**Deliverable:** A working `tensorrt_llm` Python import on learn02 in an isolated venv.

Concrete tasks:
1. Create `~/venvs/trtllm/` venv on learn02 (Python 3.10 — required, NOT 3.11, by TRT-LLM v1.2 support matrix).
2. Install `tensorrt_llm==1.2.1` via pip — accept that it will pin `tensorrt==10.9.x`. **Do NOT touch the system TRT 10.16** which our Phase-1 `vision_proj.engine` and YOLOv12 engines depend on.
3. Verify with `python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"` and `python -c "from tensorrt_llm.runtime import ModelRunner; print(ModelRunner)"`.
4. Document the exact pip freeze in `trtllm_prod/requirements.lock.txt`.
5. Colab L4 sibling check: confirm `pip install tensorrt_llm==1.2.1` resolves cleanly in a fresh Colab; capture timing.

**Gate:** `python -m tensorrt_llm.tools.helper` returns version info on both learn02 and Colab L4.

**Risk:** The TRT 10.9 downgrade inside the venv must not bleed into the system. Mitigate with `--prefix` and explicit `LD_LIBRARY_PATH` scoping in our launcher.

---

### Phase B — HF checkpoint → TRT-LLM checkpoint convert (2–3 days, **medium**, risk: medium)

**Deliverable:** `trtllm_prod/convert.py` produces a TRT-LLM checkpoint directory from `nvidia/LocateAnything-3B`.

Concrete tasks:
1. Load `nvidia/LocateAnything-3B` HF checkpoint, isolate the LM body (`model.language_model.*` — Qwen2.5-3B with extended vocab 152681).
2. Save the LM body to a standalone HF dir (`/tmp/locany_lm_only/`) — this is required because the upstream `convert_checkpoint.py` expects a standalone HF model dir.
3. Invoke (subprocess) `python <trtllm>/examples/models/core/qwen/convert_checkpoint.py --model_dir /tmp/locany_lm_only --output_dir /tmp/locany_tllm_ckpt --dtype bfloat16 --tp_size 1`.
4. Verify the output: `config.json` has `vocab_size=152681`, `hidden_size=2048` (Qwen2.5-3B), `num_hidden_layers=36`; per-rank `rank0.safetensors` exists; total size matches PT model size (~6 GB bf16).
5. Sanity load: instantiate `tensorrt_llm.models.QWenForCausalLM.from_checkpoint(tllm_ckpt)` and verify no missing keys.

**Gate:** Round-trip a single weight tensor (e.g. `lm_head.weight`) — load HF, load TRT-LLM ckpt, `torch.testing.assert_close(atol=0, rtol=0)`. Must be bit-exact (we only re-saved, no quantization yet).

**Risk:** Our extended vocab (152681 vs stock Qwen2.5 152064) — see *Open Decisions §1*. The recipe reads vocab from `AutoConfig.from_pretrained()`, so this *should* propagate, but `lm_head.weight` shape must be `[152681, 2048]` post-convert.

---

### Phase C — `trtllm-build` for the LLM (2 days, **medium**, risk: medium)

**Deliverable:** `llm.engine` (~6 GB) built from the Phase-B checkpoint.

Concrete tasks:
1. Author `trtllm_prod/build.py` — thin wrapper around `trtllm-build` CLI.
2. Build flags:
   ```
   trtllm-build \
     --checkpoint_dir /tmp/locany_tllm_ckpt \
     --output_dir <out>/llm \
     --gemm_plugin auto \
     --gpt_attention_plugin auto \
     --max_input_len 4096 \
     --max_seq_len 4224 \
     --max_batch_size 1 \
     --max_num_tokens 4224 \
     --max_multimodal_len 256 \
     --use_paged_context_fmha enable \
     --kv_cache_type paged
   ```
3. Build on learn02 GPU 1 (3080 Ti, 11.9 GB free) — set `CUDA_VISIBLE_DEVICES=1`. If VRAM-tight, fall back to GPU 0 + offloading.
4. Verify: engine size 5.8–6.2 GB; `trtllm-build` log shows 36 transformer layers profiled; `config.json` paged_kv enabled; `rank0.engine` present.
5. Smoke test: `tensorrt_llm.runtime.ModelRunner.from_dir(engine_dir).generate(input_ids=[[1,2,3]], max_new_tokens=4)` returns 4 tokens.

**Gate:** Engine builds without OOM, loads, and generates ≥1 token without error.

**Risk:** GPU 1's 11.9 GB free may be marginal for a 6 GB engine build (peak build-time VRAM is ~2x final engine size). Fall back to Colab L4 (24 GB) if learn02 OOMs.

---

### Phase D — MoonViT port to TRT-LLM `build_multimodal_engine.py` (10–15 days, **xlarge**, risk: high)

**Deliverable:** `vision_encoder.engine` (~2 GB) built natively via TRT-LLM tooling — single TRT 10.9 runtime, no dual-runtime split.

**Decision (final, 2026-06-03):** PORT MoonViT into TRT-LLM's `build_multimodal_engine.py` pattern. Do NOT keep the dual-runtime `vision_proj.engine` (TRT 10.16) + adapter.

Rationale:
- A single TRT 10.9 runtime end-to-end eliminates R5 (cross-version deserialization risk) and the launcher's `LD_LIBRARY_PATH` juggling.
- Long-term maintenance is cleaner: one toolchain, one engine-build script, one parity surface per quantization mode.
- Cost is upfront: we have to translate MoonViT's custom ops (Rope2DReal, sdpa_packed without `cu_seqlens`, grid_hws baking, patch_merger static reshape) into a TRT-LLM-native template.

Concrete sub-deliverables:

- **D.1 — Template audit.** Read TRT-LLM's vision encoder templates: CLIP, SigLIP, Qwen2-VL ViT, InternVL ViT. Pick the closest base (likely InternVL or Qwen2-VL — both are MoonViT-adjacent). Output: short comparison note + selected base in `trtllm_prod/vision/BASE.md`.

- **D.2 — Port MoonViT's custom patches into the chosen template.** Translate:
  - `Rope2DReal` position embeddings (swap from `Rope2DPosEmb`)
  - `sdpa_packed` without `cu_seqlens` (TRT-LLM defaults to cu_seqlens; need an explicit non-cu_seqlens path or weight-bake)
  - `grid_hws` baking (constant tensor, our canonical `(36, 46)`)
  - `patch_merger` static reshape (the Phase 1 work — `[N, H]` → `[N/4, 4*H]` → `[N/4, H_out]`)

- **D.3 — Weight-name compatibility.** TRT-LLM expects specific naming for vision encoders (e.g. `vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.weight`). Our MoonViT uses different conventions. Build a name-mapping table; verify every weight loads with no missing/unexpected keys.

- **D.4 — Convert + build.** Run TRT-LLM's tooling:
  ```
  python <trtllm>/examples/models/core/multimodal/build_multimodal_engine.py \
    --model_type <chosen_base> \
    --model_path /tmp/moonvit_hf \
    --output_dir <out>/vision_encoder \
    --max_batch_size 1 \
    --dtype bfloat16
  ```
  Output: `vision_encoder.engine` (~2 GB), `config.json`.

- **D.5 — Parity test (PT vs TRT-LLM vision encoder).** Same image → MoonViT (PT) vs same image → `vision_encoder.engine`. `torch.testing.assert_close(atol=1e-2, rtol=1e-2)` on bf16 features. Build fixture: `tests/fixtures/lane_axis_frame_001.jpg` + golden `visual_features_pt.npy`. Cos-sim ≥ 0.995 required.

**Gate:** D.5 cos-sim ≥ 0.995 AND all 256×2048 features match within atol=1e-2.

**Effort:** 10–15 days (vs 3–4 days for the adapter path). Net project estimate: **24–33 days** (was 17–22).

**Risk:** Custom-op porting (R3.1) is the dominant unknown. If Rope2DReal or sdpa_packed don't translate cleanly to TRT-LLM's vision encoder template, we may need custom plugins or weight pre-baking. See Risk R3 / R3.1.

---

### Phase E — Runner: `ModelRunner` integration (3–4 days, **medium-large**, risk: medium)

**Deliverable:** `trtllm_prod/runner.py` exposes `LocateAnythingTRTLLMRunner.run(image: PIL.Image, prompt: str) -> List[Box]`.

Concrete tasks:
1. `LocateAnythingTRTLLMRunner.__init__(vision_engine_path, llm_engine_dir, tokenizer_dir)`:
   - Load `vision_proj.engine` via TRT runtime.
   - Load `llm.engine` via `tensorrt_llm.runtime.ModelRunner.from_dir(llm_engine_dir)`.
   - Load HF tokenizer (Qwen2.5 + LocateAnything's added box-special tokens).
2. `.run(image, prompt)`:
   - Preprocess image (existing Phase-1 code).
   - Run vision_proj → `visual_features [1, 256, 2048]`.
   - Build prompt: text tokens + 256 placeholder slots (`<vision_pad>` token repeated 256x, or whatever LocateAnything's chat template uses).
   - Call `model_runner.generate(input_ids=..., prompt_table=visual_features.reshape(256, 2048), max_new_tokens=128, temperature=0.0, end_id=tokenizer.eos_token_id)`.
   - Decode tokens → call existing `parse_boxes_with_labels()`.
3. Wire stop criteria: `</box>` token + `max_new_tokens=128` (sufficient for ~30 boxes per LocateAnything format).
4. Single-image inference path only — batched is out of scope for v1.

**Gate:** `runner.run(fixture_image, fixture_prompt)` returns a non-empty `List[Box]` without exception.

---

### Phase F — Parity testing (EXTENSIVE) (4–5 days, **large**, risk: medium)

**Deliverable:** `tests/trtllm/test_parity.py` + `tests/trtllm/test_audit_replay.py` + `bench/trtllm_vs_phase1.py`.

This is the canonical extensive-testing phase, per user directive (and per `feedback_speculation_discipline.md` — verify by replay before claiming parity).

Concrete tasks:

**F.1 PT parity (logits-level):**
- 10 fixture images, 5 prompts each.
- For each (image, prompt) pair, generate via:
  - PT path: `transformers.AutoModelForCausalLM` HF generate, greedy, `temperature=0`.
  - TRT-LLM path: `LocateAnythingTRTLLMRunner.run`, greedy, `temperature=0`.
- Assert **token-level greedy match ≥ 95%** (TRT-LLM is deterministic at T=0; the 5% slack accommodates the documented TRT-LLM tolerance `atol=0.4 rtol=0.4` from `test_modeling_qwen.py`).
- Assert **cos_sim of final-token logits ≥ 0.99**.
- Report failures with token-by-token diff.

**F.2 Audit-clip replay:**
- Re-run audit clips A5_F1 through A5_F15 through the new TRT-LLM path.
- Compare per-class IN/OUT counts (`vehicle`, `person`, `forklift`) to current 5-engine TRT path.
- Tolerance: **±5% per class per clip**. Anything wider blocks merge.
- Inspect 3 random sample frames per clip visually (overlay PNG diff).

**F.3 Performance benchmark:**
- Single-image latency: 100 trials, report p50 / p90 / p99.
- Expectation: **1.5–3x speedup** vs current 5-engine path (paged KV + plugin fusion).
- VRAM peak during inference: must fit in 10 GB (3080 GPU 0 budget).

**F.4 Edge cases:**
- Empty detection: image with no objects → output `<box>None</box>` (or LocateAnything's no-detection sentinel) — verify TRT-LLM stop logic handles it.
- Max-length output: synthetic prompt that triggers 128-token output — verify we hit `max_new_tokens` cleanly, no truncation mid-`<box>`.
- Low-prompt-token-count: 5-token prompt — verify the paged-KV block manager doesn't crash on undersized first block.
- Vocab-edge tokens: prompts containing LocateAnything's added tokens (ids 152064..152680) — verify they tokenize and decode round-trip.

**Gates:**
1. F.1: token-match ≥ 95% on 50/50 pairs.
2. F.2: per-class counts within ±5% on 15/15 clips.
3. F.3: p50 latency strictly less than current Phase-1 p50.
4. F.4: all 4 edge cases pass.

**Regression guard:** No merge if mAP on the audit clip drops below **0.95× the current PT path**. (Per *feedback_no_fallbacks_or_gates.md* — fix the root cause if it does, don't gate around it.)

---

### Phase G — Colab packaging + L4 benchmark (2–3 days, **small-medium**, risk: low)

**Deliverable:** `lrai_locate_anything_trtllm_demo.ipynb` runnable on a free Colab L4.

Concrete tasks:
1. Notebook cell 1: `pip install tensorrt_llm==1.2.1` (Colab L4 has driver 555+, sm_89, satisfies TRT-LLM matrix).
2. Notebook cell 2: download `vision_proj.engine` + `llm.engine` from S3 (`s3://data-labeling.livereachmedia.com/.../locany_trtllm/`).
3. Notebook cell 3: `from trtllm_prod.runner import LocateAnythingTRTLLMRunner; runner = LocateAnythingTRTLLMRunner(...); boxes = runner.run(img, prompt)`.
4. Notebook cell 4: render overlay PNG.
5. L4 benchmark: 50 images, report p50/p90 latency. Target: **< 5 seconds** end-to-end per image at bf16.

**Gate:** Notebook executes top-to-bottom on a clean Colab L4 in under 10 minutes total (including engine download).

---

## Test Plan

Each phase has its own gate (listed in-phase above). The integrated test plan:

1. **Unit tests (per phase):** Phase B (round-trip weight check), Phase C (engine smoke test), Phase D (vision parity).
2. **Integration test (Phase E):** Single-image end-to-end run returns non-empty boxes.
3. **Parity tests (Phase F):**
   - F.1 logits parity: ≥ 95% token match, ≥ 0.99 cos_sim final-token logits.
   - F.2 audit replay: per-class IN/OUT within ±5% on 15 labeled clips.
   - F.3 perf: p50 strictly faster than Phase 1.
   - F.4 edge cases: 4/4 pass.
4. **Regression gate (continuous, post-merge):** mAP on audit clip ≥ 0.95× PT baseline. Wired into the existing benchmark replay harness (the one v1.7.2 cold-start TTC alerter uses).
5. **No fallback path** — if TRT-LLM regresses, root-cause and fix, do not add a gate that silently routes around (`feedback_no_fallbacks_or_gates.md`).

Test infrastructure follows the TRT-LLM upstream pattern (`tests/unittest/_torch/modeling/test_modeling_qwen.py`): instantiate HF reference + TRT-LLM model, compare final-token logits with `torch.testing.assert_close(atol=0.4, rtol=0.4)`. We tighten to `atol=0.1, rtol=0.1` for our test suite since we are not testing arbitrary attention backends — only our one production config.

---

## Open Decisions for User — RESOLVED 2026-06-03

All five decisions are locked in. Recorded here for traceability.

1. **Quantization target.** bf16 only (matches current PT path, simplest parity), or also int4 weight-only (`--use_weight_only --weight_only_precision int4`) for the Colab tier? Int4 cuts `llm.engine` from ~6 GB to ~2 GB and roughly doubles throughput but introduces a second parity surface to test. **DECISION (2026-06-03): bf16 + int4 weight-only. Build BOTH engines; Phase F tests both parity surfaces.**

2. **Multi-GPU build.** Single GPU on learn02 (3080 Ti, `--tp_size 1`) or multi-rank if Colab Pro offers 2x L4? **DECISION (2026-06-03): single GPU `--tp_size 1`.**

3. **MTP (multi-token prediction) future.** Drop entirely in the TRT-LLM path (cleanest), or re-implement via TRT-LLM's Medusa speculative-decoding head (more code but preserves the 1.5–2x speculative speedup we had in PT)? **DECISION (2026-06-03): drop MTP / Medusa entirely in v1. Not even filing a v2 ticket — revisit only if measured throughput is insufficient.**

4. **Vision encoder ownership.** Adapter path (keep `vision_proj.engine`, write a thin reshape adapter) or port MoonViT into TRT-LLM's `build_multimodal_engine.py` pattern (single TRT 10.9 runtime, no dual-runtime split)? **DECISION (2026-06-03): PORT MoonViT to TRT-LLM `build_multimodal_engine.py` pattern. Reverse-engineer MoonViT into TRT-LLM's vision encoder template (CLIP/SigLIP/InternVL-style base). Single TRT 10.9 runtime end-to-end. Phase D is re-scoped accordingly (see below).**

5. **TRT version isolation strategy.** Use a dedicated `~/venvs/trtllm/` venv with TRT 10.9, leaving system TRT 10.16 intact for YOLOv12 + Phase-1 `vision_proj.engine`? Or attempt a TRT-LLM source build against TRT 10.16 (officially unsupported, may not compile)? **DECISION (2026-06-03): isolated `~/venvs/trtllm/` venv with TRT 10.9. System TRT 10.16 stays intact.**

---

## Phase D Sub-decisions (Phase B can start before these are answered)

These are specific to the MoonViT port path (Decision #4 above). They do NOT block Phase A/B/C, which are pure LM-body work.

- **D-sub-1: Base template choice.** Which TRT-LLM vision encoder template do we fork as the base — CLIP, SigLIP, Qwen2-VL ViT, or InternVL ViT? Initial lean: InternVL or Qwen2-VL since both are MoonViT-adjacent (similar patch_merger structure, Rope-style positional encoding). Decide by end of Phase C.
- **D-sub-2: Grid mode.** Ship MoonViT baked at the canonical `(36, 46)` grid only (matches Phase 1 fusion contract), or expose dynamic grid support? Static is simpler and matches our deployment pattern (lane-axis frames are constant resolution). Recommendation pending: static.
- **D-sub-3: Patch + merge dimensions.** TRT-LLM's vision encoder templates typically assume `patch_size=14` + `merge_kernel=2` (Qwen2-VL ViT convention) — does our MoonViT match exactly? If not, do we extend the template or pre-process inputs to fit?
- **D-sub-4: Custom op coverage.** Do `Rope2DReal` position embeddings and `sdpa_packed` (without `cu_seqlens`) have direct TRT-LLM equivalents? If not, do we add custom plugins or fall back to PT-side equivalents pre-baked into weights? (See Risk R3.1.)

---

## Deployment notes (verified 2026-06-03 on learn02)

These are install-time gotchas discovered during Phase A. Bake them into
every host's setup script and the Colab notebook in Phase G.

### 1. Pin `tensorrt_llm==1.2.1`
Latest stable as of 2026-06. v2.x release candidates may exist but require
CUDA 13 + Python 3.11+ which our learn02 environment doesn't satisfy cleanly.

### 2. `tensorrt_llm==1.2.1` requires CUDA 13 libs (NOT 12.8 as initial research suggested)
The wheel pins `tensorrt-cu13==10.14.1.48.post1` directly. Our earlier research
finding "v1.2.1 pins CUDA 12.8.1" was outdated (likely from v1.1.x docs).

### 3. cu13 cuBLAS + cuDNN MUST come from NVIDIA's pypi index
PyPI.org has placeholder packages named `nvidia-cublas` + `nvidia-cudnn` that
just emit a warning telling you to use NVIDIA's index. The actual cu13 wheels
live only at `https://pypi.nvidia.com`.

```bash
pip install --extra-index-url https://pypi.nvidia.com nvidia-cublas nvidia-cudnn
```

### 4. cu13 libs land at a non-standard path — LD_LIBRARY_PATH required
After install, the libs are at:
```
<venv>/lib/python3.10/site-packages/nvidia/cu13/lib/libcublasLt.so.13
<venv>/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9
```
Python's dynamic linker doesn't search these by default. Source
`scripts/trtllm_env.sh` (added in this commit) which sets
`LD_LIBRARY_PATH` accordingly and activates the venv.

### 5. Disk: TRT-LLM install needs ~25-30 GB
- Wheel deps: ~7 GB (torch, cuDNN, cuBLAS, NCCL, cuSPARSE, etc.)
- uv cache: ~21 GB (NVIDIA wheels are large)
- Site-packages: ~15 GB resolved
- On learn02, both the venv AND `UV_CACHE_DIR` must live on `/mnt/ssd0` —
  `/home` is too small.

### 6. Python version
- v1.2.1 ships wheels for **Python 3.10** + **3.12** (NOT 3.11).
- We use 3.10 on learn02. May want to switch to 3.12 in Colab.
- TRT-LLM emits a runtime warning about "Python 3.10 below recommended 3.11" — benign, can ignore.

### 7. Verified working smoke
```bash
source scripts/trtllm_env.sh
python -c "
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm import LLM
print(f'trt={trt.__version__}')        # 10.14.1.48.post1
print(f'trtllm={tensorrt_llm.__version__}')  # 1.2.1
print('ModelRunnerCpp: OK')
print('LLM API: OK')
"
```

---

## Risks

### R1 — TRT-LLM version mismatch with TRT 10.16
**Severity: medium. Likelihood: high (will occur).**
- TRT-LLM v1.2.1 pins `tensorrt==10.9.x`. Our system has 10.16.1.11.
- Pip will downgrade TRT inside the venv. System TRT (used by YOLOv12 production engines) must stay at 10.16.
- **Mitigation:** Dedicated `~/venvs/trtllm/` venv with explicit `LD_LIBRARY_PATH` scoping in the launcher script. Document the boundary clearly. Pre-flight check at runner load: `assert tensorrt.__version__.startswith("10.9")`.
- **Mitigation extended:** Also `LD_LIBRARY_PATH` the cu13 + cudnn lib dirs via `scripts/trtllm_env.sh` — see Deployment notes section.

### R2 — Custom vocab (152681) may need recipe modification
**Severity: medium. Likelihood: medium.**
- Stock Qwen2.5-3B has `vocab_size=152064`. LocateAnything-3B extends to **152681** (617 added tokens for `<box_*>`, `<class_*>`, etc).
- The upstream Qwen2 recipe reads vocab from `AutoConfig.from_pretrained()` and propagates — *should* work, but `lm_head.weight` shape `[152681, 2048]` must survive the convert.
- **Mitigation:** Phase-B day-1 task is the weight round-trip test (atol=0, rtol=0). If `lm_head` shape is wrong, fork the convert script (~50 lines), don't try to patch upstream.
- **Related history:** `project_locateanything_lm_head_root_cause.md` — vendor skips `post_init` → `tie_weights` never runs → random `lm_head`. Verify the HF checkpoint's `lm_head` is the trained one BEFORE running convert. (Our existing repair shim does this; lift the same check into `convert.py`.)

### R3 — MoonViT port to TRT-LLM template may surface structural mismatches
**Severity: high. Likelihood: medium.** (Upgraded from medium / low after the 2026-06-03 decision to port rather than adapt — we are now rewriting MoonViT inside TRT-LLM's vision encoder template, not bridging shapes around a known-good engine.)
- TRT-LLM's vision encoder templates (CLIP, SigLIP, Qwen2-VL ViT, InternVL ViT) each impose conventions on patch_size, merge_kernel, positional encoding type, and attention masking that may not match MoonViT exactly.
- Our Phase-1 evidence (`[1, 256, 2048]` output) tells us the *contract* matches, but does not tell us whether the internal ops translate cleanly.
- **Mitigation:** D.1 template audit is a hard gate before D.2 starts. If no template is within ~80% structural fit, escalate before sinking days into D.2.

### R3.1 — MoonViT custom-op porting risk (Rope2DReal + sdpa_packed)
**Severity: high. Likelihood: medium-high.** (New risk, added 2026-06-03 with Phase D re-scope.)
- MoonViT uses `Rope2DReal` position embeddings (not the `Rope2DPosEmb` variant TRT-LLM ships) and `sdpa_packed` without `cu_seqlens` (TRT-LLM's default attention path assumes `cu_seqlens`).
- These ops may not have direct TRT-LLM equivalents. Options if they don't:
  1. Add custom TRT plugins (high effort, requires C++/CUDA work)
  2. Pre-bake the positional encoding into weights at convert time (lossy if dynamic grid is needed)
  3. Fall back to PT-side equivalents wrapping the TRT-LLM engine (defeats the point of the port)
- **Mitigation:** D.2 explicitly schedules a custom-op feasibility spike on day 1. If neither option (1) nor (2) is viable within 3 days, escalate.

### R4 — learn02 GPU 1 (~12 GB) may be too tight for the build
**Severity: medium. Likelihood: medium.**
- `trtllm-build` peak VRAM is roughly 2× final engine size during the optimization phase. For a 6 GB engine, that's 12 GB — right at GPU 1's limit.
- **Mitigation:** Pre-clear GPU 1 (`fuser -k /dev/nvidia1`), build with `--workers 1`, and if OOM, fall back to Colab L4 (24 GB). The output engine is portable across sm_86 (3080) and sm_89 (L4) — both Ampere/Ada and TRT-LLM build is GPU-agnostic for compute capability inference. **However:** TRT engines are tied to compute capability. Build on sm_89 L4 → won't load on sm_86 3080. Confirm whether `trtllm-build` produces a multi-arch engine or whether we need separate builds. **Action: research before Phase C, treat as a sub-gate.**

### R5 — [DELETED 2026-06-03]
Previously: "Phase-1 `vision_proj.engine` was built against TRT 10.16, may not deserialize in TRT-LLM 10.9 venv." **Obsolete** under the 2026-06-03 port decision — `vision_proj.engine` is no longer in the deployment. The new `vision_encoder.engine` is built natively in TRT-LLM 10.9 via `build_multimodal_engine.py`, so there is no cross-version deserialization surface. Risk eliminated by design.

### R6 — Speculation discipline
Per `feedback_speculation_discipline.md`: this design doc cites only verified TRT-LLM behavior (release notes v1.2.1, recipe `examples/models/core/qwen/`, runner code `tensorrt_llm/runtime/multimodal_model_runner.py`, unit-test `test_modeling_qwen.py`). Any item not directly sourced from those is flagged here:
- "1.5–3x speedup vs Phase 1" is an industry-comparable claim (vLLM/TGI report similar) but is NOT yet measured on our path — Phase F.3 is the verification gate.
- "Multi-arch engine portability sm_86 vs sm_89" is an explicit unknown (R4) flagged for pre-Phase-C research.

---

## Appendix: References

- `single_file_trt_design.md` — predecessor design, IIfConditional approach, post-mortem
- `learn02_polygraphy_llm_decode.txt`, `learn02_polygraphy_llm_prefill.txt` — Phase-1 polygraphy traces
- `learn02_build_strongly_typed.log` — myelin/codeGenerator.cpp:3811 failure log
- TRT-LLM v1.2.1 release notes (April 20, 2026)
- TRT-LLM `examples/models/core/qwen/README.md`
- TRT-LLM `examples/models/core/multimodal/README.md`
- TRT-LLM `tests/unittest/_torch/modeling/test_modeling_qwen.py`
- TRT-LLM issue #2104 (`max_multimodal_len` == `max_prompt_embedding_table_size`)
- `project_locateanything_lm_head_root_cause.md` (memory note)
- `feedback_no_fallbacks_or_gates.md`, `feedback_speculation_discipline.md`, `feedback_block_before_run.md` (memory notes)
