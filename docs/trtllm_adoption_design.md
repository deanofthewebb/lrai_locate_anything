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

Canonical TRT-LLM multimodal layout (two engines, one parent dir):

```
lrai_locate_anything_trtllm/
├── vision/
│   └── vision_proj.engine        ~2.0 GB   TRT 10.16, our Phase-1 MoonViT+projector, UNCHANGED
└── llm/
    ├── llm.engine                ~6.0 GB   TRT-LLM, Qwen2.5-3B body (152681 vocab)
    ├── config.json               ~ 8 KB    TRT-LLM engine config (paged_kv, dtype, plugins)
    └── rank0.engine -> llm.engine          symlink for single-rank load
```

**Total: ~8.0 GB.** Matches the original Phase-2 target (`single_file_trt_design.md` §3.2).

Distribution: both files uploaded to `s3://data-labeling.livereachmedia.com/datasets/safetunnel/models/locany_trtllm/` (mirroring the SAM3 pattern — do NOT use `s3://trackerbot/`, AccessDenied).

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

### Phase D — Vision encoder integration (3–4 days, **medium-large**, risk: high)

**Deliverable:** `trtllm_prod/moonvit_adapter.py` bridges our `vision_proj.engine` output into TRT-LLM's `prompt_embedding_table` input contract.

**Decision (final):** Keep our existing `vision_proj.engine`. Do NOT use `build_multimodal_engine.py`.

Rationale:
- TRT-LLM's `build_multimodal_engine.py --model_type qwen2_vl` hardcodes the standard Qwen ViT (window attention + RoPE, sm-flow 14x14 grid). LocateAnything's **MoonViT** is structurally different (different patch grid, different RoPE, different projector). Forcing it through that script would mean a one-off fork.
- We already have `vision_proj.engine` working from Phase 1 — output is `[1, 256, 2048]` bf16, already projected into LLM hidden dim.
- The runner-side contract is purely tensor-shape: any TRT engine that outputs `[B*N, H]` (or is reshapable to that) where `H == llm.hidden_size` is acceptable. This is documented in `tensorrt_llm/runtime/multimodal_model_runner.py` — vision engines are loaded via generic `Session.from_serialized_engine(buf)`, no TRT-LLM-specific metadata.

Concrete tasks:
1. `moonvit_adapter.py`: wraps `vision_proj.engine` via `tensorrt.Runtime` + `Session`. Runs forward, returns `visual_features.reshape(N, H)` as a contiguous CUDA tensor.
2. Verify output shape `[256, 2048]` matches `max_multimodal_len=256` from Phase C build.
3. PT-parity test: same image → MoonViT (PT) → projector (PT) vs same image → `vision_proj.engine` → reshape. `torch.testing.assert_close(atol=0.05, rtol=0.05)` on bf16 features. (Looser than the TRT-LLM unit-test atol=0.4 because we are testing the vision half in isolation, on a known-good engine from Phase 1.)
4. Build a small fixture: `tests/fixtures/lane_axis_frame_001.jpg` + golden `visual_features_pt.npy`.

**Gate:** Cos-sim ≥ 0.995 between PT visual features and engine visual features on the fixture image.

**Risk:** Phase-1 fusion may have baked-in a projector output dim that doesn't exactly match Qwen2.5-3B's `hidden_size=2048`. Verify on day 1, fall back to inserting a thin Python projector in `moonvit_adapter.py` if so. (See Risk R3.)

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

## Open Decisions for User

These must be resolved **before Phase B starts**:

1. **Quantization target.** bf16 only (matches current PT path, simplest parity), or also int4 weight-only (`--use_weight_only --weight_only_precision int4`) for the Colab tier? Int4 cuts `llm.engine` from ~6 GB to ~2 GB and roughly doubles throughput but introduces a second parity surface to test. **Recommendation: bf16 only for v1, defer int4 to v2.**

2. **Multi-GPU build.** Single GPU on learn02 (3080 Ti, `--tp_size 1`) or multi-rank if Colab Pro offers 2x L4? **Recommendation: single GPU. TP=2 doubles the build-and-test matrix for a model that already fits in 24 GB.**

3. **MTP (multi-token prediction) future.** Drop entirely in the TRT-LLM path (cleanest), or re-implement via TRT-LLM's Medusa speculative-decoding head (more code but preserves the 1.5–2x speculative speedup we had in PT)? **Recommendation: drop for v1, file a v2 ticket. Medusa requires training extra heads — out of scope for this migration.**

4. **Vision encoder ownership.** Confirmed in Phase D plan: keep our `vision_proj.engine`, do NOT use `build_multimodal_engine.py`. **Please confirm — this is load-bearing for the Phase D plan.**

5. **TRT version isolation strategy.** Use a dedicated `~/venvs/trtllm/` venv with TRT 10.9, leaving system TRT 10.16 intact for YOLOv12 + Phase-1 `vision_proj.engine`? Or attempt a TRT-LLM source build against TRT 10.16 (officially unsupported, may not compile)? **Recommendation: isolated venv. Source build adds 3+ days of unknowns.**

---

## Risks

### R1 — TRT-LLM version mismatch with TRT 10.16
**Severity: medium. Likelihood: high (will occur).**
- TRT-LLM v1.2.1 pins `tensorrt==10.9.x`. Our system has 10.16.1.11.
- Pip will downgrade TRT inside the venv. System TRT (used by YOLOv12 production engines) must stay at 10.16.
- **Mitigation:** Dedicated `~/venvs/trtllm/` venv with explicit `LD_LIBRARY_PATH` scoping in the launcher script. Document the boundary clearly. Pre-flight check at runner load: `assert tensorrt.__version__.startswith("10.9")`.

### R2 — Custom vocab (152681) may need recipe modification
**Severity: medium. Likelihood: medium.**
- Stock Qwen2.5-3B has `vocab_size=152064`. LocateAnything-3B extends to **152681** (617 added tokens for `<box_*>`, `<class_*>`, etc).
- The upstream Qwen2 recipe reads vocab from `AutoConfig.from_pretrained()` and propagates — *should* work, but `lm_head.weight` shape `[152681, 2048]` must survive the convert.
- **Mitigation:** Phase-B day-1 task is the weight round-trip test (atol=0, rtol=0). If `lm_head` shape is wrong, fork the convert script (~50 lines), don't try to patch upstream.
- **Related history:** `project_locateanything_lm_head_root_cause.md` — vendor skips `post_init` → `tie_weights` never runs → random `lm_head`. Verify the HF checkpoint's `lm_head` is the trained one BEFORE running convert. (Our existing repair shim does this; lift the same check into `convert.py`.)

### R3 — MoonViT integration may surface feature-shape mismatches
**Severity: medium. Likelihood: low (we have Phase-1 evidence it's `[1, 256, 2048]`).**
- TRT-LLM's `prompt_embedding_table` expects `[N, H]` where `H == llm.hidden_size = 2048`.
- Our Phase-1 `vision_proj.engine` outputs `[1, 256, 2048]` (verified in `learn02_polygraphy_summary.log`).
- **Mitigation:** Day-1 of Phase D verifies the contract empirically. If misaligned, insert a Python `nn.Linear` projector in `moonvit_adapter.py` (no engine rebuild needed for v1).

### R4 — learn02 GPU 1 (~12 GB) may be too tight for the build
**Severity: medium. Likelihood: medium.**
- `trtllm-build` peak VRAM is roughly 2× final engine size during the optimization phase. For a 6 GB engine, that's 12 GB — right at GPU 1's limit.
- **Mitigation:** Pre-clear GPU 1 (`fuser -k /dev/nvidia1`), build with `--workers 1`, and if OOM, fall back to Colab L4 (24 GB). The output engine is portable across sm_86 (3080) and sm_89 (L4) — both Ampere/Ada and TRT-LLM build is GPU-agnostic for compute capability inference. **However:** TRT engines are tied to compute capability. Build on sm_89 L4 → won't load on sm_86 3080. Confirm whether `trtllm-build` produces a multi-arch engine or whether we need separate builds. **Action: research before Phase C, treat as a sub-gate.**

### R5 — Phase-1 `vision_proj.engine` was built against TRT 10.16
**Severity: low. Likelihood: low.**
- Vision engine runs in the *system* TRT 10.16 path (separate from TRT-LLM's bundled 10.9). The TRT-LLM runner loads `vision_proj.engine` via the venv's TRT (10.9), which may refuse to deserialize a 10.16-serialized engine.
- **Mitigation:** TRT engines are forward-compatible within a major version (10.x). Test on Phase-D day-1: deserialize 10.16 engine inside the 10.9 venv. If it fails, rebuild `vision_proj.engine` against TRT 10.9 (a one-time operation; the source code from Phase 1 is preserved).

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
