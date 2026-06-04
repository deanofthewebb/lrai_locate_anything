# Phase D — MoonViT-on-SigLIP Port Delta Map

Side-by-side delta between TRT-LLM's stock SigLIP vision tower
(`/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_siglip.py`,
which re-exports HF `transformers.models.siglip.modeling_siglip` for the embedding
+ attention layers and uses `CLIPEncoder` as the encoder body) and our existing
MoonViT export wrapper (`lrai_locate_anything/export/vision.py` +
`apply_vision_patches` monkey-patcher). The goal is to mutate SigLIP into MoonViT
with the smallest possible diff so we inherit TRT-LLM's plugin path, KV layout,
and bf16 autocast contract.

The donor module is the SigLIP ViT tower (patch-embed + N transformer layers).
The grafts are: (a) MoonViT's baked, pre-interpolated 2D pos_emb in place of
SigLIP's 1D learnable `position_embedding`; (b) real-valued 2D RoPE inside
attention; (c) MoonViT's `StaticPatchMerger` after the encoder; (d) MoonViT's
`mlp1` projector. Everything else — pre-norm placement, MLP geometry, residuals,
layer count, head-dim partitioning — stays exactly as SigLIP ships it.

---

## 1. Side-by-side architecture table

| Layer / responsibility | SigLIP (TRT-LLM stock) | MoonViT (our export/vision.py + apply_vision_patches) | Port action |
|---|---|---|---|
| Pixel input contract | `(B, 3, H, W)` whole image; `patch_embedding = nn.Conv2d(3, D, k=patch, s=patch)`; output flatten(2).transpose(1,2) -> `(B, N, D)` | Pre-patchified `(L_pre, 3, 14, 14)` tensor; `vit.patch_embed.proj` (Conv2d 3->D k=14 s=14) applied per-patch then `.view(L_pre, -1)` -> `(L_pre, D)` | In `SiglipVisionEmbeddings.forward` accept `(L_pre, 3, 14, 14)` and emit `(L_pre, D)` (no batch axis); update `SiglipVisionTransformer.forward` upstream to skip its `(B, N, D)` reshape. Patch conv weights map 1:1 — both are `Conv2d(3, D, 14, 14)` |
| Positional embedding storage | `self.position_embedding = nn.Embedding(num_positions, D)` + `register_buffer("position_ids", arange(num_positions))`, `num_positions = (image_size//patch)**2` | `self.register_buffer("pos_emb_baked", F.interpolate(weight.permute(2,0,1).unsqueeze(0), size=(grid_h, grid_w), mode=interp_mode, align_corners=False).squeeze(0).permute(1,2,0).flatten(end_dim=1))` baked from `vit.patch_embed.pos_emb.weight (H_max, W_max, D)` | Replace `position_embedding` + `position_ids` registration with `pos_emb_baked` buffer of shape `(grid_h*grid_w, D)`; delete `interpolate_pos_encoding` branch (we pre-interpolate at init) |
| Positional embedding application | `embeddings = embeddings + self.position_embedding(self.position_ids)` (1-D table lookup, absolute) | `x = x + self.pos_emb_baked` straight broadcast on the `(L_pre, D)` token stream | Replace lookup line with direct buffer add; `position_ids` no longer needed |
| CLS token | None (SigLIP has no CLS — matches MoonViT) | None | No change |
| Pre-norm layout | `residual=x; x=layer_norm1(x); x=self_attn(...); x=residual+x; residual=x; x=layer_norm2(x); x=mlp(x); x=residual+x` | Identical pre-norm pattern (MoonViT canonical encoder uses the same Pre-LN block) | No change |
| Attention class | `CLIPAttention` (subclass of TRT-LLM `Attention` base) with `pos_embd_params=None`, `bias=True`, `num_key_value_heads=num_attention_heads` (MHA) | Module-level fn dispatched via `VL_VISION_ATTENTION_FUNCTIONS["sdpa_packed"]`; (q,k,v) packed `(L, H*D)` then 2D RoPE applied to q,k, then `F.scaled_dot_product_attention` | Insert `apply_rope_real(q, k, freqs_cos, freqs_sin)` into `CLIPAttention.forward` between q/k/v projection and SDPA; `o_proj` (bias=True) and head split unchanged |
| RoPE storage | None — SigLIP is absolute-pos only | `Rope2DReal` module with `freqs_cos/freqs_sin` real buffers shape `(H_max, W_max, dim/2)`; `get_freqs_cis(h, w)` slices to `[:h, :w]` and stacks | Add `self.freqs_cos` / `self.freqs_sin` buffers to each ported attention layer (or hoist onto the `SiglipVisionTransformer` and pass down); slice to `[:grid_h, :grid_w]` at init since grid is baked |
| Attention kernel | Dispatch through `ALL_ATTENTION_FUNCTIONS` (eager/sdpa/flash); standard `(B, H, T, D_head)` layout | `sdpa_packed`: `(L, H, D_head)` packed-sequence SDPA, no `cu_seqlens` (single-image), output `.flatten(start_dim=-2)` | Keep SigLIP's `(B, H, T, D_head)` SDPA — we run B=1, T=L_pre, identical math. Do not adopt `sdpa_packed`; the stock path is already traceable |
| MLP | `nn.Linear(D, 4D) -> gelu_pytorch_tanh -> nn.Linear(4D, D)` | Same shape; MoonViT canonical uses GELU as well | No change (verify activation string matches — `gelu_pytorch_tanh` vs `gelu`; weights may need an activation-name remap) |
| Encoder body | `SiglipEncoder = CLIPEncoder`, N identical pre-norm layers, returns `(B, N, D)` | `vit.encoder` — N identical pre-norm layers, returns `(L_pre, D)` | Adopt SigLIP encoder verbatim; only the attention forward is modified for RoPE |
| Post-encoder norm | `SiglipVisionTransformer` ends with `post_layernorm(last_hidden_state)` (HF) | MoonViT has no trailing LN before merger (canonical) | Delete or no-op `post_layernorm` in our subclass (set to `nn.Identity()`); MoonViT weights have no analog |
| Patch merger | None — SigLIP feeds `(B, N, D)` straight to the projector | `StaticPatchMerger`: reshape `(L_pre, D) -> (nh, kh, nw, kw, D) -> permute(0,2,1,3,4) -> view(nh*nw, kh*kw*D)` | New post-encoder module on `SiglipVisionModel`, before projector. Reads `(merge_kh, merge_kw)` baked at construction |
| Projector | TRT-LLM SigLIP has no built-in projector (LLM-side MM-projector lives in the multimodal model wrapper) | `model.mlp1` — bf16 Linear stack mapping `(kh*kw*D) -> hidden_text` | Fuse `mlp1` weights into our `SiglipVisionModel` subclass as `self.projector`; final output `(L_post, hidden_text)` |
| Output shape | `(B, N, D)` (raw vision hidden) | `(L_post, hidden_text)` post-projector | Match MoonViT: emit `(L_post, hidden_text)` after merger + projector, then unsqueeze(0) in `MoonViTAdapter` for TRT-LLM's prompt-table `(1, L_post, hidden)` |
| dtype | TRT-LLM uses `config.torch_dtype` (commonly fp16/bf16); SDPA runs in that dtype | `torch.autocast("cuda", dtype=torch.bfloat16)` wraps the whole vision forward in the canonical MoonViT runner | Wrap our `SiglipVisionModel.forward` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`; let SDPA and Linear ops promote internally |

---

## 2. Port sites (exact code changes)

1. **Site 1 — `SiglipVisionEmbeddings.__init__` (HF transformers, subclass it in `lrai_locate_anything/trtllm_prod/siglip_moonvit.py`):** delete `self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)` and `self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)`; replace with `self.register_buffer("pos_emb_baked", _bake_pos_emb(moonvit_pos_emb_weight, grid_h, grid_w))` using the exact `F.interpolate` + `permute(2,0,1).unsqueeze(0)` + `squeeze().permute(1,2,0).flatten(end_dim=1)` chain from `export/vision.py:35-44`.
2. **Site 2 — `SiglipVisionEmbeddings.forward`:** replace `embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1,2); embeddings = embeddings + self.position_embedding(self.position_ids)` with `embeddings = self.patch_embedding(pixel_values).view(pixel_values.size(0), -1); embeddings = embeddings + self.pos_emb_baked`. Delete the `if interpolate_pos_encoding:` branch entirely — grid is baked.
3. **Site 3 — `CLIPAttention.forward` (used as `SiglipAttention`):** after `q = self.q_proj(hidden_states); k = self.k_proj(hidden_states); v = self.v_proj(hidden_states)` and the `(B, T, H, D_head)` reshape, insert `q, k = apply_rope_real(q, k, self.freqs_cos, self.freqs_sin)`. Both freqs buffers are pre-sliced to `(grid_h, grid_w, D_head/2)` and flattened to `(L_pre, D_head/2)` at init so the `freqs.unsqueeze(-2)` broadcast against `(L_pre, H, D_head/2)` works.
4. **Site 4 — `CLIPAttention.__init__`:** register `self.freqs_cos` and `self.freqs_sin` buffers, derived from `Rope2DReal.get_freqs_cis(grid_h, grid_w)` then `.unbind(-1)`. Each layer shares the same freqs, so an alternative is to hoist them onto the encoder root and pass via a `freqs` kwarg through `CLIPEncoderLayer.forward`. Prefer the buffer-per-layer approach to avoid touching `CLIPEncoder.forward` signatures.
5. **Site 5 — `SiglipVisionTransformer.__init__` / `forward`:** set `self.post_layernorm = nn.Identity()` (MoonViT has no trailing LN); change forward to take `(L_pre, 3, 14, 14)` and skip the `(B, N, D)` reshape — `B=1` is implicit and `L_pre` is the only token axis. Remove the `interpolate_pos_encoding` kwarg from the call site.
6. **Site 6 — `SiglipVisionModel` (new subclass `MoonViTOnSiglip`):** after `self.vision_model(pixel_values)` returns `(L_pre, D)`, run a new `self.patch_merger = StaticPatchMerger(merge_kh, merge_kw, grid_h, grid_w)` to produce `(L_post, kh*kw*D)`, then `self.projector = _build_projector_from_mlp1(moonvit_model.mlp1)` to land at `(L_post, hidden_text)`.
7. **Site 7 — autocast wrapper:** wrap `MoonViTOnSiglip.forward` body in `with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):` so SDPA, Linear, and Conv2d all run in bf16; matches MoonViT canonical runtime and TRT-LLM's plugin dtype for the vision tower.
8. **Site 8 — weight loader (`from_moonvit_state_dict`):** classmethod on `MoonViTOnSiglip` that takes a MoonViT `state_dict` and remaps keys: `vit.patch_embed.proj.* -> vision_model.embeddings.patch_embedding.*`, `vit.encoder.layers.{i}.{ln1,ln2,attn,mlp}.* -> vision_model.encoder.layers.{i}.{layer_norm1,layer_norm2,self_attn,mlp}.*`, `vit.patch_embed.pos_emb.weight` is consumed by `_bake_pos_emb` at construction (not loaded as a param), `mlp1.* -> projector.*`. Strict=False to allow `post_layernorm` to be absent.

---

## 3. What stays unchanged

- **Encoder block structure**: pre-norm, residual sandwich, layer count, head count, head dim, hidden dim — all read from the MoonViT config and fed into the SigLIP config object so the layer geometry matches.
- **MLP**: same `Linear(D, 4D) -> GELU -> Linear(4D, D)` (verify activation string `gelu_pytorch_tanh` vs MoonViT's `gelu`; if mismatch, override `config.hidden_act`).
- **LayerNorm placement**: `layer_norm1` before attn, `layer_norm2` before mlp — identical to MoonViT.
- **Attention math**: q/k/v Linear with bias, head split, SDPA, o_proj with bias. We only inject RoPE between q/k projection and SDPA.
- **Patch conv**: `Conv2d(3, D, kernel=14, stride=14)` — bit-identical between SigLIP and MoonViT; weights load 1:1.
- **TRT-LLM `Attention` base class**: keep `pos_embd_params=None` so TRT-LLM's internal RoPE path stays disabled — our 2D real-RoPE runs in the Python forward before SDPA.
- **`SiglipEncoder = CLIPEncoder` aliasing**: leave it; we only subclass `CLIPAttention`.

---

## 4. Dynamic grid support (D-sub-2)

The MVP bakes `(grid_h, grid_w)` at construction — same contract as `VisionForExport` today. For dynamic grids we re-interpolate `pos_emb_baked` and re-slice `freqs_cos/sin` at runtime:

- **pos_emb_baked**: store the *un-interpolated* `pos_emb.weight (H_max, W_max, D)` as a non-persistent buffer. On grid change, call `F.interpolate(weight.permute(2,0,1).unsqueeze(0), size=(grid_h, grid_w), mode="bicubic", align_corners=False)` and overwrite `self.pos_emb_baked`. Cost: one `F.interpolate` per resolution change, cached by `(grid_h, grid_w)` key.
- **freqs_cos/freqs_sin**: store the full `(H_max, W_max, D_head/2)` Rope2DReal tables; on grid change, slice `freqs_cos[:grid_h, :grid_w].flatten(end_dim=1)` and re-bind. Each attention layer reads from a shared `freqs_provider` to avoid N copies.
- **Trace-friendly path**: keep the baked-at-init contract for ONNX/TRT export (one engine per resolution). The dynamic path runs only in eager PT (parity tests, dev mode, MoonViTAdapter fallback).
- **TRT-LLM plugin compatibility**: the SigLIP TRT-LLM model expects a static `image_size` in config; for dynamic vision we'll build one engine per `(grid_h, grid_w)` bucket and route by aspect-ratio bucket in `MoonViTAdapter`.

---

## 5. Projector + bf16 (D-sub-3 + D-sub-4)

- **Projector graft**: `mlp1` from the MoonViT model — typically `LayerNorm(kh*kw*D) -> Linear(kh*kw*D, hidden_text) -> GELU -> Linear(hidden_text, hidden_text)`. We expose `self.projector = nn.Sequential(...)` on `MoonViTOnSiglip` and load weights via `_build_projector_from_mlp1(moonvit_model.mlp1)` which copies sub-modules directly (no remap). The projector lives on the vision-model wrapper, not inside `SiglipVisionTransformer`, so TRT-LLM's existing `SiglipVisionModel` plugin slot stays clean and the projector becomes an outer-graph extension.
- **bf16 autocast**: wrap `MoonViTOnSiglip.forward` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16)`. Internal Linear/SDPA/Conv2d auto-cast; pos-emb buffer stored in fp32 and cast on the add (or stored bf16 if memory matters). RoPE freqs stay fp32 — `apply_rope_real` does the float-up reshape (`(xr, xi) -> out_r=xr*cos-xi*sin`) in fp32 for numerical stability, then casts back to the input dtype before SDPA.
- **TRT-LLM contract**: the SigLIP TRT-LLM model reads `config.torch_dtype`; set it to `torch.bfloat16` on the config object so plugin selection matches. The autocast context is a belt-and-braces for the PT-eager parity path.

---

## 6. Parity test plan

`tests/test_phase_d_parity.py`:

1. **Load canonical MoonViT** via the existing `apply_vision_patches`-patched model (the source of truth that `export/vision.py` mirrors). Build `VisionForExport(vit, grid_h=36, grid_w=46)` plus the StaticPatchMerger + mlp1 projector — together the "reference" stack.
2. **Load ported `MoonViTOnSiglip`** with `from_moonvit_state_dict(moonvit.state_dict(), grid_h=36, grid_w=46)`.
3. **Fixed input**: `pixel_values = torch.randn(1656, 3, 14, 14, dtype=torch.float16, device="cuda")` (1656 = 36*46) with a fixed seed.
4. **Run both** under `torch.no_grad()` and `torch.autocast("cuda", bfloat16)` and collect output tensors.
5. **Assert** `torch.max(torch.abs(ref - ported)).item() < 1e-3` on the post-projector `(L_post, hidden_text)` tensor. Also gate on `cosine_similarity > 0.9999` per token row to catch shape-correct but numerically-degenerate ports.
6. **Sub-test per layer**: also dump intermediate `(L_pre, D)` after `vision_model` (pre-merger) and assert max-abs-diff < 5e-4 there, so a mismatch points to embedding vs attention vs merger vs projector unambiguously.
7. **Run on the SAM3 calibration image** (s3://data-labeling.livereachmedia.com/datasets/safetunnel/...) end-to-end through the LLM head to confirm the bounding-box JSON output matches the canonical model on a known frame.

---

## Appendix A — Tightened Site Specs

These specs are the concrete shape contracts produced by the parallel deep-dive on each port site. The corresponding `# TODO Port site N` markers in `lrai_locate_anything/trtllm_prod/modeling_moonvit.py` reference the section IDs below (A.1 through A.6). When implementing, prefer these specs over the higher-level prose in section 2 — these resolve the actual shape/dtype/broadcasting ambiguities that surfaced during the prep pass.

Canonical baked dims (one-shot MVP engine, aspect-ratio bucket 36x46):
- `B = 1` (single-image vision-only, no KV cache).
- `L_pre = grid_h * grid_w = 36 * 46 = 1656` tokens after patchification.
- `D = config.hidden_size = 1152` (SigLIP-So400m).
- `H = config.num_attention_heads = 16`.
- `D_head = D / H = 72`.
- `kh = kw = 2` (StaticPatchMerger), so `L_post = L_pre / (kh*kw) = 414`, projector-in dim = `kh*kw*D = 4608`.
- `text_hidden = 2048` (LM-side hidden, used by `mlp1` projector).

---

### Site 1 (SiglipVisionEmbeddings.__init__) — Concrete spec

**Buffers registered at init:**
- `self.pos_emb_baked`: `(L_pre, D) = (1656, 1152)`, dtype = module dtype (bf16 at runtime, fp32 source weight, cast at register time). Baked from MoonViT `vit.patch_embed.pos_emb.weight` of shape `(H_max, W_max, D)` via:
  - `w = weight.permute(2, 0, 1).unsqueeze(0)` -> `(1, D, H_max, W_max)`
  - `w = F.interpolate(w, size=(grid_h, grid_w), mode="bicubic", align_corners=False)` -> `(1, D, grid_h, grid_w)`
  - `pos_emb_baked = w.squeeze(0).permute(1, 2, 0).flatten(end_dim=1)` -> `(L_pre, D)`
- `self.grid_hws_baked`: `(1, 2)` int32, just stored for VisionForExport parity (Site 3/4 RoPE reads from a separate freqs buffer).
- `self.L_pre_baked`: python int (not a buffer), used by forward to recover B.

**Removed from stock SigLIP:**
- `self.position_embedding = nn.Embedding(num_positions, embed_dim)` — deleted.
- `self.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)` — deleted.

**Edge cases:**
- Source weight dtype is fp32; cast to module dtype after interpolation, not before, to avoid bf16 bicubic accuracy loss.
- `interpolate_pos_encoding` branch is dead — grid is baked, so the kwarg path is removed entirely (also from the forward signature in Site 2).

**Residual risks (resolve in code):**
- Whether `pos_emb_baked` should be persistent: default `persistent=True` so checkpointing carries it; flip to `False` only if we adopt the dynamic-grid re-bake path.
- Dtype storage tradeoff (fp32 stable add vs bf16 memory) — punt to runtime decision after first parity run.

---

### Site 2 (SiglipVisionEmbeddings.forward) — Concrete spec

**Input contract (differs from stock SigLIP):**
- `pixel_values`: `(B*L_pre, 3, 14, 14)`, bf16/fp16, pre-patchified by orchestrator. NOT `(B, 3, H, W)`.

**Tensor shapes at each step:**
- After `patch_embedding(pixel_values)` (Conv2d 3->1152, k=14, s=14): `(B*L_pre, 1152, 1, 1)`.
- After `.view(B_times_L, -1)`: `(B*L_pre, 1152)`.
- `self.pos_emb_baked`: `(L_pre, 1152)`.
- After broadcast add (B=1 fast path): `(L_pre, 1152)`.
- After final `.view(B, L_pre, D)`: `(1, 1656, 1152)` — this is the shape contract returned to `SiglipVisionTransformer.forward`.

**Code skeleton:**
```python
def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    B_times_L = pixel_values.shape[0]
    assert B_times_L % self.L_pre_baked == 0, \
        f"got {B_times_L} tokens, not divisible by baked L_pre={self.L_pre_baked}"
    B = B_times_L // self.L_pre_baked

    x = self.patch_embedding(pixel_values)        # (B*L_pre, D, 1, 1)
    x = x.view(B_times_L, -1)                      # (B*L_pre, D)

    if B == 1:
        x = x + self.pos_emb_baked                 # (L_pre, D) + (L_pre, D)
    else:
        x = x.view(B, self.L_pre_baked, -1)
        x = x + self.pos_emb_baked.unsqueeze(0)    # (B, L_pre, D)
        x = x.view(B_times_L, -1)

    return x.view(B, self.L_pre_baked, -1)         # (B, L_pre, D)
```

**Edge cases:**
- We DO return `(B, L_pre, D)` to preserve the `SiglipVisionTransformer.forward` contract (modeling_siglip.py:51-53). Site 5 (caller) keeps its reshape to `(B*N, D)` for `CLIPEncoder`/`AttentionMetadata`. Do NOT preempt the reshape here.
- `interpolate_pos_encoding` kwarg removed from signature.
- `B == 1` fast path avoids an unnecessary `unsqueeze`+`view` pair on the hot path.

**Residual risks (resolve in code):**
- Whether the TRT-LLM multimodal-engine builder traces with `(B, 3, H, W)` dummy — we override the dummy in `build.py` (Site 6 cross-ref) to a `(L_pre, 3, 14, 14)` tensor.

---

### Site 3 (CLIPAttention.forward) — Concrete spec

**Tensor shapes at each step (B=1, T=L_pre=1656, H=16, D_head=72):**
- `hidden_states` in: `(B*T, D) = (1656, 1152)` after Site 5's reshape (per `CLIPEncoder` contract).
- `q = self.q_proj(hidden_states)`: `(B*T, D) = (1656, 1152)`.
- `q.view(B, T, H, D_head)`: `(1, 1656, 16, 72)`.
- `q.transpose(1, 2)`: `(B, H, T, D_head) = (1, 16, 1656, 72)` (SDPA layout).
- Same for `k`, `v`.

**RoPE freq shapes:**
- `self.freqs_cos`, `self.freqs_sin`: `(L_pre, D_head/2) = (1656, 36)` (pre-flattened at init from `(grid_h, grid_w, D_head/2)`).
- For broadcast against `(B, H, T, D_head/2)` (the half-dim view of q/k inside `apply_rope_real`): reshape freqs to `(1, 1, L_pre, D_head/2)` — i.e. `unsqueeze(0).unsqueeze(0)`. NOT `unsqueeze(-2)`; that pattern was for the `(L, H, D_head/2)` packed layout in MoonViT canonical. SigLIP's `(B, H, T, D_head)` puts the head axis at dim=1, not dim=-2.

**Code skeleton:**
```python
# After q,k,v projection but before SDPA
B, T, _ = hidden_states.shape[0] // self.L_pre_baked, self.L_pre_baked, None
# (or recover B,T from hidden_states.view(B, T, D) — see Site 5 contract)

q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
# q, k, v each: (B, H, T, D_head)

# 2D real RoPE — freqs broadcast over batch and head axes.
freqs_cos = self.freqs_cos.view(1, 1, T, self.head_dim // 2)  # (1, 1, T, D_head/2)
freqs_sin = self.freqs_sin.view(1, 1, T, self.head_dim // 2)
q, k = apply_rope_real(q, k, freqs_cos, freqs_sin)  # same (B, H, T, D_head)

# Then standard SigLIP SDPA path.
attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
# (B, H, T, D_head) -> (B, T, D) -> o_proj
attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
out = self.o_proj(attn_out)
```

**Edge cases:**
- `apply_rope_real` splits `q` into `(q_real, q_imag)` along the last dim (pairs of consecutive channels). Verify pairing convention matches MoonViT canonical (`patches.py:apply_rope_real`) — half-then-half vs interleaved is a silent failure.
- fp32 RoPE math, then cast back to module dtype before SDPA (matches MoonViT canonical for numerical stability).
- `o_proj` keeps `bias=True` from stock SigLIP — no change.

**Residual risks (resolve in code):**
- Whether to recover `B`, `T` from `hidden_states.shape` directly or rely on a baked `self.L_pre_baked`. If the encoder loop passes `(B, T, D)` (not yet flattened to `(B*T, D)`), the view ops simplify — depends on Site 5's exact reshape decision.
- Freq-buffer placement: per-layer copy vs shared on the encoder root. MVP uses per-layer to keep `CLIPEncoderLayer.forward` signature untouched.

---

### Site 4 (CLIPAttention.__init__) — Concrete spec

**Buffers registered at init (one set per attention layer):**
- `self.freqs_cos`: `(L_pre, D_head/2) = (1656, 36)`, fp32.
- `self.freqs_sin`: `(L_pre, D_head/2) = (1656, 36)`, fp32.

**Construction:**
- `freqs_cis = Rope2DReal(D_head, max_grid_h, max_grid_w).get_freqs_cis(grid_h, grid_w)`. Returns `(grid_h, grid_w, D_head/2, 2)` (last axis = (cos, sin)).
- `freqs_cos, freqs_sin = freqs_cis.unbind(-1)` — each `(grid_h, grid_w, D_head/2)`.
- Flatten: `freqs_cos = freqs_cos.flatten(end_dim=1)`, same for sin — each becomes `(L_pre, D_head/2)`.
- Register both as non-persistent buffers (re-baked from config each construction).

**Edge cases:**
- All N=27 layers share identical freqs (the 2D-RoPE table is position-only, not layer-specific). Per-layer copy wastes ~27 * 1656 * 36 * 4 = 6.4 MB fp32 but avoids touching `CLIPEncoderLayer.forward(self, hidden_states, attention_mask)` to add a `freqs=` kwarg.
- fp32 storage is intentional — `apply_rope_real` consumes fp32 freqs and computes in fp32, then casts back.

**Residual risks (resolve in code):**
- If TRT-LLM's `CLIPAttention` base does NOT round-trip arbitrary buffers through its `Attention` mixin, may need to register on a sibling `nn.Module` and pass into `forward` via a captured closure. Verify on first build.

---

### Site 5 (SiglipVisionTransformer) — Concrete spec

**Init changes:**
- `self.post_layernorm = nn.Identity()` — MoonViT has no trailing LN. We do NOT delete the attribute; the HF parent constructor sets it and downstream code may probe `hasattr`. Setting to `Identity()` is the minimal-diff fix.
- `interpolate_pos_encoding` argument removed from forward signature.

**Forward shape contract:**
- Input `pixel_values`: `(B*L_pre, 3, 14, 14)` (matches Site 2's input).
- `hidden_states = self.embeddings(pixel_values)`: `(B, L_pre, D) = (1, 1656, 1152)` (per Site 2 return contract).
- Reshape for encoder: `hidden_states.view(B * L_pre, D) = (1656, 1152)` if the encoder expects packed `(B*T, D)`; OR keep `(B, L_pre, D)` if the encoder expects standard `(B, T, D)`. CLIPEncoder takes `(B, T, D)` — preserve it.
- Encoder out: `(B, L_pre, D)`.
- `last_hidden_state = self.post_layernorm(encoder_out)`: `(B, L_pre, D)` — Identity passthrough.
- Return: `(B, L_pre, D)` matching HF return contract.

**Edge cases:**
- `B=1` is implicit but we still emit a batch axis so downstream `MoonViTOnSiglip` (Site 6) can run the patch merger on `(B, L_pre, D)` -> `(B, L_post, kh*kw*D)`.

**Residual risks (resolve in code):**
- TRT plugin baking the LN — even with `Identity` in PT-eager, the plugin may still have an LN slot. May need to load identity weights (gamma=1, beta=0) into the plugin's LN parameters at engine-build time.

---

### Site 6 (MoonViTOnSiglip wrapper + projector) — Concrete spec

**Composition:**
- `self.vision_model = SiglipVisionTransformer(config)` — patched with sites 1-5.
- `self.patch_merger = StaticPatchMerger(merge_kh=2, merge_kw=2, grid_h=36, grid_w=46)`.
- `self.projector = _build_projector_from_mlp1(moonvit_model.mlp1)` — copies sub-modules directly.

**Tensor shapes through the pipeline:**
- Vision-model out: `(B, L_pre, D) = (1, 1656, 1152)`.
- Patch merger reshape chain: `(B, L_pre, D) -> (B, nh, kh, nw, kw, D) = (1, 18, 2, 23, 2, 1152) -> permute(0, 1, 3, 2, 4, 5) -> view(B, nh*nw, kh*kw*D) = (1, 414, 4608)`.
- Projector in: `(B, L_post, 4608) = (1, 414, 4608)`.
- Projector layers: `LayerNorm(4608) -> Linear(4608, 2048) -> GELU -> Linear(2048, 2048)`.
- Projector out: `(B, L_post, text_hidden) = (1, 414, 2048)`.

**Autocast wrapper (Site 7):**
- `with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):` around the entire forward body.
- RoPE freqs stay fp32 — `apply_rope_real` handles the float-up/down internally.
- `pos_emb_baked` stored fp32, cast on the add.

**Edge cases:**
- `StaticPatchMerger` assumes `grid_h % kh == 0` and `grid_w % kw == 0` — assert at init for 36x46 / 2x2.
- The permute order MUST be `(0, 1, 3, 2, 4, 5)` (nh, nw, kh, kw) not `(0, 2, 1, 3, 4, 5)` (nh, kh, nw, kw). MoonViT canonical uses `(nh, nw, kh, kw)` ordering for the merge; getting this wrong shuffles tokens silently.

**Residual risks (resolve in code):**
- Projector activation: MoonViT canonical may use `gelu` or `gelu_pytorch_tanh`; copy directly from `mlp1` sub-modules rather than reconstructing from config to avoid the activation-string trap.

---

## Appendix B — RoPE Injection Research (research workflow wrfp679ez, 2026-06-03)

### Decision: Path (b) — override Attention.apply_rope

Research read /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/attention.py
+ rotary_embedding.py + interface.py inside the v1.0.0 container. Findings:

#### apply_rope hook EXISTS and is documented for this use case

```python
# attention.py:494-513
def apply_rope(self, q, k, v, position_ids):
    """This method could be overridden in the subclass, in which extra
    functionalities such as q_norm/k_norm could be added."""
    q, k, v = self.split_qkv(q, k, v)
    if not self.rope_fusion:
        self.rotary_emb(position_ids, [q, k])
    return self.convert_qkv(q, k, v)
```

This is the documented subclass override point. Signature gives us individually-addressable q, k, v — the ONLY Python window where they're separate (qkv_proj outputs a fused tensor, kernels see fused QKV again post-convert_qkv).

#### pos_embd_params does NOT support 2D real-RoPE

`PositionalEmbeddingParams` (interface.py:495-511) is a frozen dataclass wrapping
`RopeParams` which takes scalar config only: dim, theta, scale_type, max_positions.
NO `freqs_cos`/`freqs_sin` field. NO inv_freq override. Supported variants are all 1D
or M-RoPE (multimodal 1D w/ position deltas). Path (c) ruled out without invasive
framework surgery.

#### VANILLA backend is mandatory for vision

The default TRTLLM backend has `support_fused_rope=True` (trtllm.py:1269), meaning
the kernel re-applies RoPE internally from pos_embd_params even if apply_rope already
rotated. To avoid double-rotation, we MUST set
`ModelConfig.attn_backend = "VANILLA"` on the vision tower. VANILLA's
`support_fused_rope=False` causes the framework to auto-disable rope_fusion and use
our apply_rope output directly.

Perf cost of VANILLA on vision: minimal (no KV cache, no in-flight batching, fixed
seq_len = 1656). The savings TRTLLM offers (paged FMHA, NVFP4 output) are
LM-decode-loop optimizations that don't apply to a single-shot vision forward.

#### 5 residual risks (validate empirically in implementation)

1. `convert_qkv` re-concat layout (interleaved vs split-by-head) — shape/stride assertion on first forward.
2. VANILLA's `no_kv_cache_forward` uses `flash_attn_varlen_func` which needs `cu_seqlens`
   for variable patch counts under dynamic grid — may need a vision-specific
   AttentionMetadata factory.
3. apply_rope is invoked unconditionally — if MoonViT has registers-only blocks
   that skip RoPE, need a per-layer bypass flag.
4. bf16 autocast around RoPE cos/sin multiply — cast cos/sin to match q/k dtype
   or silent fp32 upcast breaks downstream fused kernels.
5. Parity drift if HF MoonViT applies RoPE at a different point than our hook.

### Revised Phase D estimate: 13 days (was 12)

+1 day for VANILLA backend dispatch verification + bf16-RoPE numerical stability work.

### NO-FALLBACK DISCIPLINE (per user directive 2026-06-03)

Per project memory `feedback_no_fallbacks_or_gates`:

- **NO try/except** around weight loading. Missing/mis-shaped weights MUST raise.
- **NO random-init fallbacks** anywhere — see `project_locateanything_lm_head_root_cause`
  (vendor skipped post_init → random lm_head → mode collapse). Every weight load
  MUST assert trained signature (e.g. std > 0.015 for layer norms, sha256 match
  for re-loadable buffers).
- **NO silent shape coercion** — if shapes mismatch, raise with the actual shape.
- **NO "if backend X unavailable, use Y"** patterns. If VANILLA backend isn't
  selectable for some reason, raise — don't quietly downgrade to a path that
  produces different outputs.

Implementation will use explicit `assert` and `raise RuntimeError` with
descriptive messages at every weight-load and shape-validation boundary.

---

## 7. Risks

1. **TRT-LLM SigLIP plugin signature**: the stock plugin may require `(B, N, D)` token shape and a static `num_positions` derived from `image_size // patch_size`. Our `(L_pre, D)` no-batch contract may force a wrapper that unsqueezes/squeezes around the plugin call, OR a re-export of the plugin with a custom token-count axis.
2. **RoPE freq tensor broadcasting**: `apply_rope_real` does `freqs.unsqueeze(-2)` to broadcast `(L, D_head/2)` against `(L, H, D_head/2)`. SigLIP `CLIPAttention` reshapes to `(B, H, T, D_head)` — head axis is at dim=1, not dim=-2. We must either transpose freqs to match OR reshape q/k to `(B, T, H, D_head)` momentarily, apply RoPE, transpose back. Easy to get wrong; covered by the per-layer parity sub-test.
3. **Activation mismatch**: SigLIP HF uses `gelu_pytorch_tanh` by default; MoonViT canonical uses plain `gelu`. Numerical drift is small but accumulates across N layers — verify which one MoonViT actually trained with and override `config.hidden_act`.
4. **Post-layernorm presence**: HF `SiglipVisionTransformer` always applies `post_layernorm(last_hidden_state)` and the TRT-LLM stock plugin may bake that LN into the graph. Setting it to `Identity` in PT-eager works; on the TRT side we may need to load identity weights (zeros + ones) into the plugin's LN slot OR build with a custom config that skips it.
5. **bf16 vs fp16 mismatch**: our existing `export/vision.py` runs in fp16; MoonViT canonical runs in bf16 via autocast. Mixing — fp16 input, bf16 autocast — can desync from the reference. Pick one (bf16) and update `dtype` defaults in `export_vision` + parity test accordingly.
6. **Weight remap key drift**: MoonViT's encoder layer keys (`vit.encoder.layers.{i}.ln1.*`) vs HF SigLIP's (`vision_model.encoder.layers.{i}.layer_norm1.*`) — any typo in the remap dict silently leaves a layer at random init and parity diff blows up. Mitigated by `strict=True` on a copy of the state_dict with non-projector keys first, then a separate projector load.
7. **`pos_emb_baked` dtype**: stored as the source weight's dtype (fp32 typically). When added to bf16/fp16 token stream, PT will up-cast then down-cast. Numerically fine but slower; consider casting the buffer to bf16 at init. Will surface only as a perf regression, not a correctness one.
8. **Dynamic-grid drift from baked grid**: the MVP runs one fixed `(36, 46)` engine. The first time we re-export at a different aspect ratio, both `pos_emb_baked` AND `freqs_cos/sin` need to be re-baked together — a partial re-bake (e.g. forgot the freqs) produces silently-wrong attention with zero shape error. Add an assertion in `MoonViTOnSiglip.__init__` that grid dims used to bake pos_emb match grid dims used to slice freqs.
