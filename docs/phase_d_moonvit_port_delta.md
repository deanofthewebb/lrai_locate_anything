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

## 7. Risks

1. **TRT-LLM SigLIP plugin signature**: the stock plugin may require `(B, N, D)` token shape and a static `num_positions` derived from `image_size // patch_size`. Our `(L_pre, D)` no-batch contract may force a wrapper that unsqueezes/squeezes around the plugin call, OR a re-export of the plugin with a custom token-count axis.
2. **RoPE freq tensor broadcasting**: `apply_rope_real` does `freqs.unsqueeze(-2)` to broadcast `(L, D_head/2)` against `(L, H, D_head/2)`. SigLIP `CLIPAttention` reshapes to `(B, H, T, D_head)` — head axis is at dim=1, not dim=-2. We must either transpose freqs to match OR reshape q/k to `(B, T, H, D_head)` momentarily, apply RoPE, transpose back. Easy to get wrong; covered by the per-layer parity sub-test.
3. **Activation mismatch**: SigLIP HF uses `gelu_pytorch_tanh` by default; MoonViT canonical uses plain `gelu`. Numerical drift is small but accumulates across N layers — verify which one MoonViT actually trained with and override `config.hidden_act`.
4. **Post-layernorm presence**: HF `SiglipVisionTransformer` always applies `post_layernorm(last_hidden_state)` and the TRT-LLM stock plugin may bake that LN into the graph. Setting it to `Identity` in PT-eager works; on the TRT side we may need to load identity weights (zeros + ones) into the plugin's LN slot OR build with a custom config that skips it.
5. **bf16 vs fp16 mismatch**: our existing `export/vision.py` runs in fp16; MoonViT canonical runs in bf16 via autocast. Mixing — fp16 input, bf16 autocast — can desync from the reference. Pick one (bf16) and update `dtype` defaults in `export_vision` + parity test accordingly.
6. **Weight remap key drift**: MoonViT's encoder layer keys (`vit.encoder.layers.{i}.ln1.*`) vs HF SigLIP's (`vision_model.encoder.layers.{i}.layer_norm1.*`) — any typo in the remap dict silently leaves a layer at random init and parity diff blows up. Mitigated by `strict=True` on a copy of the state_dict with non-projector keys first, then a separate projector load.
7. **`pos_emb_baked` dtype**: stored as the source weight's dtype (fp32 typically). When added to bf16/fp16 token stream, PT will up-cast then down-cast. Numerically fine but slower; consider casting the buffer to bf16 at init. Will surface only as a perf regression, not a correctness one.
8. **Dynamic-grid drift from baked grid**: the MVP runs one fixed `(36, 46)` engine. The first time we re-export at a different aspect ratio, both `pos_emb_baked` AND `freqs_cos/sin` need to be re-baked together — a partial re-bake (e.g. forgot the freqs) produces silently-wrong attention with zero shape error. Add an assertion in `MoonViTOnSiglip.__init__` that grid dims used to bake pos_emb match grid dims used to slice freqs.
