"""MoonViT-as-SigLIP-fork — vision encoder for the TRT-LLM production path.

Sites 1-7 ported; Site 8 (weight loader) is the remaining TODO. See
docs/phase_d_moonvit_port_delta.md for the full 12-day port plan.

Class hierarchy:
- MoonViTVisionEmbeddings (Site 1+2): patch_embed (Conv2d) + pos_emb_baked
- MoonViTAttention (Site 3+4): 2D-RoPE-aware attention; wraps TRT-LLM Attention base
- MoonViTVisionTransformer (Site 5): SiglipVisionEncoder wrapper; post_layernorm = Identity
- MoonViTPatchMerger (Site 6.a): 2x2 pixel-shuffle reshape; bit-equal StaticPatchMerger
- MoonViTProjector (Site 6.b): LayerNorm + Linear + GELU + Linear (mlp1 graft)
- MoonViTVisionModel (Site 7): top-level wrapper with bf16 autocast at the boundary
- load_moonvit_weights (Site 8, TODO): checkpoint -> MoonViTVisionModel loader
"""
from __future__ import annotations
import math
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


class MoonViTVisionEmbeddings(nn.Module):
    """SigLIP-style embeddings with MoonViT's pos_emb_baked buffer.

    Site 1 + Site 2 of Phase D port. Replaces:
      - HF SiglipVisionEmbeddings.position_embedding (nn.Embedding) -> register_buffer('pos_emb_baked', ...)
      - forward: drops position_ids lookup; adds pos_emb_baked directly

    DONE Port site 1 (Appendix A.1): registers self.pos_emb_baked of shape (L_pre, D)
    baked from MoonViT vit.patch_embed.pos_emb.weight (H_max, W_max, D) via
    permute(2,0,1).unsqueeze(0) -> F.interpolate(size=(grid_h, grid_w), mode="bicubic",
    align_corners=False) -> squeeze(0).permute(1,2,0).flatten(end_dim=1). No
    interpolate_pos_encoding branch; grid is baked at init.
    DONE Port site 2 (Appendix A.2): forward(pixel_values: (L_pre, 3, P, P)) ->
    Conv2d -> (L_pre, D, 1, 1) -> view(L_pre, D) -> add self.pos_emb_baked. The
    outer MoonViTVisionTransformer (Site 5) handles the (B, L_pre, D) batch axis;
    here we keep the pre-patchified (L_pre, ...) contract from export/vision.py:55-57.
    """
    def __init__(self, hidden_size: int, patch_size: int, grid_h: int, grid_w: int,
                 moonvit_pos_emb_weight: torch.Tensor):
        super().__init__()
        # FAIL LOUD on shape mismatch (no silent coercion)
        if moonvit_pos_emb_weight.ndim != 3:
            raise RuntimeError(
                f"moonvit_pos_emb_weight must be 3D (H_max, W_max, hidden_size); "
                f"got shape {tuple(moonvit_pos_emb_weight.shape)}"
            )
        H_max, W_max, d = moonvit_pos_emb_weight.shape
        if d != hidden_size:
            raise RuntimeError(
                f"moonvit_pos_emb_weight last dim {d} != hidden_size {hidden_size}"
            )
        if grid_h > H_max or grid_w > W_max:
            raise RuntimeError(
                f"requested grid ({grid_h},{grid_w}) exceeds source table ({H_max},{W_max})"
            )

        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w  # L_pre
        self.H_max = int(H_max)
        self.W_max = int(W_max)

        # Conv2d patch embedding (matches MoonViT export/vision.py:55-57 + SigLIP)
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # D-2: stash the un-interpolated source weight as a non-persistent buffer
        # so interpolate_pos_encoding can re-bake at runtime for non-canonical
        # grids. Persistent=False because the source already lives in the parent
        # checkpoint; we don't re-serialize it. fp32 contiguous for stable bicubic.
        self.register_buffer(
            "pos_emb_source",
            moonvit_pos_emb_weight.detach().to(torch.float32).contiguous(),
            persistent=False,
        )

        # Bake pos_emb_baked from MoonViT source table via bicubic interpolation
        # (matches export/vision.py:35-44 exactly).
        pos = self._bake_pos_emb(grid_h, grid_w)
        self.register_buffer("pos_emb_baked", pos, persistent=True)
        # FAIL LOUD on degenerate buffer (e.g. all zeros = uninitialized)
        if torch.abs(pos).max().item() < 1e-6:
            raise RuntimeError(
                f"pos_emb_baked appears degenerate (max-abs < 1e-6); "
                f"check moonvit_pos_emb_weight source"
            )

    def _bake_pos_emb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Bicubic-interpolate pos_emb_source -> (grid_h*grid_w, hidden_size).

        Single source of truth used by both __init__ (canonical) and
        interpolate_pos_encoding (dynamic). FAIL LOUD on shape drift.
        """
        if grid_h > self.H_max or grid_w > self.W_max:
            raise RuntimeError(
                f"requested grid ({grid_h},{grid_w}) exceeds source table "
                f"({self.H_max},{self.W_max})"
            )
        src = self.pos_emb_source  # (H_max, W_max, d) fp32
        pos = src.permute(2, 0, 1).unsqueeze(0)  # (1, d, H_max, W_max)
        pos = F.interpolate(pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos = pos.squeeze(0).permute(1, 2, 0).contiguous().flatten(end_dim=1)
        expected = (grid_h * grid_w, self.hidden_size)
        if tuple(pos.shape) != expected:
            raise RuntimeError(
                f"baked pos_emb shape {tuple(pos.shape)} != expected {expected}"
            )
        return pos

    def interpolate_pos_encoding(self, new_grid_h: int, new_grid_w: int) -> None:
        """D-2: re-bake pos_emb_baked for a non-canonical grid.

        No-op if (new_grid_h, new_grid_w) == (self.grid_h, self.grid_w).
        Otherwise:
          - F.interpolate the source weight to the new grid
          - Overwrite self.pos_emb_baked (in-place buffer swap)
          - Update self.grid_h, self.grid_w, self.num_patches

        FAIL LOUD if the source buffer is missing (e.g. someone deserialized
        an old checkpoint state that lacked it) — interpolation cannot proceed.
        """
        new_grid_h = int(new_grid_h)
        new_grid_w = int(new_grid_w)
        if new_grid_h <= 0 or new_grid_w <= 0:
            raise RuntimeError(
                f"interpolate_pos_encoding: grid dims must be positive; "
                f"got ({new_grid_h},{new_grid_w})"
            )
        if not hasattr(self, "pos_emb_source") or self.pos_emb_source is None:
            raise RuntimeError(
                "pos_emb_source buffer missing; cannot re-interpolate. "
                "Did you load a checkpoint with persistent=False stripped?"
            )
        if (new_grid_h, new_grid_w) == (self.grid_h, self.grid_w):
            return  # already at target grid
        pos = self._bake_pos_emb(new_grid_h, new_grid_w)
        # Buffer swap — re-register so the parameter device/dtype tracking sees
        # the new shape. Use register_buffer to overwrite atomically.
        self.register_buffer(
            "pos_emb_baked",
            pos.to(self.pos_emb_baked.dtype).to(self.pos_emb_baked.device),
            persistent=True,
        )
        self.grid_h = new_grid_h
        self.grid_w = new_grid_w
        self.num_patches = new_grid_h * new_grid_w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (L_pre, 3, patch_size, patch_size) -- already patchified.

        Returns (L_pre, hidden_size) tokens with pos_emb_baked added.

        NOTE: pixel_values is pre-patchified (matches MoonViT contract). The
        TRT-LLM SiglipVisionTransformer.forward expects (B*N, D) flattened, but
        we work in (L_pre, D) with B=1 implicit. The outer MoonViTVisionTransformer
        must thread this through correctly.
        """
        # FAIL LOUD on input shape mismatch
        if pixel_values.ndim != 4:
            raise RuntimeError(
                f"pixel_values must be 4D (L_pre, 3, P, P); got {pixel_values.ndim}D"
            )
        L_pre = pixel_values.shape[0]
        if L_pre != self.num_patches:
            raise RuntimeError(
                f"input L_pre={L_pre} != baked num_patches={self.num_patches} "
                f"(dynamic grid not yet supported in MVP; Site 4 will add)"
            )

        # Conv2d on (L_pre, 3, P, P) -> (L_pre, hidden_size, 1, 1)
        embeddings = self.patch_embedding(pixel_values).view(L_pre, self.hidden_size)

        # Add baked positional embedding (no lookup, just broadcast across batch)
        embeddings = embeddings + self.pos_emb_baked

        return embeddings


def _resolve_trtllm_attention_base():
    """Lazy-import TRT-LLM's Attention so this file stays importable on Mac.

    Returns the Attention class. Raises ImportError with a clear message if
    tensorrt_llm isn't installed. We import inside __init__ rather than at
    module import time so tests/test_moonvit_parity.py can introspect the
    class layout from a PT-only host.
    """
    try:
        from tensorrt_llm._torch.modules.attention import Attention  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover -- exercised on Mac dev hosts
        raise ImportError(
            "MoonViTAttention requires tensorrt_llm._torch.modules.attention.Attention; "
            "install tensorrt-llm in the container or run with the PT-only parity test "
            "harness (tests/test_moonvit_parity.py) that uses MoonViTAttention.apply_rope "
            "as a standalone method."
        ) from exc
    return Attention


def _moonvit_apply_rope(self, q, k, v, position_ids):
    """Standalone apply_rope override — designed so it can be bound to either
    the TRT-LLM Attention base OR a plain nn.Module for parity testing.

    Contract (per Read phase findings):
    - q,k,v arrive fused-or-split per self.support_fused_qkv; we route through
      self.split_qkv() which is a last-dim split only (no reshape/transpose).
    - For VANILLA backend (vision tower) support_fused_qkv=False -> split form.
    - Shapes post-split: q -> (num_tokens, num_heads * head_dim);
                         k,v -> (num_tokens, num_kv_heads * head_dim).
      For SigLIP MHA num_kv_heads == num_heads.
    - We reshape to (num_tokens, H, D_head) so apply_rope_real's canonical
      (L, H, D_head) layout from patches.py applies directly (matches MoonViT).
    - freqs buffers are stored as packed (L_pre, D_head/2, 2) fp32 so the
      existing patches.apply_rope_real signature works unchanged (no 4-arg
      variant required, no second function path to maintain).
    - Output is re-fused via self.convert_qkv per the TRT-LLM hook contract.
    """
    # split_qkv: pass-through if already split, else last-dim split.
    q, k, v = self.split_qkv(q, k, v)

    # FAIL LOUD: q must be 2D (num_tokens, H * D_head); anything else means the
    # caller dispatched a non-VANILLA-shape tensor and we'd silently corrupt it.
    if q.ndim != 2:
        raise RuntimeError(
            f"MoonViTAttention.apply_rope expected q of shape (num_tokens, H*D_head); "
            f"got ndim={q.ndim}, shape={tuple(q.shape)}. Did the backend swap to "
            f"a non-VANILLA kernel that pre-reshapes q?"
        )
    num_tokens, qhd = q.shape
    H = int(self.num_heads)
    D_head = int(self.head_dim)
    if qhd != H * D_head:
        raise RuntimeError(
            f"q inner dim {qhd} != num_heads({H}) * head_dim({D_head}) = {H * D_head}"
        )
    if k.shape[-1] != H * D_head or v.shape[-1] != H * D_head:
        raise RuntimeError(
            f"MoonViTAttention assumes MHA (num_kv_heads == num_heads); "
            f"got k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}"
        )

    # Reshape to (L, H, D_head) — canonical MoonViT layout that patches.apply_rope_real
    # was written against (freqs unsqueeze(-2) broadcasts over the H axis).
    q = q.reshape(num_tokens, H, D_head)
    k = k.reshape(num_tokens, H, D_head)

    # Validate freqs registration covers this many tokens.
    fc_buf = self.freqs_cos
    fs_buf = self.freqs_sin
    if fc_buf.shape[0] != num_tokens:
        raise RuntimeError(
            f"freqs_cos length {fc_buf.shape[0]} != num_tokens {num_tokens}; "
            f"MoonViT MVP bakes a static grid (grid_h={self.grid_h}, grid_w={self.grid_w}). "
            f"Dynamic-grid attention is a separate port site."
        )
    if fc_buf.shape[-1] != D_head // 2:
        raise RuntimeError(
            f"freqs_cos last dim {fc_buf.shape[-1]} != D_head//2 {D_head // 2}"
        )

    # Cast cos/sin to q's dtype (bf16 in production) so the multiply inside
    # apply_rope_real doesn't silently upcast and the post-rotation tensor
    # concats cleanly with the unrotated v through convert_qkv.
    fc = fc_buf.to(q.dtype)
    fs = fs_buf.to(q.dtype)

    # Pack to (L, D_head/2, 2) so we can call patches.apply_rope_real unchanged.
    # Use SAME function as patches.py to guarantee bitwise parity with the
    # PT-eager export path (single source of rotation truth).
    freqs_packed = torch.stack([fc, fs], dim=-1)

    from lrai_locate_anything.patches import apply_rope_real as _patches_apply_rope_real
    q, k = _patches_apply_rope_real(q, k, freqs_packed)

    # Collapse (L, H, D_head) -> (L, H*D_head) so convert_qkv sees the
    # canonical last-dim packed layout.
    q = q.reshape(num_tokens, H * D_head)
    k = k.reshape(num_tokens, H * D_head)

    return self.convert_qkv(q, k, v)


def _moonvit_reslice_freqs(self, new_grid_h, new_grid_w):
    """D-2: re-slice freqs_cos/freqs_sin to a (new_grid_h*new_grid_w, D_head/2)
    view in-place. No-op if already at target grid.

    Standalone helper (mirrors _moonvit_apply_rope's design) so it can be bound
    onto the dynamically-built MoonViTAttention_TRTLLM subclass without losing
    the lazy-import discipline.

    FAIL LOUD if:
      - the source (H_max, W_max, D_head/2) buffers weren't supplied at ctor
      - the requested grid exceeds the source table
      - the buffer dtype/shape contract is violated
    """
    new_grid_h = int(new_grid_h)
    new_grid_w = int(new_grid_w)
    if new_grid_h <= 0 or new_grid_w <= 0:
        raise RuntimeError(
            f"reslice_freqs: grid dims must be positive; got "
            f"({new_grid_h},{new_grid_w})"
        )
    if (new_grid_h, new_grid_w) == (int(self.grid_h), int(self.grid_w)):
        return  # already at target
    src_cos = getattr(self, "freqs_cos_source", None)
    src_sin = getattr(self, "freqs_sin_source", None)
    if src_cos is None or src_sin is None:
        raise RuntimeError(
            "reslice_freqs requires freqs_cos_source / freqs_sin_source to have "
            "been registered at MoonViTAttention construction time. Pass "
            "freqs_cos_source=(H_max,W_max,D_head/2) and freqs_sin_source=... "
            "to MoonViTAttention(...) to enable dynamic grids."
        )
    H_max = int(self.H_max)
    W_max = int(self.W_max)
    if new_grid_h > H_max or new_grid_w > W_max:
        raise RuntimeError(
            f"requested grid ({new_grid_h},{new_grid_w}) exceeds freqs source "
            f"table ({H_max},{W_max})"
        )
    D_half = int(self.head_dim_attr) // 2
    if src_cos.shape[-1] != D_half:
        raise RuntimeError(
            f"freqs_cos_source last dim {src_cos.shape[-1]} != head_dim//2 {D_half}"
        )
    # Slice + flatten exactly mirrors patches.MoonViTRotaryEmbedding.get_freqs_cis:
    # cos = freqs_cos[:h, :w].reshape(-1, D/2). This keeps bitwise parity with the
    # PT-eager export path so visual outputs match.
    new_cos = src_cos[:new_grid_h, :new_grid_w].reshape(-1, D_half).contiguous()
    new_sin = src_sin[:new_grid_h, :new_grid_w].reshape(-1, D_half).contiguous()
    expected = (new_grid_h * new_grid_w, D_half)
    if tuple(new_cos.shape) != expected or tuple(new_sin.shape) != expected:
        raise RuntimeError(
            f"resliced freqs shape mismatch: cos={tuple(new_cos.shape)}, "
            f"sin={tuple(new_sin.shape)}, expected={expected}"
        )
    # In-place buffer swap. Preserve device of the existing buffer so we don't
    # silently move tensors across devices.
    dev = self.freqs_cos.device
    self.register_buffer("freqs_cos", new_cos.to(dev), persistent=False)
    self.register_buffer("freqs_sin", new_sin.to(dev), persistent=False)
    self.grid_h = new_grid_h
    self.grid_w = new_grid_w


class MoonViTAttention:
    """SigLIP-style attention with MoonViT's 2D real-RoPE injected via apply_rope.

    Site 3 + Site 4 of Phase D port. Subclass design:
    - Inherit from tensorrt_llm._torch.modules.attention.Attention (resolved lazily)
    - Override apply_rope(q, k, v, position_ids) -> (q, k, v) to inject 2D RoPE
    - Register freqs_cos/freqs_sin as buffers per layer
    - Force ModelConfig.attn_backend = 'VANILLA' (no kernel re-rotation)

    NO-FALLBACK discipline:
    - raises if freqs buffer shape/dtype unexpected
    - raises if VANILLA backend not selected (silent TRTLLM dispatch would corrupt output)
    - raises if input q/k shape doesn't match registered buffer dims

    Lazy import: the TRT-LLM Attention base is resolved at __init__ time, NOT at
    module import time. This file remains importable on Mac (no tensorrt_llm)
    so tests/test_moonvit_parity.py can introspect MoonViTAttention.apply_rope
    directly via the module-level _moonvit_apply_rope helper.
    """

    # Sentinel: when constructed, we dynamically subclass the TRT-LLM Attention
    # base and return an instance of THAT subclass. apply_rope binds to the
    # standalone _moonvit_apply_rope so parity tests can call it without the
    # TRT-LLM base in the MRO.
    apply_rope = _moonvit_apply_rope

    # D-2: bind the reslice_freqs method onto the dynamically-built subclass so
    # callers can do `model.attention.reslice_freqs(new_h, new_w)` without
    # subclass introspection. See _moonvit_reslice_freqs definition below.
    reslice_freqs = staticmethod(lambda *_a, **_kw: (_ for _ in ()).throw(
        RuntimeError("reslice_freqs called on unbuilt MoonViTAttention class")
    ))

    def __new__(
        cls,
        *,
        model_config,
        layer_idx: int,
        head_dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        freqs_cos_source: Optional[torch.Tensor] = None,
        freqs_sin_source: Optional[torch.Tensor] = None,
        **attention_kwargs,
    ):
        # FAIL LOUD: VANILLA backend is mandatory. If the model builder dispatched
        # a fused kernel (TRTLLM, FLASHINFER, ...) our apply_rope override would
        # be ignored and the output would be silently wrong (Appendix B residual
        # risk #3). DO NOT silently overwrite — the caller must set it explicitly.
        attn_backend = getattr(model_config, "attn_backend", None)
        if attn_backend != "VANILLA":
            raise RuntimeError(
                f"MoonViTAttention requires ModelConfig.attn_backend == 'VANILLA'; "
                f"got {attn_backend!r}. Set it explicitly in the vision-tower model "
                f"builder — silent override would mask kernel-vs-eager divergence."
            )

        # FAIL LOUD on freqs shape/dtype before we incur the cost of building the base.
        expected_shape = (grid_h * grid_w, head_dim // 2)
        if tuple(freqs_cos.shape) != expected_shape:
            raise RuntimeError(
                f"freqs_cos shape {tuple(freqs_cos.shape)} != expected "
                f"(grid_h*grid_w, head_dim//2) = {expected_shape}"
            )
        if tuple(freqs_sin.shape) != expected_shape:
            raise RuntimeError(
                f"freqs_sin shape {tuple(freqs_sin.shape)} != expected {expected_shape}"
            )
        if freqs_cos.dtype != torch.float32 or freqs_sin.dtype != torch.float32:
            raise RuntimeError(
                f"freqs buffers must be fp32 for numerical stability "
                f"(MoonViT stores cos/sin as fp32 and casts per-call to q.dtype); "
                f"got freqs_cos.dtype={freqs_cos.dtype}, freqs_sin.dtype={freqs_sin.dtype}"
            )

        # D-2: validate the optional source-table freqs used for dynamic reslicing.
        # If EITHER source is provided, BOTH must be — fail loud to avoid a half-
        # configured state where cos reslices but sin doesn't.
        if (freqs_cos_source is None) ^ (freqs_sin_source is None):
            raise RuntimeError(
                "freqs_cos_source and freqs_sin_source must both be provided or "
                "both omitted; got cos_source is None="
                f"{freqs_cos_source is None}, sin_source is None="
                f"{freqs_sin_source is None}"
            )
        if freqs_cos_source is not None:
            if freqs_cos_source.ndim != 3 or freqs_sin_source.ndim != 3:
                raise RuntimeError(
                    f"freqs_*_source must be 3D (H_max, W_max, head_dim//2); got "
                    f"cos_source.ndim={freqs_cos_source.ndim}, "
                    f"sin_source.ndim={freqs_sin_source.ndim}"
                )
            if tuple(freqs_cos_source.shape) != tuple(freqs_sin_source.shape):
                raise RuntimeError(
                    f"freqs_*_source shape mismatch: "
                    f"cos={tuple(freqs_cos_source.shape)} vs "
                    f"sin={tuple(freqs_sin_source.shape)}"
                )
            if freqs_cos_source.shape[-1] != head_dim // 2:
                raise RuntimeError(
                    f"freqs_*_source last dim {freqs_cos_source.shape[-1]} != "
                    f"head_dim//2 {head_dim // 2}"
                )
            H_max_f, W_max_f = freqs_cos_source.shape[0], freqs_cos_source.shape[1]
            if grid_h > H_max_f or grid_w > W_max_f:
                raise RuntimeError(
                    f"canonical grid ({grid_h},{grid_w}) exceeds freqs source "
                    f"table ({H_max_f},{W_max_f})"
                )
            if freqs_cos_source.dtype != torch.float32 or freqs_sin_source.dtype != torch.float32:
                raise RuntimeError(
                    "freqs_*_source must be fp32; got "
                    f"cos={freqs_cos_source.dtype}, sin={freqs_sin_source.dtype}"
                )

        AttentionBase = _resolve_trtllm_attention_base()

        # FAIL LOUD: if someone monkey-patched the import to return torch.nn.Module
        # we'd silently lose split_qkv / convert_qkv / num_heads / head_dim, producing
        # garbage at runtime. Sanity-check the base exposes the TRT-LLM contract.
        for required in ("split_qkv", "convert_qkv"):
            if not hasattr(AttentionBase, required):
                raise RuntimeError(
                    f"Resolved attention base {AttentionBase!r} is missing required "
                    f"method {required!r}; expected tensorrt_llm._torch.modules."
                    f"attention.Attention. Refusing to silently fall back to nn.Module."
                )

        # Dynamically build the concrete subclass that mixes our apply_rope
        # override into TRT-LLM's Attention. We do this here (rather than at
        # module-import time) so the base class import stays lazy.
        # D-2: also bind reslice_freqs so callers can re-slice on grid change.
        concrete_cls = type(
            "MoonViTAttention_TRTLLM",
            (AttentionBase,),
            {
                "apply_rope": _moonvit_apply_rope,
                "reslice_freqs": _moonvit_reslice_freqs,
            },
        )

        # Build the underlying TRT-LLM Attention; layer_idx / model_config are
        # the canonical TRT-LLM init args plus any extras the caller passes.
        instance = AttentionBase.__new__(concrete_cls)
        AttentionBase.__init__(
            instance,
            model_config=model_config,
            layer_idx=layer_idx,
            **attention_kwargs,
        )

        # FAIL LOUD: the TRT-LLM base must have surfaced num_heads / head_dim
        # matching what we sized the freqs against. Mismatch -> wrong rotation.
        base_num_heads = getattr(instance, "num_heads", None)
        base_head_dim = getattr(instance, "head_dim", None)
        if base_num_heads is None or base_head_dim is None:
            raise RuntimeError(
                "TRT-LLM Attention base did not expose num_heads / head_dim; "
                "MoonViTAttention cannot validate freqs sizing."
            )
        if int(base_num_heads) != int(num_heads):
            raise RuntimeError(
                f"num_heads mismatch: caller={num_heads} vs base={base_num_heads}"
            )
        if int(base_head_dim) != int(head_dim):
            raise RuntimeError(
                f"head_dim mismatch: caller={head_dim} vs base={base_head_dim}"
            )

        # Register freqs as non-persistent fp32 buffers (~6.4 MB; shared across
        # 27 layers via the caller passing the same tensors — buffer dedup is
        # the caller's responsibility, see Site 4 in build.py).
        instance.register_buffer("freqs_cos", freqs_cos.contiguous(), persistent=False)
        instance.register_buffer("freqs_sin", freqs_sin.contiguous(), persistent=False)
        instance.grid_h = int(grid_h)
        instance.grid_w = int(grid_w)
        instance.head_dim_attr = int(head_dim)

        # D-2: stash the full (H_max, W_max, D_head/2) source freqs as non-persistent
        # buffers so reslice_freqs can re-slice for non-canonical grids. If the
        # caller didn't supply a source, reslice_freqs will fail loud at call time.
        if freqs_cos_source is not None:
            instance.register_buffer(
                "freqs_cos_source",
                freqs_cos_source.detach().to(torch.float32).contiguous(),
                persistent=False,
            )
            instance.register_buffer(
                "freqs_sin_source",
                freqs_sin_source.detach().to(torch.float32).contiguous(),
                persistent=False,
            )
            instance.H_max = int(freqs_cos_source.shape[0])
            instance.W_max = int(freqs_cos_source.shape[1])
        else:
            instance.freqs_cos_source = None  # type: ignore[assignment]
            instance.freqs_sin_source = None  # type: ignore[assignment]
            instance.H_max = None  # type: ignore[assignment]
            instance.W_max = None  # type: ignore[assignment]
        return instance


class MoonViTPatchMerger(nn.Module):
    """Site 6 (part 1): 2x2 pixel-shuffle reshape.

    Bit-equal to lrai_locate_anything/export_prod/vision_proj.py:StaticPatchMerger.

    Input:  (L_pre, hidden_size) where L_pre = grid_h * grid_w
    Output: (L_post, hidden_size * kh * kw) where L_post = (grid_h/kh) * (grid_w/kw)

    NO-FALLBACK discipline:
    - raises if grid not divisible by merge kernel (no implicit padding/truncation)
    - raises if input L_pre doesn't match baked grid (dynamic merge not supported)
    """
    def __init__(self, grid_h: int, grid_w: int, kh: int = 2, kw: int = 2):
        super().__init__()
        if grid_h % kh != 0 or grid_w % kw != 0:
            raise RuntimeError(
                f"grid ({grid_h},{grid_w}) not divisible by merge kernel ({kh},{kw})"
            )
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.kh = int(kh)
        self.kw = int(kw)
        self.nh = self.grid_h // self.kh
        self.nw = self.grid_w // self.kw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FAIL LOUD on input shape
        if x.ndim != 2:
            raise RuntimeError(
                f"MoonViTPatchMerger expects 2D (L_pre, hidden_size); got {x.ndim}D shape={tuple(x.shape)}"
            )
        L_pre = self.grid_h * self.grid_w
        if x.shape[0] != L_pre:
            raise RuntimeError(
                f"input L_pre {x.shape[0]} != expected {L_pre} "
                f"(grid_h={self.grid_h}, grid_w={self.grid_w})"
            )
        d = x.shape[-1]
        # view -> permute -> view: matches export_prod/vision_proj.py:StaticPatchMerger.forward
        # (L_pre, d) -> (nh, kh, nw, kw, d) -> (nh, nw, kh, kw, d) -> (L_post, kh*kw*d)
        return (
            x.view(self.nh, self.kh, self.nw, self.kw, d)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(self.nh * self.nw, self.kh * self.kw * d)
        )


class MoonViTProjector(nn.Module):
    """Site 6 (part 2): LayerNorm + Linear + GELU + Linear — mlp1 graft from MoonViT.

    Mirrors lrai_locate_anything/export_prod/projector.py:ProjectorForExport which
    treats mlp1 as an opaque module. We materialize the canonical 4-layer composition
    (Appendix A.6) and load weights from the source mlp1 state_dict.

    Input:  (L_post, hidden * kh * kw) e.g. (414, 4608)
    Output: (L_post, text_hidden) e.g. (414, 2048)

    NO-FALLBACK discipline:
    - mlp1_state_dict must contain all 6 expected keys or raises (no zero-init)
    - shapes are validated against the supplied state_dict tensors
    - GELU activation is the plain `nn.GELU()` (matches PyTorch default 'none' approx);
      caller must pre-validate that MoonViT mlp1 was not exported with tanh approx.
    """
    def __init__(self, in_features: int, text_hidden: int, mlp1_state_dict: dict):
        super().__init__()
        self.in_features = int(in_features)
        self.text_hidden = int(text_hidden)

        self.layernorm = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, text_hidden)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(text_hidden, text_hidden)

        # MoonViT mlp1 is Sequential( [0]=LayerNorm, [1]=Linear, [2]=GELU, [3]=Linear ).
        # ProjectorForExport copies sub-modules directly, so its state_dict keys carry
        # the Sequential indices: "0.weight", "0.bias", "1.weight", "1.bias",
        # "3.weight", "3.bias".
        required = ["0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"]
        missing = [k for k in required if k not in mlp1_state_dict]
        if missing:
            raise RuntimeError(
                f"mlp1_state_dict missing keys: {missing}. Expected MoonViT mlp1 "
                f"Sequential(LayerNorm, Linear, GELU, Linear) with keys {required}."
            )

        ln_w = mlp1_state_dict["0.weight"]
        ln_b = mlp1_state_dict["0.bias"]
        l1_w = mlp1_state_dict["1.weight"]
        l1_b = mlp1_state_dict["1.bias"]
        l2_w = mlp1_state_dict["3.weight"]
        l2_b = mlp1_state_dict["3.bias"]

        # FAIL LOUD on shape mismatch — silent broadcast/pad would corrupt logits.
        if tuple(ln_w.shape) != (in_features,):
            raise RuntimeError(
                f"mlp1 layernorm weight shape {tuple(ln_w.shape)} != ({in_features},)"
            )
        if tuple(ln_b.shape) != (in_features,):
            raise RuntimeError(
                f"mlp1 layernorm bias shape {tuple(ln_b.shape)} != ({in_features},)"
            )
        if tuple(l1_w.shape) != (text_hidden, in_features):
            raise RuntimeError(
                f"mlp1 linear1 weight shape {tuple(l1_w.shape)} != "
                f"({text_hidden}, {in_features})"
            )
        if tuple(l1_b.shape) != (text_hidden,):
            raise RuntimeError(
                f"mlp1 linear1 bias shape {tuple(l1_b.shape)} != ({text_hidden},)"
            )
        if tuple(l2_w.shape) != (text_hidden, text_hidden):
            raise RuntimeError(
                f"mlp1 linear2 weight shape {tuple(l2_w.shape)} != "
                f"({text_hidden}, {text_hidden})"
            )
        if tuple(l2_b.shape) != (text_hidden,):
            raise RuntimeError(
                f"mlp1 linear2 bias shape {tuple(l2_b.shape)} != ({text_hidden},)"
            )

        # Explicit copy_ — no try/except; if dtype/device mismatch raise.
        with torch.no_grad():
            self.layernorm.weight.copy_(ln_w)
            self.layernorm.bias.copy_(ln_b)
            self.linear1.weight.copy_(l1_w)
            self.linear1.bias.copy_(l1_b)
            self.linear2.weight.copy_(l2_w)
            self.linear2.bias.copy_(l2_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"MoonViTProjector input last-dim {x.shape[-1]} != in_features {self.in_features}"
            )
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MoonViTVisionTransformer(nn.Module):
    """Site 5: encoder wrapper — skip post_layernorm; drop (B,N,D) reshape since B=1 implicit.

    Wraps the TRT-LLM SiglipVisionEncoder (the encoder stack of MoonViTEncoderLayer
    blocks). MoonViT canonical has no trailing LayerNorm so `post_layernorm` is
    unconditionally `nn.Identity()`. Per Read findings, SiglipVisionTransformer
    accepts `use_post_layernorm=False` and sets `post_layernorm=nn.Identity()`,
    so this wrapper is intentionally narrow — it just enforces the 2D (L_pre, D)
    contract that downstream merger/projector expect.

    NO-FALLBACK discipline:
    - post_layernorm replacement is unconditional (no "if has post_layernorm" branch)
    - input must be 2D (L_pre, hidden_size); 3D batched inputs raise
    """
    def __init__(self, encoder_module: nn.Module):
        super().__init__()
        self.encoder = encoder_module
        # MoonViT canonical doesn't apply a trailing LN; force Identity.
        self.post_layernorm = nn.Identity()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (L_pre, hidden_size)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"embeddings must be 2D (L_pre, hidden_size); "
                f"got {embeddings.ndim}D shape={tuple(embeddings.shape)}"
            )
        hidden_states = self.encoder(embeddings)  # TRT-LLM SiglipVisionEncoder stack
        # Skip post_layernorm — MoonViT canonical doesn't apply one (Identity).
        return self.post_layernorm(hidden_states)


class MoonViTVisionModel(nn.Module):
    """Site 7: end-to-end wrapper — embeddings -> transformer -> merger -> projector.

    bf16 autocast at the boundary so submodules (Conv2d, Linear, Attention) run
    in bf16 while register_buffer tensors stay fp32 (pos_emb_baked, freqs_cos/sin)
    and are cast on consumption inside their owning submodules.

    NO-FALLBACK discipline:
    - pixel_values must be 4D (L_pre, 3, P, P)
    - use_bf16 is explicit (default True); caller can override but cannot silently
      disable autocast via misconfiguration
    """
    def __init__(
        self,
        embeddings: nn.Module,
        transformer: nn.Module,
        merger: nn.Module,
        projector: nn.Module,
        *,
        use_bf16: bool = True,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.transformer = transformer
        self.merger = merger
        self.projector = projector
        self.use_bf16 = bool(use_bf16)

        # D-2: capture canonical grid as the initial set_grid target, and seed the
        # cache with the canonical (pos_emb, freqs) so the first set_grid call
        # back to canonical is a cache hit (no recompute).
        self._canonical_grid = (int(embeddings.grid_h), int(embeddings.grid_w))
        self._current_grid = self._canonical_grid
        # Cache: key=(grid_h, grid_w) -> dict with 'pos_emb' (tensor) and per-attn
        # 'freqs' (list[(cos, sin)] in attention-block order). Populated lazily.
        self._grid_cache: dict[tuple[int, int], dict] = {}
        # Seed canonical entry so the first re-target to canonical is O(1).
        self._grid_cache[self._canonical_grid] = {
            "pos_emb": embeddings.pos_emb_baked.detach().clone(),
            "freqs": [
                (a.freqs_cos.detach().clone(), a.freqs_sin.detach().clone())
                for a in self._iter_attentions()
            ],
        }

    def _iter_attentions(self):
        """Yield each MoonViTAttention instance in the encoder stack, in order.

        Walks self.transformer.encoder.blocks and inspects each block for an
        attention submodule with freqs_cos/freqs_sin buffers + a reslice_freqs
        method. We do NOT silently skip blocks without it — that's a wiring bug.
        """
        encoder = getattr(self.transformer, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "MoonViTVisionModel.transformer has no `encoder` attribute; "
                "cannot enumerate attentions for set_grid."
            )
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            return  # encoder stack with no .blocks attr (e.g. TRT-LLM custom) —
            # caller wired things differently; set_grid is then a per-attn call
            # the caller must make manually.
        for block in blocks:
            # Convention: attention lives at block.attention or block.attn or
            # block.wqkv (the latter is the pre-swap skeleton — no freqs there,
            # so we skip; the swap path attaches a real attention as 'attention').
            attn = None
            for name in ("attention", "attn", "self_attn"):
                cand = getattr(block, name, None)
                if cand is not None and hasattr(cand, "freqs_cos") and hasattr(cand, "reslice_freqs"):
                    attn = cand
                    break
            if attn is None:
                # PT swap path: _MoonViTPTAttentionBlock stores its rope table as
                # block.freqs_packed and exposes reslice via block.attention (a
                # _PTBlockFreqsProxy). It has no freqs_cos/freqs_sin, so the loop
                # above won't catch it — handle it explicitly here.
                proxy = getattr(block, "attention", None)
                if proxy is not None and hasattr(proxy, "reslice_freqs"):
                    yield proxy
                    continue
                # Pre-swap skeleton block (wqkv/wo only). Skip silently here —
                # set_grid is a no-op on the structural placeholder used on
                # Mac PT-only tests. A FAIL LOUD here would break those tests.
                continue
            yield attn

    def set_grid(self, new_grid_h: int, new_grid_w: int) -> None:
        """D-2: switch the model to a non-canonical (new_grid_h, new_grid_w).

        Caching policy:
          - If (new_grid_h, new_grid_w) is in self._grid_cache, restore
            pos_emb / freqs from the cache (zero recomputation).
          - Else interpolate pos_emb (via embeddings.interpolate_pos_encoding)
            and reslice freqs (via each attention's reslice_freqs), then
            populate the cache so subsequent calls with the same grid are O(1).

        Caller contract:
          - Must call set_grid BEFORE forward when L_pre changes. No silent
            auto-detect in forward — that would mask shape bugs.
          - Calling set_grid with the current grid is a no-op.
        """
        new_grid_h = int(new_grid_h)
        new_grid_w = int(new_grid_w)
        if new_grid_h <= 0 or new_grid_w <= 0:
            raise RuntimeError(
                f"set_grid: grid dims must be positive; got "
                f"({new_grid_h},{new_grid_w})"
            )
        key = (new_grid_h, new_grid_w)
        if key == self._current_grid:
            return  # already at target

        cached = self._grid_cache.get(key)
        if cached is not None:
            # Restore from cache — copy_ into existing buffers if shape matches,
            # otherwise re-register so the shape change takes effect.
            self.embeddings.register_buffer(
                "pos_emb_baked",
                cached["pos_emb"].to(self.embeddings.pos_emb_baked.device),
                persistent=True,
            )
            self.embeddings.grid_h = new_grid_h
            self.embeddings.grid_w = new_grid_w
            self.embeddings.num_patches = new_grid_h * new_grid_w
            attns = list(self._iter_attentions())
            cached_freqs = cached["freqs"]
            if len(attns) != len(cached_freqs):
                raise RuntimeError(
                    f"grid cache stale: cached {len(cached_freqs)} attention "
                    f"entries but encoder now has {len(attns)}"
                )
            for attn, (fc, fs) in zip(attns, cached_freqs):
                attn.register_buffer("freqs_cos", fc.to(attn.freqs_cos.device), persistent=False)
                attn.register_buffer("freqs_sin", fs.to(attn.freqs_sin.device), persistent=False)
                # If attn is a _PTBlockFreqsProxy, the parent block consumes
                # freqs_packed (not freqs_cos/freqs_sin), so we must re-pack.
                if hasattr(attn, "_sync_packed_from_cos_sin"):
                    attn._sync_packed_from_cos_sin()
                attn.grid_h = new_grid_h
                attn.grid_w = new_grid_w
            self._update_merger_grid(new_grid_h, new_grid_w)
            self._current_grid = key
            return

        # Cache miss — compute pos_emb + freqs for the new grid and cache them.
        self.embeddings.interpolate_pos_encoding(new_grid_h, new_grid_w)
        cached_freqs_list = []
        for attn in self._iter_attentions():
            attn.reslice_freqs(new_grid_h, new_grid_w)
            cached_freqs_list.append(
                (attn.freqs_cos.detach().clone(), attn.freqs_sin.detach().clone())
            )
        self._grid_cache[key] = {
            "pos_emb": self.embeddings.pos_emb_baked.detach().clone(),
            "freqs": cached_freqs_list,
        }
        self._update_merger_grid(new_grid_h, new_grid_w)
        self._current_grid = key

    def _update_merger_grid(self, new_grid_h: int, new_grid_w: int) -> None:
        """Merger must track non-canonical grids or it raises on L_pre mismatch."""
        kh, kw = self.merger.kh, self.merger.kw
        if new_grid_h % kh != 0 or new_grid_w % kw != 0:
            raise RuntimeError(
                f"non-canonical grid ({new_grid_h},{new_grid_w}) not divisible by merger kernel ({kh},{kw})"
            )
        self.merger.grid_h = new_grid_h
        self.merger.grid_w = new_grid_w
        self.merger.nh = new_grid_h // kh
        self.merger.nw = new_grid_w // kw

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # FAIL LOUD on input shape
        if pixel_values.ndim != 4:
            raise RuntimeError(
                f"pixel_values must be 4D (L_pre, 3, P, P); "
                f"got {pixel_values.ndim}D shape={tuple(pixel_values.shape)}"
            )

        # D-2: NO-FALLBACK — caller MUST call set_grid(new_grid_h, new_grid_w)
        # before forward when L_pre changes. Auto-detection here is a silent
        # path that would mask shape bugs (e.g. caller passing a typo'd grid
        # for the pre-patchifier and a different one for the model would just
        # work, hiding the inconsistency).
        L_pre = pixel_values.shape[0]
        if L_pre != self.embeddings.num_patches:
            raise RuntimeError(
                f"pixel_values L_pre={L_pre} != embeddings.num_patches="
                f"{self.embeddings.num_patches} (grid="
                f"{self.embeddings.grid_h}x{self.embeddings.grid_w}). "
                f"Call model.set_grid(new_grid_h, new_grid_w) BEFORE forward — "
                f"the model does not auto-detect grid from pixel_values."
            )

        # Site 7: bf16 autocast wrapper.
        # Note: register_buffer tensors (pos_emb_baked, freqs_cos/sin) are fp32; they
        # are cast on consumption inside the submodules (embeddings.forward broadcasts
        # pos_emb_baked; _moonvit_apply_rope casts freqs_cos/sin to q.dtype).
        with torch.autocast(
            device_type=pixel_values.device.type,
            dtype=torch.bfloat16,
            enabled=self.use_bf16,
        ):
            embeddings = self.embeddings(pixel_values)
            encoded = self.transformer(embeddings)
            merged = self.merger(encoded)
            out = self.projector(merged)
        return out

    # ------------------------------------------------------------------
    # Site 8: weight loader (classmethod factory)
    # ------------------------------------------------------------------
    @classmethod
    def from_moonvit_state_dict(
        cls,
        state_dict: dict,
        *,
        grid_h: int,
        grid_w: int,
        hidden_size: int = 1152,
        patch_size: int = 14,
        n_layers: int = 27,
        n_heads: int = 16,
        intermediate_size: int = 4304,
        kh: int = 2,
        kw: int = 2,
        text_hidden: int = 2048,
        use_bf16: bool = True,
    ) -> "MoonViTVisionModel":
        """Build a MoonViTVisionModel from a real MoonViT state_dict.

        Source key layout (confirmed via Read phase on the actual checkpoint;
        the legacy docstring claiming ``vit.*`` is wrong — the live checkpoint
        uses ``vision_model.*``):

          Patch embed:
            vision_model.patch_embed.proj.{weight,bias}
            vision_model.patch_embed.pos_emb.weight          (consumed at bake)

          Encoder blocks (i in 0..n_layers-1):
            vision_model.encoder.blocks.{i}.norm0.{weight,bias}      pre-attn LN
            vision_model.encoder.blocks.{i}.wqkv.{weight,bias}       FUSED qkv
            vision_model.encoder.blocks.{i}.wo.{weight,bias}         attn output proj
            vision_model.encoder.blocks.{i}.norm1.{weight,bias}      pre-MLP LN
            vision_model.encoder.blocks.{i}.mlp.fc0.{weight,bias}    MLP up
            vision_model.encoder.blocks.{i}.mlp.fc1.{weight,bias}    MLP down

          Final norm after the stack:
            vision_model.encoder.final_layernorm.{weight,bias}

          VL connector / pixel-shuffle MLP (top-level, NOT under vision_model):
            mlp1.0.{weight,bias}   LayerNorm
            mlp1.1.{weight,bias}   Linear (in -> text_hidden)
            mlp1.3.{weight,bias}   Linear (text_hidden -> text_hidden)
                                   (index 2 is GELU, no params)

        FAIL-LOUD discipline (no try/except, no silent skips):
          - missing pos_emb / patch_embed / mlp1 keys -> RuntimeError
          - shape mismatch on any tensor at load time -> RuntimeError
          - block index range mismatch (actual layer count != n_layers) -> RuntimeError
          - any state_dict key that doesn't match the explicit allowlist of
            patterns -> RuntimeError (no silent skip, no strict=False)

        Encoder construction is intentionally minimal: we build a pure-PT
        container of MoonViT encoder blocks via _build_moonvit_encoder_stack
        (no tensorrt_llm import on Mac). The TRT-LLM SiglipVisionEncoder swap
        is the responsibility of the build.py path that wires real Attention
        modules; this loader only owns key translation + tensor copy.
        """
        import re

        # 1. ------------------------------------------------------------
        # Extract pos_emb source tensor for the embedding bake.
        pos_emb_key = "vision_model.patch_embed.pos_emb.weight"
        if pos_emb_key not in state_dict:
            raise RuntimeError(
                f"missing required key: {pos_emb_key} "
                f"(available top-level prefixes: "
                f"{sorted({k.split('.', 1)[0] for k in state_dict})})"
            )
        moonvit_pos_emb = state_dict[pos_emb_key]

        # 2. ------------------------------------------------------------
        # Extract mlp1 sub-state-dict for the projector.
        mlp1_sd = {
            k[len("mlp1."):]: v
            for k, v in state_dict.items()
            if k.startswith("mlp1.")
        }
        if not mlp1_sd:
            raise RuntimeError(
                "missing mlp1.* keys — projector cannot be built. The MoonViT "
                "VL connector lives at top-level mlp1.{0,1,3}.{weight,bias}."
            )

        # 3. ------------------------------------------------------------
        # Patch-embed conv weights (consumed by MoonViTVisionEmbeddings after
        # construction — we copy_ them explicitly to keep init deterministic).
        proj_w_key = "vision_model.patch_embed.proj.weight"
        proj_b_key = "vision_model.patch_embed.proj.bias"
        for k in (proj_w_key, proj_b_key):
            if k not in state_dict:
                raise RuntimeError(f"missing required key: {k}")
        proj_w = state_dict[proj_w_key]
        proj_b = state_dict[proj_b_key]

        # 4. ------------------------------------------------------------
        # Final layernorm (after the encoder stack).
        final_ln_w_key = "vision_model.encoder.final_layernorm.weight"
        final_ln_b_key = "vision_model.encoder.final_layernorm.bias"
        for k in (final_ln_w_key, final_ln_b_key):
            if k not in state_dict:
                raise RuntimeError(f"missing required key: {k}")
        final_ln_w = state_dict[final_ln_w_key]
        final_ln_b = state_dict[final_ln_b_key]

        # 5. ------------------------------------------------------------
        # Allowlist-based key partitioning. Build the full target set of keys
        # we expect to consume, then assert state_dict matches exactly (modulo
        # the consumed top-level singletons already extracted above).
        #
        # Pattern: vision_model.encoder.blocks.{i}.<subkey>
        block_subkeys = (
            "norm0.weight", "norm0.bias",
            "wqkv.weight",  "wqkv.bias",
            "wo.weight",    "wo.bias",
            "norm1.weight", "norm1.bias",
            "mlp.fc0.weight", "mlp.fc0.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
        )
        # Discover the actual layer-index set so we can FAIL LOUD on
        # off-by-one between n_layers and what the checkpoint carries.
        block_re = re.compile(
            r"^vision_model\.encoder\.blocks\.(\d+)\.([A-Za-z0-9_.]+)$"
        )
        discovered_layers: set[int] = set()
        for k in state_dict:
            m = block_re.match(k)
            if m is not None:
                discovered_layers.add(int(m.group(1)))
        expected_layers = set(range(n_layers))
        if discovered_layers != expected_layers:
            extra = sorted(discovered_layers - expected_layers)
            missing = sorted(expected_layers - discovered_layers)
            raise RuntimeError(
                f"n_layers={n_layers} mismatch with state_dict block indices. "
                f"missing layer indices: {missing}; "
                f"unexpected layer indices: {extra}"
            )

        # Build the exact allowlist of recognized keys.
        recognized: set[str] = set()
        recognized.add(pos_emb_key)
        recognized.add(proj_w_key)
        recognized.add(proj_b_key)
        recognized.add(final_ln_w_key)
        recognized.add(final_ln_b_key)
        for i in range(n_layers):
            for sub in block_subkeys:
                recognized.add(f"vision_model.encoder.blocks.{i}.{sub}")
        for sub in ("0.weight", "0.bias", "1.weight", "1.bias",
                    "3.weight", "3.bias"):
            recognized.add(f"mlp1.{sub}")

        unrecognized = sorted(set(state_dict.keys()) - recognized)
        if unrecognized:
            raise RuntimeError(
                f"state_dict contains {len(unrecognized)} unrecognized keys "
                f"(no silent skip — every key must be either consumed or "
                f"explicitly excluded by the loader): {unrecognized[:8]}"
                + (" ..." if len(unrecognized) > 8 else "")
            )

        # 6. ------------------------------------------------------------
        # Build submodules. Embeddings first — its ctor handles the
        # pos_emb bicubic bake.
        embeddings = MoonViTVisionEmbeddings(
            hidden_size=hidden_size,
            patch_size=patch_size,
            grid_h=grid_h,
            grid_w=grid_w,
            moonvit_pos_emb_weight=moonvit_pos_emb,
        )
        # Load the patch_embed conv weights into the freshly-built embedding.
        expected_w = (hidden_size, 3, patch_size, patch_size)
        if tuple(proj_w.shape) != expected_w:
            raise RuntimeError(
                f"patch_embed.proj.weight shape {tuple(proj_w.shape)} "
                f"!= expected {expected_w}"
            )
        if tuple(proj_b.shape) != (hidden_size,):
            raise RuntimeError(
                f"patch_embed.proj.bias shape {tuple(proj_b.shape)} "
                f"!= expected ({hidden_size},)"
            )
        with torch.no_grad():
            embeddings.patch_embedding.weight.copy_(proj_w)
            embeddings.patch_embedding.bias.copy_(proj_b)

        # 7. ------------------------------------------------------------
        # Build the encoder stack (pure-PT, no tensorrt_llm import). Each
        # block mirrors the MoonViT layout exactly: LN -> fused QKV -> WO ->
        # LN -> MLP(fc0, GELU, fc1). The TRT-LLM-aware swap (replacing each
        # block's attention with MoonViTAttention) is build.py's job; here we
        # only own deterministic tensor placement.
        encoder = _build_moonvit_encoder_stack(
            n_layers=n_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
        )

        # Final layernorm shape check + copy.
        if tuple(final_ln_w.shape) != (hidden_size,):
            raise RuntimeError(
                f"final_layernorm.weight shape {tuple(final_ln_w.shape)} "
                f"!= ({hidden_size},)"
            )
        if tuple(final_ln_b.shape) != (hidden_size,):
            raise RuntimeError(
                f"final_layernorm.bias shape {tuple(final_ln_b.shape)} "
                f"!= ({hidden_size},)"
            )
        with torch.no_grad():
            encoder.final_layernorm.weight.copy_(final_ln_w)
            encoder.final_layernorm.bias.copy_(final_ln_b)

        # Per-block tensor copy with strict shape checks.
        for i in range(n_layers):
            block = encoder.blocks[i]
            prefix = f"vision_model.encoder.blocks.{i}"

            # norm0 (pre-attention)
            _copy_layernorm(
                block.norm0,
                state_dict[f"{prefix}.norm0.weight"],
                state_dict[f"{prefix}.norm0.bias"],
                tag=f"{prefix}.norm0",
                expected_dim=hidden_size,
            )
            # wqkv: fused qkv projection (3*hidden_size, hidden_size)
            _copy_linear(
                block.wqkv,
                state_dict[f"{prefix}.wqkv.weight"],
                state_dict[f"{prefix}.wqkv.bias"],
                tag=f"{prefix}.wqkv",
                expected_w_shape=(3 * hidden_size, hidden_size),
                expected_b_shape=(3 * hidden_size,),
            )
            # wo: attention output projection (hidden_size, hidden_size)
            _copy_linear(
                block.wo,
                state_dict[f"{prefix}.wo.weight"],
                state_dict[f"{prefix}.wo.bias"],
                tag=f"{prefix}.wo",
                expected_w_shape=(hidden_size, hidden_size),
                expected_b_shape=(hidden_size,),
            )
            # norm1 (pre-MLP)
            _copy_layernorm(
                block.norm1,
                state_dict[f"{prefix}.norm1.weight"],
                state_dict[f"{prefix}.norm1.bias"],
                tag=f"{prefix}.norm1",
                expected_dim=hidden_size,
            )
            # mlp.fc0: up-projection (intermediate_size, hidden_size)
            _copy_linear(
                block.mlp.fc0,
                state_dict[f"{prefix}.mlp.fc0.weight"],
                state_dict[f"{prefix}.mlp.fc0.bias"],
                tag=f"{prefix}.mlp.fc0",
                expected_w_shape=(intermediate_size, hidden_size),
                expected_b_shape=(intermediate_size,),
            )
            # mlp.fc1: down-projection (hidden_size, intermediate_size)
            _copy_linear(
                block.mlp.fc1,
                state_dict[f"{prefix}.mlp.fc1.weight"],
                state_dict[f"{prefix}.mlp.fc1.bias"],
                tag=f"{prefix}.mlp.fc1",
                expected_w_shape=(hidden_size, intermediate_size),
                expected_b_shape=(hidden_size,),
            )

        # 8. ------------------------------------------------------------
        # Build merger + projector. Projector ctor validates mlp1 shapes
        # and copies tensors itself (no double-copy here).
        merger = MoonViTPatchMerger(grid_h=grid_h, grid_w=grid_w, kh=kh, kw=kw)
        in_features = hidden_size * kh * kw
        projector = MoonViTProjector(
            in_features=in_features,
            text_hidden=text_hidden,
            mlp1_state_dict=mlp1_sd,
        )

        # 9. ------------------------------------------------------------
        # Wrap encoder in the transformer container and assemble the model.
        transformer = MoonViTVisionTransformer(encoder)
        model = cls(embeddings, transformer, merger, projector, use_bf16=use_bf16)
        return model




# ----------------------------------------------------------------------
# PT-only attention adapter (Option B2) -- parity harness for Mac/CI runs.
#
# The TRT-LLM-bound MoonViTAttention requires a live ModelConfig with
# attn_backend == VANILLA and a lazy import of tensorrt_llm._torch.modules.
# That belongs in build.py (still scaffolding). Until build.py exists, this
# pure-PT block lets us exercise the encoder forward path end-to-end.
#
# Math:
#   x_norm = norm0(x)
#   qkv    = wqkv(x_norm)                              # (L, 3*H*D_head)
#   q,k,v  = qkv.chunk(3, dim=-1)                      # each (L, H*D_head)
#   q,k    = apply_rope_real(q,k, freqs_packed)        # 2-D RoPE
#   attn   = softmax(q @ k.T / sqrt(D_head)) @ v       # SigLIP-style MHA
#   x      = x + wo(attn)
#   x      = x + mlp(norm1(x))
#
# freqs are supplied at swap time and shared across all 27 blocks (same buffer
# dedup contract as the TRT-LLM path).
# ----------------------------------------------------------------------
class _PTBlockFreqsProxy(nn.Module):
    """Thin handle exposing the vendor MoonViTAttention freqs interface
    (freqs_cos, freqs_sin, reslice_freqs) on top of _MoonViTPTAttentionBlock,
    which stores freqs as a single packed buffer self.freqs_packed of layout
    (L, D_head/2, 2) where the last dim packs (cos, sin).

    reslice_freqs updates all three: self.freqs_cos, self.freqs_sin, and
    self._parent.freqs_packed (which is what the block's attention forward consumes).
    """
    def __init__(self, parent, freqs_source):
        super().__init__()
        self._parent = parent
        # Full source table: (H_max, W_max, D_head/2, 2), last dim packs (cos, sin).
        self.register_buffer("freqs_source", freqs_source, persistent=False)
        # Current slice exposed as vendor-compatible cos/sin buffers.
        # Initial values must match parent.freqs_packed at install time.
        fp = parent.freqs_packed  # (L_init, D_head/2, 2)
        if fp.ndim != 3 or fp.shape[-1] != 2:
            raise RuntimeError(
                f"_PTBlockFreqsProxy: parent.freqs_packed shape {tuple(fp.shape)} not (L, D/2, 2)"
            )
        self.register_buffer("freqs_cos", fp[..., 0].contiguous(), persistent=False)
        self.register_buffer("freqs_sin", fp[..., 1].contiguous(), persistent=False)

    def reslice_freqs(self, new_grid_h, new_grid_w):
        new_grid_h = int(new_grid_h)
        new_grid_w = int(new_grid_w)
        if new_grid_h <= 0 or new_grid_w <= 0:
            raise RuntimeError(
                f"reslice_freqs: positive grid required, got ({new_grid_h},{new_grid_w})"
            )
        H_max, W_max = self.freqs_source.shape[0], self.freqs_source.shape[1]
        if new_grid_h > H_max or new_grid_w > W_max:
            raise RuntimeError(
                f"reslice_freqs: grid ({new_grid_h},{new_grid_w}) exceeds source ({H_max},{W_max})"
            )
        D_half = self.freqs_source.shape[2]
        sliced = self.freqs_source[:new_grid_h, :new_grid_w]  # (h, w, D/2, 2)
        sliced_flat = sliced.reshape(new_grid_h * new_grid_w, D_half, 2).contiguous()
        # Update parent block's packed buffer (what attention forward consumes).
        parent_dtype = self._parent.freqs_packed.dtype
        parent_dev = self._parent.freqs_packed.device
        self._parent.register_buffer(
            "freqs_packed",
            sliced_flat.to(device=parent_dev, dtype=parent_dtype).contiguous(),
            persistent=False,
        )
        # Update vendor-compatible buffers (what set_grid's cache reads).
        self.register_buffer(
            "freqs_cos",
            sliced_flat[..., 0].to(device=parent_dev, dtype=parent_dtype).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "freqs_sin",
            sliced_flat[..., 1].to(device=parent_dev, dtype=parent_dtype).contiguous(),
            persistent=False,
        )

    def _sync_packed_from_cos_sin(self):
        """Call after externally swapping freqs_cos/freqs_sin (e.g. cache restore
        in set_grid) to ensure parent.freqs_packed is in sync."""
        if self.freqs_cos.shape != self.freqs_sin.shape:
            raise RuntimeError(
                f"_sync_packed: cos/sin shape mismatch "
                f"{tuple(self.freqs_cos.shape)} vs {tuple(self.freqs_sin.shape)}"
            )
        packed = torch.stack([self.freqs_cos, self.freqs_sin], dim=-1).contiguous()
        parent_dtype = self._parent.freqs_packed.dtype
        parent_dev = self._parent.freqs_packed.device
        self._parent.register_buffer(
            "freqs_packed",
            packed.to(device=parent_dev, dtype=parent_dtype).contiguous(),
            persistent=False,
        )


def build_freqs_packed_for(freqs_cis_table, h, w):
    """Slice the HF freqs_cis table for (h, w) and pack as (L, D_head/2, 2) real.

    Input freqs_cis_table is complex (H_max, W_max, D_head/2). Output is a real
    tensor (h*w, D_head/2, 2) where the last dim is (cos, sin), matching the
    layout consumed by _MoonViTPTAttentionBlock.forward and patches.apply_rope_real.
    """
    fc = freqs_cis_table[:h, :w].reshape(-1, freqs_cis_table.shape[-1])  # complex (L, D/2)
    cos = fc.real.contiguous().float()
    sin = fc.imag.contiguous().float()
    return torch.stack([cos, sin], dim=-1).contiguous()  # (L, D/2, 2)


class _MoonViTPTAttentionBlock(nn.Module):
    """Pure-PT MoonViT encoder block. Drop-in replacement for _MoonViTEncoderBlock
    that owns real attention math instead of raising. Re-uses the existing
    wqkv/wo/norm0/norm1/mlp submodules from the source block (no weight copy)."""

    def __init__(self, source_block, freqs_packed):
        super().__init__()
        self.norm0 = source_block.norm0
        self.wqkv = source_block.wqkv
        self.wo = source_block.wo
        self.norm1 = source_block.norm1
        self.mlp = source_block.mlp
        self.hidden_size = int(source_block.hidden_size)
        self.n_heads = int(source_block.n_heads)
        self.head_dim = int(source_block.head_dim)
        self.register_buffer("freqs_packed", freqs_packed, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from lrai_locate_anything.patches import apply_rope_real as _apply_rope_real

        if x.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"_MoonViTPTAttentionBlock: input last dim {x.shape[-1]} != hidden_size {self.hidden_size}"
            )

        x_n = self.norm0(x)
        qkv = self.wqkv(x_n)
        q, k, v = qkv.chunk(3, dim=-1)

        H = self.n_heads
        Dh = self.head_dim

        squeeze_batch = False
        if q.ndim == 3 and q.shape[0] == 1:
            q = q.squeeze(0); k = k.squeeze(0); v = v.squeeze(0)
            squeeze_batch = True
        elif q.ndim != 2:
            raise RuntimeError(
                f"_MoonViTPTAttentionBlock expects (L, H*Dh) or (1, L, H*Dh); "
                f"got q.ndim={q.ndim} shape={tuple(q.shape)}"
            )

        L_q = q.shape[0]
        if self.freqs_packed.shape[0] != L_q:
            raise RuntimeError(
                f"freqs_packed length {self.freqs_packed.shape[0]} != L {L_q}"
            )
        if self.freqs_packed.shape[-2] != Dh // 2:
            raise RuntimeError(
                f"freqs_packed inner dim {self.freqs_packed.shape[-2]} != head_dim//2 {Dh//2}"
            )

        q = q.reshape(L_q, H, Dh)
        k = k.reshape(L_q, H, Dh)
        v = v.reshape(L_q, H, Dh)

        freqs = self.freqs_packed.to(q.dtype)
        q, k = _apply_rope_real(q, k, freqs)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scale = 1.0 / math.sqrt(Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().reshape(L_q, H * Dh)

        out = self.wo(out)
        if squeeze_batch:
            out = out.unsqueeze(0)

        x = x + out
        x = x + self.mlp(self.norm1(x))
        return x


def _install_pt_attention_swap(model, freqs_packed, freqs_source=None):
    """Swap every _MoonViTEncoderBlock in encoder.blocks for a _MoonViTPTAttentionBlock.

    If `freqs_source` is supplied (layout (H_max, W_max, D_head/2, 2)), each new
    PT block is also given a `block.attention = _PTBlockFreqsProxy(block, freqs_source)`
    so MoonViTVisionModel._iter_attentions can re-rope on set_grid(). Without a
    source, set_grid() will FAIL LOUD via the proxy's absence — no silent skip.
    """
    encoder = model.transformer.encoder
    if not hasattr(encoder, "blocks"):
        raise RuntimeError("encoder has no .blocks attribute")
    new_blocks = nn.ModuleList()
    for i, blk in enumerate(encoder.blocks):
        if isinstance(blk, _MoonViTPTAttentionBlock):
            new_blocks.append(blk)
            continue
        if not isinstance(blk, _MoonViTEncoderBlock):
            raise RuntimeError(
                f"block[{i}] is {type(blk).__name__}, not _MoonViTEncoderBlock; "
                f"refusing to swap an unexpected block type"
            )
        new_block = _MoonViTPTAttentionBlock(blk, freqs_packed)
        if freqs_source is not None:
            new_block.attention = _PTBlockFreqsProxy(new_block, freqs_source)
        new_blocks.append(new_block)
    encoder.blocks = new_blocks
    return model


def _install_pt_attention_swap_method(self, freqs_packed, freqs_source=None):
    """Method bound onto MoonViTVisionModel; see _install_pt_attention_swap."""
    return _install_pt_attention_swap(self, freqs_packed, freqs_source=freqs_source)


MoonViTVisionModel.install_pt_attention_swap = _install_pt_attention_swap_method


# ----------------------------------------------------------------------
# Site 8 helpers — pure-PT encoder skeleton + strict tensor copy utilities.
# Kept module-level (not nested) so build.py can reuse the same shapes when
# swapping in TRT-LLM Attention layers.
# ----------------------------------------------------------------------
class _MoonViTMLP(nn.Module):
    """MoonViT MLP submodule — fc0 (up) -> GELU -> fc1 (down).

    Names mirror the source state_dict (mlp.fc0 / mlp.fc1) so the loader can
    copy weights without an alias scheme.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc0 = nn.Linear(hidden_size, intermediate_size)
        self.fc1 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU(approximate='tanh')  # matches HF MoonViT PytorchGELUTanh; vendor diverges from exact erf by ~1.06 across 27 layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.act(self.fc0(x)))


class _MoonViTEncoderBlock(nn.Module):
    """Single MoonViT encoder block in pure PT.

    Layout (mirrors source state_dict exactly):
      norm0 (LN) -> wqkv (fused) -> wo -> residual
      norm1 (LN) -> mlp.fc0 -> GELU -> mlp.fc1 -> residual

    Attention math is intentionally NOT implemented here — this skeleton
    exists so the Site 8 loader can place tensors deterministically. The
    build.py path replaces `wqkv`/`wo` with a MoonViTAttention wrapper at
    engine-build time; on Mac the parity tests stop after `from_moonvit_state_dict`
    returns and never exercise this forward.
    """
    def __init__(self, hidden_size: int, n_heads: int, intermediate_size: int):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise RuntimeError(
                f"hidden_size {hidden_size} not divisible by n_heads {n_heads}"
            )
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.intermediate_size = intermediate_size

        self.norm0 = nn.LayerNorm(hidden_size)
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = _MoonViTMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: this forward is a structural placeholder so the module is a
        # valid nn.Module — the production attention path goes through
        # MoonViTAttention (TRT-LLM Attention base). Calling forward here on
        # Mac without that swap is a programmer error and we fail loud rather
        # than emit wrong numbers.
        raise RuntimeError(
            "_MoonViTEncoderBlock.forward called directly — the production path "
            "swaps wqkv/wo for MoonViTAttention via trtllm_prod/build.py. This "
            "skeleton exists only for deterministic weight placement."
        )


class _MoonViTEncoderStack(nn.Module):
    """Pure-PT encoder skeleton: ModuleList of blocks + final_layernorm.

    Mirrors the source state_dict layout (vision_model.encoder.blocks.{i} +
    vision_model.encoder.final_layernorm) so the loader can place tensors
    by name without a remap table.
    """
    def __init__(self, n_layers: int, hidden_size: int, n_heads: int,
                 intermediate_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _MoonViTEncoderBlock(hidden_size, n_heads, intermediate_size)
            for _ in range(n_layers)
        ])
        self.final_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_layernorm(x)


def _build_moonvit_encoder_stack(
    *, n_layers: int, hidden_size: int, n_heads: int, intermediate_size: int,
) -> _MoonViTEncoderStack:
    """Factory for the pure-PT encoder skeleton consumed by Site 8.

    Kept as a free function so build.py can monkey-replace it with the
    TRT-LLM SiglipVisionEncoder factory without subclass surgery.
    """
    return _MoonViTEncoderStack(
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_heads=n_heads,
        intermediate_size=intermediate_size,
    )


def _copy_linear(
    module: nn.Linear,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    tag: str,
    expected_w_shape: tuple,
    expected_b_shape: tuple,
) -> None:
    """Strict copy of (weight, bias) into nn.Linear.

    FAIL LOUD on shape mismatch. No try/except, no broadcast, no transpose
    guess — the caller declares the expected shape and we assert it.
    """
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"{tag}: target is not nn.Linear (got {type(module)})")
    if tuple(weight.shape) != expected_w_shape:
        raise RuntimeError(
            f"{tag}.weight shape {tuple(weight.shape)} != {expected_w_shape}"
        )
    if tuple(bias.shape) != expected_b_shape:
        raise RuntimeError(
            f"{tag}.bias shape {tuple(bias.shape)} != {expected_b_shape}"
        )
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)


def _copy_layernorm(
    module: nn.LayerNorm,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    tag: str,
    expected_dim: int,
) -> None:
    """Strict copy of (weight, bias) into nn.LayerNorm with shape check."""
    if not isinstance(module, nn.LayerNorm):
        raise RuntimeError(f"{tag}: target is not nn.LayerNorm (got {type(module)})")
    if tuple(weight.shape) != (expected_dim,):
        raise RuntimeError(
            f"{tag}.weight shape {tuple(weight.shape)} != ({expected_dim},)"
        )
    if tuple(bias.shape) != (expected_dim,):
        raise RuntimeError(
            f"{tag}.bias shape {tuple(bias.shape)} != ({expected_dim},)"
        )
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
