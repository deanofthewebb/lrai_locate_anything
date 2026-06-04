"""MoonViT-as-SigLIP-fork — vision encoder for the TRT-LLM production path.

PHASE D SCAFFOLD ONLY. All methods raise NotImplementedError pending
the full 12-day port (see docs/phase_d_moonvit_port_delta.md).

Class hierarchy:
- MoonViTVisionEmbeddings: patch_embed (Conv2d) + pos_emb_baked + Rope2DReal buffers
- MoonViTAttention: 2D-RoPE-aware attention (apply_rope_real before sdpa)
- MoonViTEncoderLayer: pre-norm + attention + MLP (same as SigLIP)
- MoonViTEncoder: stack of n_layers=27 encoder layers
- MoonViTProjector: LayerNorm + Linear + GELU + Linear (mlp1 from our export/projector.py)
- MoonViTVisionModel: top-level wrapper feeding into a TRT-LLM ModelConfig
"""
from __future__ import annotations
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

        # Conv2d patch embedding (matches MoonViT export/vision.py:55-57 + SigLIP)
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Bake pos_emb_baked from MoonViT source table via bicubic interpolation
        # (matches export/vision.py:35-44 exactly)
        pos = moonvit_pos_emb_weight.permute(2, 0, 1).unsqueeze(0)  # (1, d, H_max, W_max)
        pos = F.interpolate(pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos = pos.squeeze(0).permute(1, 2, 0).contiguous().flatten(end_dim=1)  # (L_pre, d)
        if pos.shape != (self.num_patches, hidden_size):
            raise RuntimeError(
                f"baked pos_emb shape {tuple(pos.shape)} != expected ({self.num_patches},{hidden_size})"
            )
        self.register_buffer("pos_emb_baked", pos, persistent=True)
        # FAIL LOUD on degenerate buffer (e.g. all zeros = uninitialized)
        if torch.abs(pos).max().item() < 1e-6:
            raise RuntimeError(
                f"pos_emb_baked appears degenerate (max-abs < 1e-6); "
                f"check moonvit_pos_emb_weight source"
            )

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
        concrete_cls = type(
            "MoonViTAttention_TRTLLM",
            (AttentionBase,),
            {"apply_rope": _moonvit_apply_rope},
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
        return instance


class MoonViTEncoderLayer(nn.Module):
    """Same as SiglipEncoderLayer (pre-norm + attn + MLP). No change needed."""
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError


class MoonViTEncoder(nn.Module):
    """Stack of n_layers=27 encoder blocks."""
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError


class MoonViTProjector(nn.Module):
    """LayerNorm + Linear + GELU + Linear (our mlp1 baked inside vision engine).

    TODO Port site 6 (Appendix A.6) / projector half: input (B, L_post=414, kh*kw*D=4608)
    after StaticPatchMerger; layers LayerNorm(4608) -> Linear(4608, 2048) -> GELU ->
    Linear(2048, 2048); output (B=1, L_post=414, text_hidden=2048). Copy sub-modules
    directly from moonvit_model.mlp1 — do NOT reconstruct from config (activation
    string trap: gelu vs gelu_pytorch_tanh).
    """
    def __init__(self, vit_hidden=1152, kh=2, kw=2, text_hidden=2048):
        super().__init__()
        raise NotImplementedError


class MoonViTVisionModel(nn.Module):
    """Top-level: embeddings -> encoder -> patch_merge -> projector. bf16 autocast."""
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# Module-level helpers
def apply_rope_real(q, k, freqs_cos, freqs_sin):
    """Real-space 2-D RoPE. TODO Port site 3 helper (Appendix A.3): inputs q,k each
    (B=1, H=16, T=L_pre=1656, D_head=72); freqs_cos/freqs_sin each (1, 1, T, D_head/2=36).
    Split q along last dim into (q_real, q_imag) — verify half-then-half vs interleaved
    pairing against MoonViT canonical patches.py. Compute out_r = q_real*cos - q_imag*sin,
    out_i = q_real*sin + q_imag*cos in fp32; reconcat to (B, H, T, D_head); cast back to
    q.dtype before return. Same op on k.
    """
    raise NotImplementedError


def interpolate_pos_encoding(pos_emb_baked, grid_h, grid_w):
    """Dynamic grid support (D-sub-2). TODO Port site 1 helper (Appendix A.1, dynamic path):
    input pos_emb_baked is the *un-interpolated* (H_max, W_max, D) weight (stored as a
    non-persistent buffer in the dynamic path). On grid change call
    F.interpolate(weight.permute(2,0,1).unsqueeze(0), size=(grid_h, grid_w), mode="bicubic",
    align_corners=False) -> squeeze(0).permute(1,2,0).flatten(end_dim=1) ->
    (L_pre=grid_h*grid_w, D=1152). MVP bakes (36, 46) at __init__; this fn is for the
    eager-PT dynamic path only (parity tests, MoonViTAdapter fallback) — TRT engines are
    static and built one-per-bucket."""
    raise NotImplementedError
