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


class MoonViTAttention(nn.Module):
    """Replaces SiglipAttention with 2D-RoPE application before SDPA.

    TODO Port site 3 (Appendix A.3): q,k,v each (B=1, H=16, T=L_pre=1656, D_head=72)
    after view(B, T, H, D_head).transpose(1, 2); self.freqs_cos/self.freqs_sin are
    (L_pre, D_head/2) = (1656, 36); reshape to (1, 1, T, D_head/2) before broadcast
    over batch and head axes (NOT unsqueeze(-2) — head axis is at dim=1, not dim=-2).
    Call apply_rope_real(q, k, freqs_cos, freqs_sin) in fp32 then cast back to module
    dtype before scaled_dot_product_attention. o_proj keeps bias=True.
    TODO Port site 4 (Appendix A.4): in __init__, build Rope2DReal(D_head=72, max_grid_h,
    max_grid_w).get_freqs_cis(grid_h=36, grid_w=46) -> (36, 46, 36, 2); unbind(-1) ->
    freqs_cos/freqs_sin each (36, 46, 36); flatten(end_dim=1) -> (L_pre=1656, 36);
    register both as fp32 non-persistent buffers (~6.4 MB shared across 27 layers).
    """
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, hidden_states, attention_mask=None):
        raise NotImplementedError


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
