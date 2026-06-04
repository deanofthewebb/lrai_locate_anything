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
from torch import nn


class MoonViTVisionEmbeddings(nn.Module):
    """Replaces SiglipVisionEmbeddings.

    TODO Port site 1 (modeling_siglip.py:N): replace 1D nn.Embedding position_embedding
    with register_buffer('pos_emb_baked', shape (max_L_pre, hidden_size)).
    TODO Port site 2: forward() pulls pos_emb_baked[:L_pre] instead of position_embedding(ids).
    """
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError("MoonViTVisionEmbeddings: see phase_d_moonvit_port_delta.md")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MoonViTAttention(nn.Module):
    """Replaces SiglipAttention with 2D-RoPE application before SDPA.

    TODO Port site 3: call apply_rope_real(q, k, self.freqs_cos, self.freqs_sin)
    after q,k,v split but before scaled_dot_product_attention.
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

    TODO Port site 7: copy LayerNorm + 2-layer MLP from export/projector.py:ProjectorForExport.
    Output: (L_post, text_hidden=2048).
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
    """Real-space 2-D RoPE. TODO Port site 4: copy from patches.py:apply_rope_real."""
    raise NotImplementedError


def interpolate_pos_encoding(pos_emb_baked, grid_h, grid_w):
    """Dynamic grid support (D-sub-2). TODO Port site 5: bicubic interpolate
    the baked pos_emb to a new (grid_h, grid_w) when called with non-canonical grids."""
    raise NotImplementedError
