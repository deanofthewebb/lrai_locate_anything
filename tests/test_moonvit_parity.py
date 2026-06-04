"""Parity gate for MoonViT TRT-LLM port (Phase D).

PHASE D SCAFFOLD: all parity tests are pytest.skip-marked until the port lands.
"""
import pytest
import torch


pytestmark = pytest.mark.skip(reason="Phase D MoonViT port in progress — see docs/phase_d_moonvit_port_delta.md")


def test_embeddings_parity():
    """Asserts MoonViTVisionEmbeddings output matches export/vision.py within 1e-3 fp16."""
    pass


def test_attention_parity():
    """Asserts MoonViTAttention with apply_rope_real matches our patches.py:sdpa_packed."""
    pass


def test_full_forward_parity():
    """End-to-end: same (1656,3,14,14) input through both stacks; max-abs-diff < 1e-3."""
    pass


def test_dynamic_grid_interpolation():
    """Non-canonical grid (e.g. 24x32) re-interpolates pos_emb_baked cleanly."""
    pass
