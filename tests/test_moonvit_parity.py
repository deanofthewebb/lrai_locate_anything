"""Parity gate for MoonViT TRT-LLM port (Phase D).

PHASE D SCAFFOLD: parity tests are pytest.skip-marked until the port lands.
test_embeddings_parity is now active (sites 1+2).
"""
import pytest
import torch


_SKIP_REASON = "Phase D MoonViT port in progress — see docs/phase_d_moonvit_port_delta.md"


def test_embeddings_parity():
    """Sites 1+2 parity: MoonViTVisionEmbeddings matches export/vision.py
    on the pos_emb add step within 5e-4 fp16 max-abs-diff.
    """
    import torch
    from lrai_locate_anything.trtllm_prod.modeling_moonvit import MoonViTVisionEmbeddings

    grid_h, grid_w = 36, 46
    hidden_size = 1152
    patch_size = 14
    H_max, W_max = 64, 64  # arbitrary source table

    torch.manual_seed(0)
    moonvit_pos_emb = torch.randn(H_max, W_max, hidden_size, dtype=torch.float32) * 0.02

    # Build our subclass
    emb = MoonViTVisionEmbeddings(
        hidden_size=hidden_size,
        patch_size=patch_size,
        grid_h=grid_h,
        grid_w=grid_w,
        moonvit_pos_emb_weight=moonvit_pos_emb,
    )

    # Reference: mirror the export/vision.py:35-44 + 55-57 logic
    # (pure reference computation, no NotImplementedError or skipping)
    import torch.nn.functional as F
    pos_ref = moonvit_pos_emb.permute(2, 0, 1).unsqueeze(0)
    pos_ref = F.interpolate(pos_ref, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
    pos_ref = pos_ref.squeeze(0).permute(1, 2, 0).contiguous().flatten(end_dim=1)
    # The patch_embedding is randomly initialized in both paths via torch seed-control;
    # we test that pos_emb_baked matches pos_ref and that the forward path adds it.
    assert torch.allclose(emb.pos_emb_baked.float(), pos_ref.float(), atol=1e-6), \
        f"pos_emb_baked mismatch: max-diff={(emb.pos_emb_baked.float()-pos_ref.float()).abs().max():.6f}"

    # Forward shape + add correctness
    pixel_values = torch.randn(grid_h*grid_w, 3, patch_size, patch_size, dtype=torch.float32) * 0.1
    out = emb(pixel_values)
    assert out.shape == (grid_h*grid_w, hidden_size), f"output shape {out.shape}"

    # Difference between out and (patch_proj(pixel_values) + pos_ref) should be 0
    expected = emb.patch_embedding(pixel_values).view(grid_h*grid_w, hidden_size) + pos_ref
    max_diff = (out - expected).abs().max().item()
    assert max_diff < 5e-4, f"forward parity max-diff {max_diff:.6f} > 5e-4"


@pytest.mark.skip(reason=_SKIP_REASON)
def test_attention_parity():
    """Asserts MoonViTAttention with apply_rope_real matches our patches.py:sdpa_packed."""
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_full_forward_parity():
    """End-to-end: same (1656,3,14,14) input through both stacks; max-abs-diff < 1e-3."""
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_dynamic_grid_interpolation():
    """Non-canonical grid (e.g. 24x32) re-interpolates pos_emb_baked cleanly."""
    pass
