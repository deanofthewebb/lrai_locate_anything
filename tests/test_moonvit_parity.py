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


def test_attention_parity():
    """Sites 3+4 parity: MoonViTAttention.apply_rope produces the same q, k
    tensors as a direct patches.apply_rope_real call.

    PT-only — does NOT import tensorrt_llm so it runs on Mac.

    Strategy: the module-level _moonvit_apply_rope helper is the exact
    function bound onto the dynamically-built MoonViTAttention_TRTLLM
    subclass (modeling_moonvit.py:311). We invoke it against a minimal
    duck-typed `self` (no TRT-LLM base in MRO) so the rotation math runs
    on plain torch tensors — this is the documented testing seam (see the
    class docstring at modeling_moonvit.py:237-241).

    Scope: this test asserts THAT our override forwards to patches.apply_rope_real
    correctly. The full TRT-LLM forward path (split_qkv/convert_qkv against the
    real Attention base) is covered by test_phase_d_parity_container.py (future).
    """
    import torch
    from lrai_locate_anything.patches import apply_rope_real as patches_apply_rope_real
    from lrai_locate_anything.trtllm_prod.modeling_moonvit import _moonvit_apply_rope

    # Synthetic tensors matching MoonViT contract (1656 tokens, 16 heads, head_dim 72)
    grid_h, grid_w = 36, 46
    L = grid_h * grid_w  # 1656
    H = 16               # num_heads
    D_head = 72          # 1152 / 16

    torch.manual_seed(0)
    # Fused (L, H*D_head) layout — _moonvit_apply_rope expects this and reshapes.
    q_fused = torch.randn(L, H * D_head, dtype=torch.float32) * 0.02
    k_fused = torch.randn(L, H * D_head, dtype=torch.float32) * 0.02
    v_fused = torch.randn(L, H * D_head, dtype=torch.float32) * 0.02

    # fp32 cos/sin buffers (production stores fp32; cast happens inside override)
    freqs_cos = torch.randn(L, D_head // 2, dtype=torch.float32)
    freqs_sin = torch.randn(L, D_head // 2, dtype=torch.float32)

    # --- Reference: replicate the override's data path WITHOUT routing through
    # _moonvit_apply_rope. Reshape -> pack freqs -> call patches.apply_rope_real
    # in q's dtype (matches override cast at modeling_moonvit.py:204-205).
    q_ref_in = q_fused.clone().reshape(L, H, D_head)
    k_ref_in = k_fused.clone().reshape(L, H, D_head)
    fc_cast = freqs_cos.to(q_ref_in.dtype)
    fs_cast = freqs_sin.to(q_ref_in.dtype)
    freqs_packed = torch.stack([fc_cast, fs_cast], dim=-1)
    q_ref, k_ref = patches_apply_rope_real(q_ref_in, k_ref_in, freqs_packed)
    q_ref = q_ref.reshape(L, H * D_head)
    k_ref = k_ref.reshape(L, H * D_head)

    # --- Tested: invoke the actual override via a minimal duck-typed self.
    # split_qkv is a pass-through for already-split tensors (VANILLA backend
    # has support_fused_qkv=False); convert_qkv re-fuses for return.
    class _MockAttention:
        def __init__(self):
            self.num_heads = H
            self.head_dim = D_head
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
            self.grid_h = grid_h
            self.grid_w = grid_w

        def split_qkv(self, q, k, v):
            return q, k, v

        def convert_qkv(self, q, k, v):
            return q, k, v

    mock_self = _MockAttention()
    q_test, k_test, v_test = _moonvit_apply_rope(
        mock_self,
        q_fused.clone(),
        k_fused.clone(),
        v_fused.clone(),
        position_ids=None,
    )

    max_diff_q = (q_ref - q_test).abs().max().item()
    max_diff_k = (k_ref - k_test).abs().max().item()
    max_diff_v = (v_fused - v_test).abs().max().item()

    assert max_diff_q < 1e-6, f"q diff {max_diff_q:.6e}"
    assert max_diff_k < 1e-6, f"k diff {max_diff_k:.6e}"
    # v must pass through untouched — rotation is q/k only.
    assert max_diff_v == 0.0, f"v was mutated: max-diff {max_diff_v:.6e}"


@pytest.mark.skip(reason=_SKIP_REASON)
def test_full_forward_parity():
    """End-to-end: same (1656,3,14,14) input through both stacks; max-abs-diff < 1e-3."""
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_dynamic_grid_interpolation():
    """Non-canonical grid (e.g. 24x32) re-interpolates pos_emb_baked cleanly."""
    pass
