"""HF parity regression for MoonViTVisionModel post-Site-8/D-2/3-bug-fix.

This test runs only in the TRT-LLM container (requires real weights at /weights
and AutoModel.from_pretrained with trust_remote_code). Skips on Mac.
"""
import os
import pytest
import torch

WEIGHTS = os.environ.get("MOONVIT_WEIGHTS_DIR", "/weights")
RUN = (
    os.environ.get("MOONVIT_HF_PARITY", "0") == "1"
    and os.path.isdir(WEIGHTS)
)

pytestmark = pytest.mark.skipif(
    not RUN,
    reason="set MOONVIT_HF_PARITY=1 and ensure /weights is mounted (container-only)",
)

GRIDS = [(36, 46), (24, 32), (12, 18), (40, 52)]
TOL = 1e-3


def _load_vision_sd():
    from safetensors.torch import load_file
    sd = {}
    for fn in sorted(os.listdir(WEIGHTS)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(WEIGHTS, fn)))
    return {k: v for k, v in sd.items() if k.startswith("vision_model.") or k.startswith("mlp1.")}


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoModel
    from lrai_locate_anything.trtllm_prod.modeling_moonvit import MoonViTVisionModel
    torch.manual_seed(42)
    vision_sd = _load_vision_sd()
    model = MoonViTVisionModel.from_moonvit_state_dict(
        vision_sd, grid_h=36, grid_w=46, use_bf16=False
    ).eval()
    hf_full = AutoModel.from_pretrained(WEIGHTS, trust_remote_code=True, torch_dtype=torch.float32).eval()
    hf_vision = hf_full.vision_model
    freqs_cis = hf_vision.encoder.rope_2d.freqs_cis
    freqs_source = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1).to(torch.float32)
    model.install_pt_attention_swap(freqs_source=freqs_source)
    return model, hf_vision, hf_full


@pytest.mark.parametrize("h,w", GRIDS)
def test_hf_parity_per_grid(setup, h, w):
    model, hf_vision, hf_full = setup
    model.set_grid(h, w)
    L_pre = h * w
    torch.manual_seed(42)
    px = torch.randn(L_pre, 3, 14, 14, dtype=torch.float32)
    grid_hws = torch.tensor([[h, w]], dtype=torch.long)
    with torch.no_grad():
        ours = model(px)
        hf_out = hf_vision(px, grid_hws)
        hf_post = hf_out[0] if isinstance(hf_out, list) else hf_out
        hf_projected = hf_full.mlp1(hf_post)
    assert ours.shape == hf_projected.shape, (
        f"shape mismatch at ({h},{w}): ours={list(ours.shape)} hf={list(hf_projected.shape)}"
    )
    diff = (ours - hf_projected).abs().max().item()
    assert diff < TOL, (
        f"max-abs-diff {diff:.4e} >= {TOL:.4e} at grid ({h},{w}) — parity regression"
    )
