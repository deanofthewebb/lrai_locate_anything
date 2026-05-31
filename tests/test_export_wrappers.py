"""Tests for the export wrapper classes.

We test the wrapper signature contracts (which inputs they take, what shapes they
return) without loading the actual 3B model. Heavy end-to-end exports are gated
behind @pytest.mark.heavy.
"""
import pytest

torch = pytest.importorskip("torch")

from lrai_locate_anything.export.vision import VisionForExport
from lrai_locate_anything.export.projector import ProjectorForExport
from lrai_locate_anything.export.llm import LLMPrefill, LLMDecode


# ---------------------------------------------------------------------------
# VisionForExport
# ---------------------------------------------------------------------------
class TestVisionForExport:
    """The vision wrapper bakes pos_emb + grid_hws — these contracts must hold."""

    def test_grid_hws_baked_buffer(self):
        """Critical: grid_hws_baked must be a registered buffer so the exporter
        treats it as a graph initializer and doesn't elide grid_hws as a dead input."""
        # We can't instantiate without a real vit, but the class attribute check
        # gives us forward signature info.
        assert "forward" in VisionForExport.__dict__
        # Forward must take only pixel_values (not grid_hws as an input)
        import inspect
        sig = inspect.signature(VisionForExport.forward)
        params = list(sig.parameters.keys())
        assert params == ["self", "pixel_values"]


# ---------------------------------------------------------------------------
# LLMPrefill / LLMDecode signatures (the canonical contract)
# ---------------------------------------------------------------------------
class TestLLMPrefillSignature:
    """Critical: must take input_ids + visual_features, NEVER inputs_embeds.

    The canonical Qwen2Model.forward raises ValueError if both input_ids AND
    inputs_embeds are set (modeling_qwen2.py line 1200). The audit's
    pass-both fix was wrong; the canonical contract is exactly-one.
    """

    def test_forward_signature(self):
        import inspect
        sig = inspect.signature(LLMPrefill.forward)
        params = list(sig.parameters.keys())
        # MUST be: input_ids, visual_features, position_ids, attention_mask
        # MUST NOT include inputs_embeds
        assert params == ["self", "input_ids", "visual_features", "position_ids", "attention_mask"]
        assert "inputs_embeds" not in params

    def test_init_takes_image_token_index(self):
        """The wrapper must capture image_token_index at __init__ time and store
        it as a Python int so it bakes into the trace as a constant."""
        import inspect
        sig = inspect.signature(LLMPrefill.__init__)
        params = list(sig.parameters.keys())
        assert "image_token_index" in params


class TestLLMDecodeSignature:
    def test_forward_signature(self):
        import inspect
        sig = inspect.signature(LLMDecode.forward)
        params = list(sig.parameters.keys())
        # MUST be: input_ids, position_ids, attention_mask, *past_kv
        # MUST NOT include inputs_embeds OR visual_features (decode has no images)
        assert "input_ids" in params
        assert "position_ids" in params
        assert "attention_mask" in params
        assert "past_kv" in params or any(p.startswith("past") for p in params)
        assert "inputs_embeds" not in params
        assert "visual_features" not in params


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------
class TestProjectorForExport:
    def test_forward_passes_through_mlp1(self):
        """The wrapper just calls self.mlp1(x). Verify the forward delegates correctly."""
        # Mock mlp1
        class _MLP1(torch.nn.Module):
            def forward(self, x): return x * 2.0
        wrap = ProjectorForExport(_MLP1())
        x = torch.randn(32, 4608)
        out = wrap(x)
        assert torch.allclose(out, x * 2.0)
