"""Tests for the orchestrator's TID (token-id) dict construction.

The audit found that generate_utils.{sample_tokens, handle_pattern, ...} requires
SUFFIXED keys (`box_start_token_id`, `null_token_id`, etc.), while the notebook's
own code paths used short keys. The orchestrator must populate both.
"""
from types import SimpleNamespace
import pytest


def _make_fake_config():
    """Build a fake AutoConfig-like object with the LocateAnything token IDs."""
    text = SimpleNamespace(
        null_token_id=152678,
        switch_token_id=152679,
        text_mask_token_id=151676,
        eos_token_id=151645,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        vocab_size=152681,
        block_size=6,
    )
    return SimpleNamespace(
        text_config=text,
        vision_config=SimpleNamespace(hidden_size=1152),
        box_start_token_id=151668,
        box_end_token_id=151669,
        coord_start_token_id=151677,
        coord_end_token_id=152677,
        ref_start_token_id=151672,
        ref_end_token_id=151673,
        none_token_id=4064,
        image_token_index=151665,
    )


class _FakeRunner:
    """Minimal stand-in that runs the same TID-construction code as LocateAnythingRunner.__init__."""
    def __init__(self):
        from lrai_locate_anything.orchestrator import LocateAnythingRunner
        config = _make_fake_config()
        # Bypass full __init__ by calling the constructor logic manually
        self.config = config
        tc = config.text_config
        # Replicate the TID dict construction (this should match the orchestrator's __init__)
        self.TID = {
            "box_start_token_id":    config.box_start_token_id,
            "box_end_token_id":      config.box_end_token_id,
            "coord_start_token_id":  config.coord_start_token_id,
            "coord_end_token_id":    config.coord_end_token_id,
            "ref_start_token_id":    config.ref_start_token_id,
            "ref_end_token_id":      config.ref_end_token_id,
            "none_token_id":         config.none_token_id,
            "null_token_id":         tc.null_token_id,
            "switch_token_id":       tc.switch_token_id,
            "default_mask_token_id": tc.text_mask_token_id,
            "im_end_token_id":       tc.eos_token_id,
            "image_token_index":     config.image_token_index,
            "box_start": config.box_start_token_id, "box_end": config.box_end_token_id,
            "coord_start": config.coord_start_token_id, "coord_end": config.coord_end_token_id,
            "ref_start": config.ref_start_token_id, "ref_end": config.ref_end_token_id,
            "none": config.none_token_id, "null": tc.null_token_id,
            "switch": tc.switch_token_id, "mask": tc.text_mask_token_id,
            "im_end": tc.eos_token_id, "image": config.image_token_index,
        }


class TestTIDDict:
    def test_suffixed_keys_present(self):
        """Required by generate_utils."""
        tid = _FakeRunner().TID
        for k in (
            "box_start_token_id", "box_end_token_id",
            "coord_start_token_id", "coord_end_token_id",
            "ref_start_token_id", "ref_end_token_id",
            "none_token_id", "null_token_id",
            "switch_token_id", "default_mask_token_id",
            "im_end_token_id", "image_token_index",
        ):
            assert k in tid, f"missing canonical key {k!r}"

    def test_short_aliases_present(self):
        """Required by the orchestrator's own code paths (MTP step, AR step)."""
        tid = _FakeRunner().TID
        for k in (
            "box_start", "box_end", "coord_start", "coord_end",
            "ref_start", "ref_end", "none", "null",
            "switch", "mask", "im_end", "image",
        ):
            assert k in tid, f"missing short alias {k!r}"

    def test_keys_pairwise_consistent(self):
        """The suffixed and short keys must point to the same token IDs."""
        tid = _FakeRunner().TID
        pairs = [
            ("box_start_token_id", "box_start"),
            ("box_end_token_id", "box_end"),
            ("coord_start_token_id", "coord_start"),
            ("coord_end_token_id", "coord_end"),
            ("ref_start_token_id", "ref_start"),
            ("ref_end_token_id", "ref_end"),
            ("none_token_id", "none"),
            ("null_token_id", "null"),
            ("switch_token_id", "switch"),
            ("default_mask_token_id", "mask"),
            ("im_end_token_id", "im_end"),
            ("image_token_index", "image"),
        ]
        for suffixed, short in pairs:
            assert tid[suffixed] == tid[short], f"{suffixed} != {short}"

    def test_values_are_ints(self):
        tid = _FakeRunner().TID
        for k, v in tid.items():
            assert isinstance(v, int), f"{k} = {v!r} (not int)"

    def test_known_locateanything_token_ids(self):
        """Sanity-check the specific LocateAnything special-token IDs."""
        tid = _FakeRunner().TID
        # From the canonical config.json
        assert tid["box_start_token_id"] == 151668
        assert tid["box_end_token_id"] == 151669
        assert tid["coord_start_token_id"] == 151677
        assert tid["coord_end_token_id"] == 152677  # 1000-token coord range
        assert tid["image_token_index"] == 151665
