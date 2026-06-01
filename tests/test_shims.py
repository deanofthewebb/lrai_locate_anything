"""Tests for transformers compatibility shims."""
import pytest


@pytest.fixture
def transformers_available():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        pytest.skip("transformers not installed")
        return False


def _has_tied_weights_keys_api():
    """transformers >=4.55 introduced PreTrainedModel.get_expanded_tied_weights_keys.
    Older versions handle tied weights differently and don't need our shim."""
    try:
        from transformers import modeling_utils
        return hasattr(modeling_utils.PreTrainedModel, "get_expanded_tied_weights_keys")
    except ImportError:
        return False


def _has_mark_tied_api():
    try:
        from transformers import modeling_utils
        return hasattr(modeling_utils.PreTrainedModel, "mark_tied_weights_as_initialized")
    except ImportError:
        return False


class TestInstallShims:
    def test_install_does_not_crash(self, transformers_available):
        """Regardless of transformers version, install must complete without exception."""
        from lrai_locate_anything.shims import install_transformers_shims
        install_transformers_shims(verbose=False)

    def test_install_idempotent(self, transformers_available):
        """Calling install twice must not error and should leave a consistent state."""
        from lrai_locate_anything.shims import install_transformers_shims
        install_transformers_shims(verbose=False)
        install_transformers_shims(verbose=False)  # second call

    @pytest.mark.skipif(not _has_tied_weights_keys_api(),
                         reason="transformers < 4.55 doesn't have get_expanded_tied_weights_keys")
    def test_idempotent_no_double_wrap(self, transformers_available):
        """On transformers >=4.55 the shim should not double-wrap."""
        from lrai_locate_anything.shims import install_transformers_shims
        from transformers import modeling_utils

        install_transformers_shims(verbose=False)
        first_ref = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
        install_transformers_shims(verbose=False)
        second_ref = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
        assert first_ref is second_ref

    @pytest.mark.skipif(not _has_tied_weights_keys_api(),
                         reason="transformers < 4.55 doesn't have get_expanded_tied_weights_keys")
    def test_tied_weights_list_upcast(self, transformers_available):
        """Passing a list-form _tied_weights_keys should get upcast to a dict."""
        from lrai_locate_anything.shims import install_transformers_shims
        from transformers import modeling_utils

        install_transformers_shims(verbose=False)

        class _FakeModel:
            _tied_weights_keys = ["lm_head.weight", "embed.weight"]

        fake = _FakeModel()
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys.__get__(fake)(all_submodels=False)
        assert isinstance(fake._tied_weights_keys, dict)
        assert "lm_head.weight" in fake._tied_weights_keys

    def test_dynamic_cache_has_to_legacy_cache(self, transformers_available):
        """DynamicCache shim should always install successfully on any transformers
        version that has DynamicCache (4.36+)."""
        from lrai_locate_anything.shims import install_transformers_shims
        from transformers.cache_utils import DynamicCache

        install_transformers_shims(verbose=False)
        assert hasattr(DynamicCache, "to_legacy_cache")

    def test_to_legacy_cache_doesnt_truthiness_check_tensors(self, transformers_available):
        """Regression: `getattr(layer, 'keys', None) or getattr(layer, 'key_cache', None)`
        triggers a tensor truthiness check on filled cache layers. Filled tensors must
        be returned without `bool()` being called on them.
        """
        import torch
        from lrai_locate_anything.shims import install_transformers_shims
        from transformers.cache_utils import DynamicCache

        install_transformers_shims(verbose=False)

        # Build a minimal cache with one filled layer
        cache = DynamicCache()
        k = torch.randn(1, 2, 8, 16)   # multi-element tensor — would fail bool()
        v = torch.randn(1, 2, 8, 16)
        cache.update(k, v, layer_idx=0)

        # If the shim uses `tensor or other`, this raises
        # "Boolean value of Tensor with more than one value is ambiguous".
        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple)
        assert len(legacy) >= 1
        assert torch.allclose(legacy[0][0], k)
        assert torch.allclose(legacy[0][1], v)

    @pytest.mark.skipif(not _has_mark_tied_api(),
                         reason="transformers < 4.55 doesn't have mark_tied_weights_as_initialized")
    def test_mark_tied_handles_missing_attr(self, transformers_available):
        """Calling mark_tied_weights_as_initialized on a model that skipped
        post_init shouldn't crash."""
        from lrai_locate_anything.shims import install_transformers_shims
        from transformers import modeling_utils

        install_transformers_shims(verbose=False)

        class _Model:
            pass

        fake = _Model()
        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized.__get__(fake)()
        assert fake.all_tied_weights_keys == {}


class TestRehydrateConfig:
    def test_copies_text_config_fields(self):
        from lrai_locate_anything.shims import rehydrate_config

        class _SubConfig:
            pass

        class _Config:
            def __init__(self):
                self.text_config = _SubConfig()
                self.vision_config = _SubConfig()

        cfg = _Config()
        raw = {
            "text_config": {"rope_theta": 1000000.0, "block_size": 6},
            "vision_config": {"hidden_size": 1152, "patch_size": 14},
        }
        rehydrate_config(cfg, raw)
        assert cfg.text_config.rope_theta == 1000000.0
        assert cfg.text_config.block_size == 6
        assert cfg.vision_config.hidden_size == 1152

    def test_rope_theta_assertion(self):
        """The function asserts text_config.rope_theta exists at the end — that's the
        attribute that crashed the original Qwen2 attention init."""
        from lrai_locate_anything.shims import rehydrate_config

        class _SubConfig: pass
        class _Config:
            def __init__(self):
                self.text_config = _SubConfig()
                self.vision_config = _SubConfig()

        cfg = _Config()
        raw = {"text_config": {"other_field": 1}, "vision_config": {}}
        with pytest.raises(AssertionError):
            rehydrate_config(cfg, raw)
