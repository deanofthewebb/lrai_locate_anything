"""Tests for _ensure_lm_head_tied — THE fix for universal mode-collapse.

The vendored LocateAnythingForConditionalGeneration.__init__ skips post_init(),
so transformers never calls tie_weights() automatically. For Qwen2.5-3B with
tie_word_embeddings=True, lm_head.weight is supposed to be tied to
embed_tokens.weight at load time — without it, lm_head stays at random init
and every generation path produces <ref>$$$$$<ref>$$$$$ mode-collapse.

_ensure_lm_head_tied() detects this and fixes it (with model.tie_weights() or
direct shared-storage assignment as fallback).
"""
import pytest

torch = pytest.importorskip("torch")


def _make_fake_model(tied: bool, tie_flag: bool = True, vocab: int = 1000, dim: int = 64,
                      outer_tie_works: bool = True, inner_tie_works: bool = True):
    """Build a minimal model with .language_model.model.embed_tokens + .language_model.lm_head.

    outer_tie_works/inner_tie_works toggle whether the model.tie_weights() and
    language_model.tie_weights() calls actually do anything — used to exercise
    the cascade fallback. Real LocateAnythingForConditionalGeneration has
    outer_tie_works=False because post_init is skipped.
    """
    embed = torch.nn.Embedding(vocab, dim)
    head = torch.nn.Linear(dim, vocab, bias=False)
    torch.nn.init.normal_(embed.weight, mean=0.0, std=0.05)
    torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
    if tied:
        head.weight = embed.weight

    class _Cfg:
        class text_config:
            pass
    cfg = _Cfg()
    cfg.text_config.tie_word_embeddings = tie_flag

    class _LMMain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = embed
    lm_main = _LMMain()

    class _LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = lm_main
            self.lm_head = head
        def tie_weights(self):
            if inner_tie_works:
                self.lm_head.weight = self.model.embed_tokens.weight
    lm = _LM()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = lm
            self.config = cfg
        def tie_weights(self):
            if outer_tie_works:
                self.language_model.lm_head.weight = self.language_model.model.embed_tokens.weight

    return _Model()


class TestEnsureLmHeadTied:
    def test_returns_true_when_needs_rescue(self):
        """Untied model with tie_flag=True must be detected as rescued."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=True)
        rescued = _ensure_lm_head_tied(m, verbose=False)
        assert rescued is True

    def test_returns_false_when_already_tied(self):
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=True, tie_flag=True)
        rescued = _ensure_lm_head_tied(m, verbose=False)
        assert rescued is False

    def test_returns_false_when_tie_flag_disabled(self):
        """If tie_word_embeddings=False, untied is expected — don't rescue."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=False)
        rescued = _ensure_lm_head_tied(m, verbose=False)
        assert rescued is False

    def test_actually_ties_weights(self):
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=True)
        e_ptr_before = m.language_model.model.embed_tokens.weight.data_ptr()
        h_ptr_before = m.language_model.lm_head.weight.data_ptr()
        assert e_ptr_before != h_ptr_before  # premise: not tied

        _ensure_lm_head_tied(m, verbose=False)

        e_ptr_after = m.language_model.model.embed_tokens.weight.data_ptr()
        h_ptr_after = m.language_model.lm_head.weight.data_ptr()
        assert e_ptr_after == h_ptr_after, "lm_head.weight must share storage with embed_tokens.weight"

    def test_handles_missing_language_model(self):
        """Don't crash on a model without .language_model."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        class _M(torch.nn.Module):
            pass
        m = _M()
        m.config = type("Cfg", (), {})()
        assert _ensure_lm_head_tied(m, verbose=False) is False  # no-op, no crash

    def test_tie_weights_fallback_to_direct_assignment(self):
        """If model.tie_weights() raises, fall back to head.weight = embed.weight."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=True)
        def _broken():
            raise RuntimeError("simulated tie_weights failure")
        m.tie_weights = _broken
        rescued = _ensure_lm_head_tied(m, verbose=False)
        assert rescued is True
        assert (m.language_model.model.embed_tokens.weight.data_ptr()
                == m.language_model.lm_head.weight.data_ptr())

    def test_cascade_uses_nuclear_when_outer_and_inner_fail(self):
        """When both outer model.tie_weights() AND inner language_model.tie_weights()
        are no-ops (i.e. the bug we have in production), the cascade should still
        succeed via the direct-assignment or nuclear-replacement step."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=True,
                              outer_tie_works=False, inner_tie_works=False)
        rescued = _ensure_lm_head_tied(m, verbose=False)
        assert rescued is True
        # Must end with shared storage (tied), NOT just equal-by-value
        e_ptr = m.language_model.model.embed_tokens.weight.data_ptr()
        h_ptr = m.language_model.lm_head.weight.data_ptr()
        assert e_ptr == h_ptr, "cascade did not produce shared-storage tying"

    def test_no_silent_fallback_when_inner_works(self):
        """Per user direction: NO silent fallbacks to random init. When the cascade
        succeeds, ptrs must be shared (not just equal-by-value). Tests the
        inner_tie path which is what should fire for production models."""
        from lrai_locate_anything.model_loader import _ensure_lm_head_tied
        m = _make_fake_model(tied=False, tie_flag=True,
                              outer_tie_works=False, inner_tie_works=True)
        _ensure_lm_head_tied(m, verbose=False)
        # Storage MUST be shared, not just value-equal
        assert (m.language_model.model.embed_tokens.weight.data_ptr()
                == m.language_model.lm_head.weight.data_ptr())


class TestWipeStaleArtifacts:
    """LocateAnythingRunner._wipe_stale_artifacts must clear any files under
    ONNX_DIR + TRT_DIR when called — used after lm_head rescue invalidates
    engines that were built from the broken model in a prior session."""

    def test_wipes_existing_files(self, tmp_path, monkeypatch):
        # Patch ONNX_DIR + TRT_DIR to live under tmp_path so we don't touch real cache
        from lrai_locate_anything import orchestrator
        onnx = tmp_path / "onnx"
        trt = tmp_path / "trt"
        onnx.mkdir()
        trt.mkdir()
        (onnx / "vision.onnx").write_bytes(b"stale-onnx")
        (trt / "llm_prefill.engine").write_bytes(b"stale-engine")
        monkeypatch.setattr(orchestrator, "ONNX_DIR", onnx)
        monkeypatch.setattr(orchestrator, "TRT_DIR", trt)

        # Build a minimal runner-shaped object that can call _wipe_stale_artifacts
        class _R:
            _wipe_stale_artifacts = orchestrator.LocateAnythingRunner._wipe_stale_artifacts
        _R()._wipe_stale_artifacts(reason="test")

        assert not (onnx / "vision.onnx").exists()
        assert not (trt / "llm_prefill.engine").exists()
        # Directories themselves should still exist (re-created)
        assert onnx.exists() and trt.exists()
