"""Tests for pipeline function signatures + edge cases.

Heavy end-to-end pipeline runs are gated behind @pytest.mark.heavy.
"""
import inspect
import pytest


class TestPipelineSignatures:
    def test_run_image_signature(self):
        from lrai_locate_anything.pipelines import run_image
        sig = inspect.signature(run_image)
        params = list(sig.parameters.keys())
        assert "runner" in params
        assert "image" in params
        assert "prompt" in params
        # Defaults are documented in the signature
        assert sig.parameters["prompt"].default is not inspect.Parameter.empty

    def test_run_video_signature(self):
        from lrai_locate_anything.pipelines import run_video
        sig = inspect.signature(run_video)
        for k in ("runner", "input_path", "output_path", "prompt", "max_frames"):
            assert k in sig.parameters

    def test_run_compare_signature(self):
        from lrai_locate_anything.pipelines import run_compare
        sig = inspect.signature(run_compare)
        for k in ("runner", "input_path", "output_path", "prompt", "max_frames", "include_pytorch"):
            assert k in sig.parameters

    def test_run_compare_default_includes_pytorch(self):
        from lrai_locate_anything.pipelines import run_compare
        sig = inspect.signature(run_compare)
        assert sig.parameters["include_pytorch"].default is True


class TestPipelineMissingDeps:
    def test_run_video_requires_cv2(self, monkeypatch):
        """If opencv-python is missing, run_video should raise ImportError, not a
        confusing AttributeError mid-execution."""
        import lrai_locate_anything.pipelines as p
        monkeypatch.setattr(p, "cv2", None)
        with pytest.raises(ImportError):
            p.run_video(runner=None, input_path="x", output_path="y")

    def test_run_compare_requires_cv2(self, monkeypatch):
        import lrai_locate_anything.pipelines as p
        monkeypatch.setattr(p, "cv2", None)
        with pytest.raises(ImportError):
            p.run_compare(runner=None, input_path="x", output_path="y")
