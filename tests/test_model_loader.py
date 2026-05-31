"""Tests for model_loader helpers — normalize_image_grid_hws and lock_processor_resolution."""
import pytest
import numpy as np

torch = pytest.importorskip("torch")


class TestNormalizeImageGridHWS:
    """Critical: image_grid_hws from the processor is sometimes numpy with
    np.int64 dtype, which torch.zeros(..., dtype=...) inside the vendored
    modeling_vit.forward rejects. The helper normalizes it to torch int32."""

    def test_numpy_input_converts_to_torch_int32(self):
        from lrai_locate_anything.model_loader import normalize_image_grid_hws
        inputs = {
            "pixel_values": torch.zeros(1, 3, 14, 14),
            "image_grid_hws": np.array([[36, 46]], dtype=np.int64),
        }
        out = normalize_image_grid_hws(inputs)
        assert torch.is_tensor(out["image_grid_hws"])
        assert out["image_grid_hws"].dtype == torch.int32

    def test_torch_input_cast_to_int32(self):
        from lrai_locate_anything.model_loader import normalize_image_grid_hws
        inputs = {
            "pixel_values": torch.zeros(1, 3, 14, 14),
            "image_grid_hws": torch.tensor([[36, 46]], dtype=torch.int64),
        }
        out = normalize_image_grid_hws(inputs)
        assert out["image_grid_hws"].dtype == torch.int32

    def test_missing_key_returns_inputs_unchanged(self):
        from lrai_locate_anything.model_loader import normalize_image_grid_hws
        inputs = {"pixel_values": torch.zeros(1, 3, 14, 14)}
        out = normalize_image_grid_hws(inputs)
        assert out is inputs

    def test_values_preserved(self):
        from lrai_locate_anything.model_loader import normalize_image_grid_hws
        inputs = {
            "pixel_values": torch.zeros(1, 3, 14, 14),
            "image_grid_hws": np.array([[18, 32]], dtype=np.int64),
        }
        normalize_image_grid_hws(inputs)
        assert inputs["image_grid_hws"].tolist() == [[18, 32]]


class TestLockProcessorResolution:
    """The processor's smart_resize ignores our pre-resize unless min_pixels and
    max_pixels are locked. This was a high-impact subtle bug."""

    def test_sets_min_and_max_pixels(self):
        from lrai_locate_anything.model_loader import lock_processor_resolution

        class _FakeImageProc:
            min_pixels = 1024
            max_pixels = 200704

        class _FakeProc:
            image_processor = _FakeImageProc()

        proc = _FakeProc()
        lock_processor_resolution(proc, eng_img_w=644, eng_img_h=504, verbose=False)
        target = 644 * 504
        assert proc.image_processor.min_pixels == target
        assert proc.image_processor.max_pixels == target

    def test_silent_when_attrs_missing(self):
        """If the processor doesn't have min/max_pixels, the lock should warn but
        not crash."""
        from lrai_locate_anything.model_loader import lock_processor_resolution

        class _FakeImageProc:
            pass  # no min_pixels / max_pixels

        class _FakeProc:
            image_processor = _FakeImageProc()

        # Should not raise
        lock_processor_resolution(_FakeProc(), 644, 504, verbose=False)
