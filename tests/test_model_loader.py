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
    """The vendored LocateAnythingImageProcessor is Kimi-VL-style, NOT Qwen2-VL.
    Lock by raising in_token_limit (not min/max_pixels which don't exist on it).
    """

    def _proc(self, P=14, mk=(2, 2), in_token_limit=4096):
        class _IP:
            patch_size = P
            merge_kernel_size = mk
        ip = _IP()
        ip.in_token_limit = in_token_limit
        class _Proc:
            image_processor = ip
        return _Proc()

    def test_raises_in_token_limit_when_below_grid(self):
        """If in_token_limit is below grid_h * grid_w, the lock must raise it."""
        from lrai_locate_anything.model_loader import lock_processor_resolution
        # grid 36x46 = 1656 tokens; default in_token_limit=1024 -> must raise to 1656
        proc = self._proc(in_token_limit=1024)
        lock_processor_resolution(proc, eng_img_w=46 * 14, eng_img_h=36 * 14, verbose=False)
        assert proc.image_processor.in_token_limit == 36 * 46

    def test_leaves_in_token_limit_when_already_sufficient(self):
        from lrai_locate_anything.model_loader import lock_processor_resolution
        proc = self._proc(in_token_limit=8192)
        lock_processor_resolution(proc, eng_img_w=46 * 14, eng_img_h=36 * 14, verbose=False)
        assert proc.image_processor.in_token_limit == 8192  # unchanged

    def test_raises_on_non_mk_multiple_resolution(self):
        """Engine resolutions that aren't multiples of merge_kernel_size*patch_size
        would be silently snap-resized by the processor; lock must REFUSE."""
        from lrai_locate_anything.model_loader import lock_processor_resolution
        # patch_size=14, mk=(2,2) → must be multiple of 28. 504x645 fails on width.
        proc = self._proc()
        with pytest.raises(RuntimeError, match="not a multiple"):
            lock_processor_resolution(proc, eng_img_w=645, eng_img_h=504, verbose=False)

    def test_raises_on_pos_emb_ceiling_overflow(self):
        from lrai_locate_anything.model_loader import lock_processor_resolution
        # grid >=512 along either axis exceeds the processor's pos-emb ceiling
        proc = self._proc()
        with pytest.raises(RuntimeError, match="exceeds processor pos-emb ceiling"):
            lock_processor_resolution(proc, eng_img_w=512 * 14, eng_img_h=28, verbose=False)

    def test_warns_when_attrs_missing(self):
        """Processor that lacks patch_size/merge_kernel_size/in_token_limit should
        warn and return (not crash)."""
        from lrai_locate_anything.model_loader import lock_processor_resolution

        class _IP:
            pass
        class _Proc:
            image_processor = _IP()
        lock_processor_resolution(_Proc(), 644, 504, verbose=False)  # no raise
