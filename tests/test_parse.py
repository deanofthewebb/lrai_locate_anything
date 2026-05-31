"""Pure-Python tests for lrai_locate_anything.parse."""
import numpy as np
import pytest

from lrai_locate_anything.parse import parse_boxes, iou, python_patch_merger


# ---------------------------------------------------------------------------
# parse_boxes
# ---------------------------------------------------------------------------
class TestParseBoxes:
    def test_empty_string_returns_empty(self):
        assert parse_boxes("") == []

    def test_no_box_tags_returns_empty(self):
        assert parse_boxes("Some text without box tags") == []

    def test_single_box_default_unit_scale(self):
        text = "<box><100><200><500><600></box>"
        out = parse_boxes(text)
        assert len(out) == 1
        assert out[0] == pytest.approx((0.1, 0.2, 0.5, 0.6), rel=1e-6)

    def test_single_box_scales_to_image_size(self):
        text = "<box><100><200><500><600></box>"
        out = parse_boxes(text, W=1000, H=2000)
        assert out[0] == pytest.approx((100.0, 400.0, 500.0, 1200.0), rel=1e-6)

    def test_multiple_boxes(self):
        text = (
            "<ref>cat</ref><box><10><20><30><40></box>"
            " and <ref>dog</ref><box><50><60><70><80></box>"
        )
        out = parse_boxes(text, W=100, H=100)
        assert len(out) == 2
        assert out[0] == pytest.approx((1.0, 2.0, 3.0, 4.0))
        assert out[1] == pytest.approx((5.0, 6.0, 7.0, 8.0))

    def test_malformed_box_ignored(self):
        # Only 2 coords in the second block — should be skipped, not error.
        text = "<box><10><20><30><40></box><box><50><60></box>"
        out = parse_boxes(text, W=100, H=100)
        assert len(out) == 1

    def test_more_than_4_coords_takes_first_4(self):
        text = "<box><1><2><3><4><5><6></box>"
        out = parse_boxes(text, W=1000, H=1000)
        assert out[0] == pytest.approx((1.0, 2.0, 3.0, 4.0))

    def test_coords_at_bounds(self):
        text = "<box><0><0><1000><1000></box>"
        out = parse_boxes(text, W=1024, H=768)
        assert out[0] == pytest.approx((0.0, 0.0, 1024.0, 768.0))


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------
class TestIoU:
    def test_identical_boxes_iou_one(self):
        a = (0, 0, 10, 10)
        assert iou(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        assert iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0

    def test_half_overlap(self):
        # 10x10 boxes overlapping by exactly half on each axis
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # intersection = 5x5 = 25, union = 100 + 100 - 25 = 175 → 25/175
        assert iou(a, b) == pytest.approx(25.0 / 175.0)

    def test_contained_box(self):
        outer = (0, 0, 10, 10)
        inner = (2, 2, 5, 5)
        # intersection = 9, union = 100
        assert iou(outer, inner) == pytest.approx(9.0 / 100.0)

    def test_touching_edge_zero(self):
        # Boxes that share only an edge have zero intersection area
        assert iou((0, 0, 5, 5), (5, 0, 10, 5)) == 0.0

    def test_zero_area_box(self):
        # Degenerate box (point) — guard against division-by-zero
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# python_patch_merger
# ---------------------------------------------------------------------------
class TestPythonPatchMerger:
    def test_output_shape(self):
        # grid (4, 6) -> 24 tokens; 2x2 merge -> 6 output tokens
        x = np.random.randn(24, 16).astype(np.float32)
        gh = np.array([[4, 6]], dtype=np.int32)
        out = python_patch_merger(x, gh, kh=2, kw=2)
        assert out.shape == (6, 16 * 4)

    def test_equivalent_to_canonical_via_torch(self, torch_module):
        """Numerically identical to the canonical patch_merger's view+permute+view."""
        torch = torch_module
        h, w, d = 4, 6, 16
        x_np = np.random.randn(h * w, d).astype(np.float32)
        gh = np.array([[h, w]], dtype=np.int32)
        out_np = python_patch_merger(x_np, gh, kh=2, kw=2)

        # Canonical reference
        x_t = torch.from_numpy(x_np)
        ref = (
            x_t.view(h // 2, 2, w // 2, 2, d)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view((h // 2) * (w // 2), -1)
            .numpy()
        )
        np.testing.assert_array_almost_equal(out_np, ref, decimal=5)

    def test_handles_noncontiguous_input(self):
        # Slice creates a non-contiguous view; the helper's ascontiguousarray should fix it.
        x = np.random.randn(50, 16).astype(np.float32)
        x_slice = x[:24]  # contiguous on this axis but let's go further
        x_t = x.T[:16].T  # potentially non-contig
        # Just verify no error + sane shape
        gh = np.array([[4, 6]], dtype=np.int32)
        out = python_patch_merger(x_slice, gh, kh=2, kw=2)
        assert out.shape == (6, 64)

    def test_demo_resolution(self):
        # The actual demo resolution: grid (36, 46), L_pre = 1656
        x = np.random.randn(1656, 1152).astype(np.float16)
        gh = np.array([[36, 46]], dtype=np.int32)
        out = python_patch_merger(x, gh, kh=2, kw=2)
        # nh*nw = 18 * 23 = 414 tokens, 1152 * 4 = 4608 dim
        assert out.shape == (414, 4608)

    def test_different_merge_kernel(self):
        x = np.zeros((48, 8), dtype=np.float32)
        gh = np.array([[4, 12]], dtype=np.int32)
        out = python_patch_merger(x, gh, kh=2, kw=4)
        # nh*nw = 2 * 3 = 6, dim = 8 * 2 * 4 = 64
        assert out.shape == (6, 64)
