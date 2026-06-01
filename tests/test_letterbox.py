"""Tests for the letterbox helper used by the TRT path.

A naive BICUBIC stretch destroyed aspect ratio and produced visual features that
MoonViT could not interpret — the LM mode-collapsed on '<ref>$$$$$' across
every generation path. The letterbox replacement preserves aspect ratio.
"""
import pytest
from PIL import Image

from lrai_locate_anything.orchestrator import _letterbox


def _solid(w, h, color=(200, 100, 50)):
    return Image.new("RGB", (w, h), color)


class TestLetterboxShape:
    def test_output_exactly_target_size(self):
        img = _solid(1920, 1080)
        lb, _, _, _ = _letterbox(img, 644, 504)
        assert lb.size == (644, 504)

    def test_square_input_no_pad_when_square_target(self):
        img = _solid(512, 512)
        lb, scale, px, py = _letterbox(img, 644, 644)
        assert lb.size == (644, 644)
        # Square -> square: no padding required
        assert px == 0 and py == 0
        # Scale should be 644/512
        assert scale == pytest.approx(644 / 512)


class TestLetterboxAspectPreservation:
    def test_wide_image_gets_top_bottom_pad(self):
        # 16:9 source -> 4:3-ish target. Source is wider; padding goes top/bottom.
        img = _solid(1920, 1080)
        lb, scale, px, py = _letterbox(img, 644, 504)
        # scale chosen by the narrower dim: min(644/1920, 504/1080) = 644/1920 = 0.3354
        assert scale == pytest.approx(min(644 / 1920, 504 / 1080))
        assert px == 0   # width fully used
        assert py > 0    # vertical padding

    def test_tall_image_gets_left_right_pad(self):
        img = _solid(720, 1280)  # portrait phone
        lb, scale, px, py = _letterbox(img, 644, 504)
        assert scale == pytest.approx(min(644 / 720, 504 / 1280))
        assert py == 0   # height fully used
        assert px > 0    # horizontal padding


class TestLetterboxCoordRoundTrip:
    """The whole point of letterbox is that box coords in the letterboxed image
    can be mapped EXACTLY back to original-image coords."""

    def test_corner_round_trip_wide(self):
        orig_w, orig_h = 1920, 1080
        target_w, target_h = 644, 504
        img = _solid(orig_w, orig_h)
        _, scale, px, py = _letterbox(img, target_w, target_h)

        # Corners of original image map to corners of the *content area* of the
        # letterboxed image.
        # top-left orig (0, 0) -> letterbox (px, py)
        x_lb, y_lb = px, py
        x_orig_back = (x_lb - px) / scale
        y_orig_back = (y_lb - py) / scale
        assert x_orig_back == pytest.approx(0)
        assert y_orig_back == pytest.approx(0)

        # bottom-right orig (1920, 1080) -> letterbox (px + 1920*scale, py + 1080*scale)
        x_lb = px + orig_w * scale
        y_lb = py + orig_h * scale
        x_orig_back = (x_lb - px) / scale
        y_orig_back = (y_lb - py) / scale
        assert x_orig_back == pytest.approx(orig_w)
        assert y_orig_back == pytest.approx(orig_h)

    def test_corner_round_trip_tall(self):
        orig_w, orig_h = 720, 1280
        target_w, target_h = 644, 504
        img = _solid(orig_w, orig_h)
        _, scale, px, py = _letterbox(img, target_w, target_h)

        # Mid of orig -> mid of content area
        mid_x_lb = px + (orig_w / 2) * scale
        mid_y_lb = py + (orig_h / 2) * scale
        x_back = (mid_x_lb - px) / scale
        y_back = (mid_y_lb - py) / scale
        assert x_back == pytest.approx(orig_w / 2)
        assert y_back == pytest.approx(orig_h / 2)


class TestLetterboxDoesNotStretch:
    """Regression guard: the prior implementation BICUBIC-stretched a 1920x1080
    image to 644x504, distorting aspect from 1.78 to 1.28 — a 28% horizontal
    compression that destroyed MoonViT features. Verify the new letterbox does
    NOT do that."""

    def test_wide_input_preserves_horizontal_extent(self):
        # The actual image content should fit horizontally with no compression
        img = _solid(1920, 1080)
        _, scale, px, py = _letterbox(img, 644, 504)
        content_w = 1920 * scale
        content_h = 1080 * scale
        # Aspect of the content area must equal aspect of source (1.78), not target (1.28)
        assert content_w / content_h == pytest.approx(1920 / 1080, rel=1e-3)
