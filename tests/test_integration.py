"""End-to-end integration tests (require CUDA + full LocateAnything-3B weights).

Run only when LRAI_RUN_HEAVY=1. Downloads ~8 GB of weights on first invocation.
"""
from pathlib import Path
import pytest


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.heavy
class TestEndToEnd:
    @pytest.fixture(scope="class")
    def runner(self):
        """Load + export + build engines once per test class."""
        from lrai_locate_anything import LocateAnythingRunner
        return LocateAnythingRunner.from_pretrained(auto_export=True)

    def test_runner_constructed(self, runner):
        assert runner is not None
        assert runner.grid_h is not None
        assert runner.grid_w is not None

    def test_engines_loaded(self, runner):
        # Vision + projector must always succeed
        assert runner.vit_engine is not None
        assert runner.proj_engine is not None

    def test_detect_returns_boxes(self, runner):
        """End-to-end smoke test — detect on the COCO cats image."""
        import urllib.request, io
        from PIL import Image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
        img.thumbnail((672, 672))
        boxes, text = runner.detect(img, "Detect all cats. Return bounding boxes.")
        assert isinstance(boxes, list)
        assert isinstance(text, str)
        # Two cats expected — but be tolerant since model output varies.
        assert len(boxes) >= 1

    def test_detect_boxes_within_image_bounds(self, runner):
        """Sanity: returned boxes should be within image dimensions."""
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        boxes, _ = runner.detect(img, "Detect any object.")
        for (x1, y1, x2, y2) in boxes:
            assert 0 <= x1 <= 640 + 1
            assert 0 <= y1 <= 480 + 1
            assert 0 <= x2 <= 640 + 1
            assert 0 <= y2 <= 480 + 1
