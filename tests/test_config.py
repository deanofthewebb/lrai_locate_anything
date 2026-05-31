"""Tests for config + environment detection."""
import os
from pathlib import Path
import pytest


class TestPaths:
    def test_workdir_exists(self):
        from lrai_locate_anything.config import WORK, ONNX_DIR, TRT_DIR, WEIGHTS_DIR
        for p in (WORK, ONNX_DIR, TRT_DIR, WEIGHTS_DIR):
            assert p.exists()
            assert p.is_dir()

    def test_constants(self):
        from lrai_locate_anything.config import MODEL_ID, ENG_PATCH_SIZE, ENG_MERGE_KH, ENG_MERGE_KW
        assert MODEL_ID == "nvidia/LocateAnything-3B"
        assert ENG_PATCH_SIZE == 14
        assert ENG_MERGE_KH == 2
        assert ENG_MERGE_KW == 2

    def test_env_override_workdir(self, tmp_path, monkeypatch):
        """LRAI_WORKDIR env var should override the default workdir."""
        monkeypatch.setenv("LRAI_WORKDIR", str(tmp_path))
        # Re-execute the workdir-detection logic
        from lrai_locate_anything.config import _default_workdir
        assert _default_workdir() == tmp_path


class TestGPUDetection:
    def test_vram_returns_float(self):
        from lrai_locate_anything.config import gpu_vram_gb
        v = gpu_vram_gb()
        assert isinstance(v, float)
        assert v >= 0.0

    def test_enable_llm_trt_threshold(self):
        """Returns True iff VRAM ≥ 22 GB."""
        from lrai_locate_anything.config import enable_llm_trt, gpu_vram_gb
        v = gpu_vram_gb()
        if v >= 22.0:
            assert enable_llm_trt() is True
        else:
            assert enable_llm_trt() is False
