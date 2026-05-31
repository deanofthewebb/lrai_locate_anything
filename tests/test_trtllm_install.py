"""Tests for trtllm.install — pure version-probing logic."""
import pytest

from lrai_locate_anything.trtllm.install import probe_compatible_versions


class TestProbeVersions:
    def test_torch_2_7(self):
        v = probe_compatible_versions("2.7.0")
        assert v == ["0.22.0", "0.21.0"]

    def test_torch_2_7_with_cuda_suffix(self):
        v = probe_compatible_versions("2.7.0+cu124")
        assert v == ["0.22.0", "0.21.0"]

    def test_torch_2_6(self):
        v = probe_compatible_versions("2.6.0")
        assert v == ["0.21.0", "0.20.0"]

    def test_torch_2_5(self):
        v = probe_compatible_versions("2.5.0")
        assert v == ["0.19.0", "0.18.0"]

    def test_torch_2_4(self):
        v = probe_compatible_versions("2.4.0")
        assert v == ["0.17.0", "0.16.0"]

    def test_old_torch_fallback(self):
        v = probe_compatible_versions("2.0.0")
        # Falls into the "else" branch
        assert v == ["0.17.0"]

    def test_returns_list_of_strings(self):
        v = probe_compatible_versions("2.6.0")
        assert all(isinstance(x, str) for x in v)
        # All should look like semver-ish version strings
        for x in v:
            parts = x.split(".")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)
