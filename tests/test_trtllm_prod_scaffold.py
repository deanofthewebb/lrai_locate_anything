"""Scaffolding tests for lrai_locate_anything.trtllm_prod.

Validates that every public entrypoint is importable and raises
NotImplementedError. Replace with real behavioural tests as each
module is filled in.
"""
from pathlib import Path

import pytest


def test_imports():
    from lrai_locate_anything.trtllm_prod import (
        convert_locateanything_checkpoint,
        build_llm_engine,
        MoonViTAdapter,
        LocateAnythingTRTLLMRunner,
    )
    assert callable(convert_locateanything_checkpoint)
    assert callable(build_llm_engine)
    assert isinstance(MoonViTAdapter, type)
    assert isinstance(LocateAnythingTRTLLMRunner, type)


def test_convert_raises():
    from lrai_locate_anything.trtllm_prod import convert_locateanything_checkpoint
    with pytest.raises(NotImplementedError):
        convert_locateanything_checkpoint(Path("/tmp/in"), Path("/tmp/out"))


def test_build_raises():
    from lrai_locate_anything.trtllm_prod import build_llm_engine
    with pytest.raises(NotImplementedError):
        build_llm_engine(Path("/tmp/ckpt"), Path("/tmp/llm.engine"))


def test_moonvit_adapter_init_raises():
    from lrai_locate_anything.trtllm_prod import MoonViTAdapter
    with pytest.raises(NotImplementedError):
        MoonViTAdapter(Path("/tmp/vision_proj.engine"))


def test_moonvit_adapter_forward_raises():
    from lrai_locate_anything.trtllm_prod.moonvit_adapter import MoonViTAdapter
    # Bypass __init__ so we can exercise forward independently.
    adapter = MoonViTAdapter.__new__(MoonViTAdapter)
    with pytest.raises(NotImplementedError):
        adapter.forward(None)


def test_runner_init_raises():
    from lrai_locate_anything.trtllm_prod import LocateAnythingTRTLLMRunner
    with pytest.raises(NotImplementedError):
        LocateAnythingTRTLLMRunner(
            Path("/tmp/llm.engine"),
            Path("/tmp/vision_proj.engine"),
            Path("/tmp/hf"),
        )


def test_runner_detect_raises():
    from lrai_locate_anything.trtllm_prod.runner import LocateAnythingTRTLLMRunner
    runner = LocateAnythingTRTLLMRunner.__new__(LocateAnythingTRTLLMRunner)
    with pytest.raises(NotImplementedError):
        runner.detect(None, "locate the car")
