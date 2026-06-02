"""Fused vision + static patch_merger + mlp1 projector export.

Replaces the 3-stage R&D pipeline:
    export/vision.py (vit_engine)
  + parse.py:python_patch_merger (numpy)
  + export/projector.py (proj_engine)
with one ONNX graph and one TRT engine. Output is bit-equal to the
chained pipeline within fp16 LayerNorm/GELU rounding (parity test asserts
max_abs_diff < 1e-2 on synthetic input).
"""
from __future__ import annotations
from pathlib import Path

import torch

from lrai_locate_anything.export.vision import VisionForExport
from lrai_locate_anything.export.projector import ProjectorForExport
from lrai_locate_anything.export.llm import export_with_external_data


class StaticPatchMerger(torch.nn.Module):
    """Traceable equivalent of parse.py:python_patch_merger.

    grid_h/grid_w/kh/kw are captured as int constants at __init__ so the
    reshape+permute is fully data-independent — no `.tolist()` loop, no
    `List[Tensor]` output (those are why the canonical patch_merger
    can't be exported to ONNX).
    """

    def __init__(self, grid_h: int, grid_w: int, kh: int = 2, kw: int = 2):
        super().__init__()
        assert grid_h % kh == 0 and grid_w % kw == 0, (
            f"grid ({grid_h}, {grid_w}) must be divisible by merge_kernel ({kh}, {kw})"
        )
        self.nh = grid_h // kh
        self.nw = grid_w // kw
        self.kh = kh
        self.kw = kw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (L_pre, d) -> (nh, kh, nw, kw, d) -> (nh, nw, kh, kw, d) -> (L_post, kh*kw*d)
        d = x.shape[-1]
        return (
            x.view(self.nh, self.kh, self.nw, self.kw, d)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(self.nh * self.nw, self.kh * self.kw * d)
        )


class VisionProjForExport(torch.nn.Module):
    """vision_model + static patch_merger + mlp1 as one nn.Module.

    The parent caller must cast model.vision_model and model.mlp1 to the
    export dtype (canonical: fp16) BEFORE constructing this wrapper —
    same cast-scope contract as the R&D exporters. VisionForExport's
    baked pos_emb/grid_hws buffers inherit that cast automatically.
    """

    def __init__(self, model: torch.nn.Module, grid_h: int, grid_w: int):
        super().__init__()
        self.vision = VisionForExport(model.vision_model, grid_h, grid_w)
        kh, kw = self.vision.merge_kh, self.vision.merge_kw
        self.merger = StaticPatchMerger(grid_h, grid_w, kh=kh, kw=kw)
        self.projector = ProjectorForExport(model.mlp1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # fp16 (L_pre, 3, 14, 14) -> fp16 (L_post, text_hidden)
        vit_feats = self.vision(pixel_values)
        merged = self.merger(vit_feats)
        return self.projector(merged)


def export_vision_proj(
    model: torch.nn.Module,
    grid_h: int,
    grid_w: int,
    onnx_path: Path,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    opset: int = 17,
) -> Path:
    """Export VisionProjForExport(model, grid_h, grid_w) to ONNX at onnx_path.

    Idempotent: returns the cached file if present. Uses external-data
    side-cars so the vision-weight tensors (~700 MB fp16) stay below the
    2 GB protobuf hard limit.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export_prod] cached: {onnx_path}")
        return onnx_path

    wrap = VisionProjForExport(model, grid_h, grid_w).eval().to(device)
    L_pre = grid_h * grid_w
    px = torch.zeros(L_pre, 3, 14, 14, dtype=dtype, device=device)

    # L_pre is locked by the baked pos_emb in VisionForExport — no dynamic axis.
    # Output binding name 'visual_features' matches what prefill consumes.
    export_with_external_data(
        wrap, (px,), onnx_path,
        input_names=["pixel_values"],
        output_names=["visual_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export_prod] saved {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB + .bin)")
    return onnx_path
