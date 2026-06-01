"""MLP projector export: LayerNorm(vit_h*4) -> Linear -> GELU -> Linear."""
from __future__ import annotations
from pathlib import Path

import torch


class ProjectorForExport(torch.nn.Module):
    def __init__(self, mlp1: torch.nn.Module):
        super().__init__()
        self.mlp1 = mlp1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp1(x)


def export_projector(
    mlp1: torch.nn.Module,
    vit_feat_dim: int,
    onnx_path: Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Path:
    """Export the projector to ONNX. vit_feat_dim is the post-merger input dim
    (canonical: vit_hidden_size * 4 = 1152 * 4 = 4608)."""
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export] cached: {onnx_path}")
        return onnx_path

    wrap = ProjectorForExport(mlp1).eval().to(device)
    dummy = torch.zeros(32, vit_feat_dim, dtype=dtype, device=device)

    torch.onnx.export(
        wrap, (dummy,), str(onnx_path),
        input_names=["vit_feats_4x"],
        output_names=["proj_feats"],
        dynamic_axes={"vit_feats_4x": {0: "L_post"}, "proj_feats": {0: "L_post"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] saved {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return onnx_path
