"""Vision encoder export.

VisionForExport bakes pos_emb AND grid_hws as buffers — the engine is fixed-resolution
by design. Re-export with a new (grid_h, grid_w) to support a different resolution.

Why this design:
- The canonical MoonViT uses `Learnable2DInterpPosEmb` which has a Python-level
  `for shape in grid_hws.tolist()` loop — not traceable.
- The canonical `patch_merger` also has a `.tolist()` loop AND returns `List[Tensor]`.
- `Rope2DReal.get_freqs_cis` uses `.item()` which constant-folds at trace time anyway.

So the engine is fixed-resolution; the patch_merger runs as numpy outside the engine.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F


class VisionForExport(torch.nn.Module):
    """Single-input (`pixel_values`) wrapper around MoonViT patch_embed + encoder.

    pos_emb is interpolated once at __init__ time for (grid_h, grid_w) and stored as
    a buffer. grid_hws_baked is also a buffer — without that, the exporter would
    constant-fold grid_hws into a dead input and ORT/TRT would reject the engine.
    """

    def __init__(self, vit: torch.nn.Module, grid_h: int, grid_w: int):
        super().__init__()
        self.patch_proj = vit.patch_embed.proj
        self.encoder = vit.encoder

        # Pre-compute the interpolated pos_emb for (grid_h, grid_w).
        weight = vit.patch_embed.pos_emb.weight  # (H_max, W_max, d_model)
        interp_mode = vit.patch_embed.pos_emb.interpolation_mode
        w4 = weight.permute(2, 0, 1).unsqueeze(0)  # (1, d_model, H_max, W_max)
        ikw = {"mode": interp_mode}
        if interp_mode in ("bilinear", "bicubic"):
            ikw["align_corners"] = False
        with torch.no_grad():
            pos = F.interpolate(w4, size=(int(grid_h), int(grid_w)), **ikw)
            pos = pos.squeeze(0).permute(1, 2, 0).flatten(end_dim=1).contiguous()
        self.register_buffer("pos_emb_baked", pos)

        # Bake grid_hws as a constant buffer (1, 2) int32. Rope2DReal uses .item()
        # on grid_hws and sdpa_packed ignores cu_seqlens, so without this the exporter
        # would constant-fold the grid_hws input out of the graph.
        self.register_buffer(
            "grid_hws_baked",
            torch.tensor([[int(grid_h), int(grid_w)]], dtype=torch.int32),
        )
        self.merge_kh, self.merge_kw = vit.merge_kernel_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(pixel_values).view(pixel_values.size(0), -1)
        x = x + self.pos_emb_baked
        x = self.encoder(x, self.grid_hws_baked)
        return x  # (L, d_model) pre-merger; orchestrator runs python_patch_merger after


def export_vision(
    vit_model: torch.nn.Module,
    grid_h: int,
    grid_w: int,
    onnx_path: Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Path:
    """Export VisionForExport(vit_model, grid_h, grid_w) to ONNX at onnx_path.

    Idempotent: returns the existing file if it's present and non-empty.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export] cached: {onnx_path}")
        return onnx_path

    wrap = VisionForExport(vit_model, grid_h, grid_w).eval().to(device)
    L_pre = grid_h * grid_w
    px = torch.zeros(L_pre, 3, 14, 14, dtype=dtype, device=device)

    torch.onnx.export(
        wrap, (px,), str(onnx_path),
        input_names=["pixel_values"],
        output_names=["vit_feats"],
        dynamic_axes={"pixel_values": {0: "L_pre"}, "vit_feats": {0: "L_pre"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter: handles .item() / view_as_complex paths
    )
    print(f"[export] saved {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return onnx_path
