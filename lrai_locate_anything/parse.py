"""Output post-processing + numerical helpers.

`python_patch_merger` replaces the canonical MoonViT `patch_merger` — a deterministic
reshape+permute we run in numpy outside the engine to sidestep the canonical
`for x_shape in grid_hws.tolist()` (untraceable to ONNX).
"""
from __future__ import annotations
import re
from typing import List, Tuple

import numpy as np

BOX_RE = re.compile(r"<box>(.*?)</box>", re.S)
COORD_RE = re.compile(r"<(\d+)>")


def parse_boxes(text: str, W: float = 1.0, H: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """Extract bounding boxes from the model's coord-token output.

    The model emits `<box><x1><y1><x2><y2></box>` blocks with coordinates in [0, 1000].
    We map back to pixel space using (W, H) — pass the image dimensions the model saw.
    """
    out: List[Tuple[float, float, float, float]] = []
    for blk in BOX_RE.findall(text):
        coords = [int(x) for x in COORD_RE.findall(blk)]
        if len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            out.append((x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H))
    return out


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def python_patch_merger(x_np: np.ndarray, gh_np: np.ndarray, kh: int = 2, kw: int = 2) -> np.ndarray:
    """Numerically identical to canonical patch_merger for a single image.

    Canonical:
        seq.view(nh, kh, nw, kw, d).permute(0, 2, 1, 3, 4).contiguous().view(nh*nw, -1)
    where seq has shape (h*w, d) with row-major (outer_h, outer_w) layout.

    Going directly from (L, d) to (nh, kh, nw, kw, d) preserves that layout. The
    `ascontiguousarray` guard avoids a silent permutation if input is non-contiguous.
    """
    h, w = int(gh_np[0, 0]), int(gh_np[0, 1])
    nh, nw = h // kh, w // kw
    d = x_np.shape[-1]
    x = np.ascontiguousarray(x_np)
    return (
        x.reshape(nh, kh, nw, kw, d)
        .transpose(0, 2, 1, 3, 4)
        .reshape(nh * nw, kh * kw * d)
    )
