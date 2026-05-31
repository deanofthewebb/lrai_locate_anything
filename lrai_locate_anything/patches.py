"""ONNX-hostile op replacements for MoonViT, applied to the LIVE model after load.

Three problems in the canonical MoonViT that prevent ONNX export:

1. `flash_attn_varlen_func` is a custom CUDA op — not in ONNX. We swap to SDPA.
   Critical detail: the canonical caller does `.flatten(start_dim=-2)` AFTER the
   attention call (multihead_attention returns (L, H, D)). Our replacement bakes
   that flatten into its own return so the downstream `wo` Linear sees (L, H*D).

2. `apply_rope` uses `torch.view_as_complex` — ONNX has no complex support. We
   replicate the rotation in real space. Critical detail: the canonical applies
   `freqs_cis.unsqueeze(-2)` before the multiply so (L, dim/2) broadcasts against
   (L, num_heads, dim/2). Without that unsqueeze you get `(16) vs (1656)` shape errors.

3. `Rope2DPosEmb` stores `freqs_cis` as a complex tensor. We rebuild it with real
   cos/sin buffers so neither the buffer nor any op in the graph is complex.
"""
from __future__ import annotations
import sys

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) Vision attention replacement
# ---------------------------------------------------------------------------
def sdpa_packed(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    """Replacement for flash_attn_varlen_func / sdpa_attention.

    Single-image path: no per-sample masking required (the orchestrator feeds one
    image per call; multi-image production would use the FlashAttention plugin from
    the original notebook's §10).

    Returns (L, H*D), matching the canonical multihead_attention's post-flatten
    contract. Without the `.flatten(start_dim=-2)` the downstream wo Linear sees the
    wrong inner dim and fails with `mat1 ... cannot be multiplied`.
    """
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0)
    v_ = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=False)
    return out.squeeze(0).transpose(0, 1).flatten(start_dim=-2).contiguous()


# ---------------------------------------------------------------------------
# 2) Real-valued 2-D RoPE
# ---------------------------------------------------------------------------
def apply_rope_real(xq, xk, freqs):
    """Real-valued 2-D RoPE. Accepts complex (canonical) or packed-real freqs.

    Mirrors the canonical apply_rope's `freqs_cis.unsqueeze(-2)` so (L, dim/2)
    broadcasts against xq/xk of shape (L, num_heads, dim/2).
    """
    if torch.is_complex(freqs):
        freqs = torch.stack([freqs.real, freqs.imag], dim=-1)
    cos = freqs[..., 0].unsqueeze(-2)  # (..., L, 1, dim/2)
    sin = freqs[..., 1].unsqueeze(-2)

    def rot(x):
        xp = x.float().reshape(*x.shape[:-1], -1, 2)
        xr, xi = xp.unbind(-1)
        out_r = xr * cos - xi * sin
        out_i = xr * sin + xi * cos
        return torch.stack([out_r, out_i], dim=-1).flatten(-2).to(x.dtype)

    return rot(xq), rot(xk)


class Rope2DReal(torch.nn.Module):
    """Drop-in replacement for the canonical Rope2DPosEmb.

    Stores precomputed cos/sin buffers in real space; the ONNX graph never sees
    `view_as_complex`. Lazy-init guard: if the original Rope2DPosEmb hasn't been
    forwarded yet (its `freqs_cis` is None), force the precompute now.
    """

    def __init__(self, orig: torch.nn.Module):
        super().__init__()
        self.dim = orig.dim
        f = orig.freqs_cis
        if f is None:
            _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            f = orig._precompute_freqs_cis(_dev)
            orig.freqs_cis = f
        self.register_buffer("freqs_cos", f.real.contiguous())
        self.register_buffer("freqs_sin", f.imag.contiguous())

    def get_freqs_cis(self, grid_hws):
        # Convert grid_hws scalars to Python ints so ONNX trace sees a shape-only
        # slice (not a Range/Cast/Gather chain whose shape depends on grid_hws values).
        h_i = int(grid_hws[0, 0].item()) if torch.is_tensor(grid_hws[0, 0]) else int(grid_hws[0, 0])
        w_i = int(grid_hws[0, 1].item()) if torch.is_tensor(grid_hws[0, 1]) else int(grid_hws[0, 1])
        H_max, W_max = self.freqs_cos.shape[0], self.freqs_cos.shape[1]
        assert 1 <= h_i <= H_max and 1 <= w_i <= W_max, (h_i, w_i, H_max, W_max)
        cos = self.freqs_cos[:h_i, :w_i].reshape(-1, self.dim // 2)
        sin = self.freqs_sin[:h_i, :w_i].reshape(-1, self.dim // 2)
        return torch.stack([cos, sin], dim=-1)


# ---------------------------------------------------------------------------
# 3) Apply all three to the live model
# ---------------------------------------------------------------------------
def apply_vision_patches(model, verbose: bool = True) -> None:
    """Patch the loaded model in-place.

    Called AFTER `AutoModel.from_pretrained` because transformers' dynamic-module
    loader registers modules under `transformers_modules.<hash>.modeling_*`. We
    locate those via `sys.modules[type(model).__module__]` and rebind there — that's
    the module the running model's methods actually look up names against.
    """
    mvit_mod = sys.modules[type(model.vision_model).__module__]

    # (a) Vision attention: rebind every variant the model might pick from.
    for nm in ("multihead_attention", "sdpa_attention", "eager_attention", "flash_attn_varlen_func"):
        if hasattr(mvit_mod, nm):
            setattr(mvit_mod, nm, sdpa_packed)
    if hasattr(mvit_mod, "VL_VISION_ATTENTION_FUNCTIONS"):
        for k in list(mvit_mod.VL_VISION_ATTENTION_FUNCTIONS.keys()):
            mvit_mod.VL_VISION_ATTENTION_FUNCTIONS[k] = sdpa_packed

    # (b) apply_rope rebind. Some classes capture the function into their __dict__
    # at import time; rebind those instance attributes too.
    mvit_mod.apply_rope = apply_rope_real
    for _name in dir(mvit_mod):
        obj = getattr(mvit_mod, _name)
        if isinstance(obj, type) and "apply_rope" in getattr(obj, "__dict__", {}):
            setattr(obj, "apply_rope", apply_rope_real)

    # (c) Rope2DPosEmb instance swap. Walk to find it; common names vary.
    vm = model.vision_model
    target = None
    for nm in ("rope_2d", "rope2d", "pos_emb_2d", "rope_emb"):
        if hasattr(vm, nm):
            target = (vm, nm, getattr(vm, nm))
            break
    if target is None:
        for sub_name, sub in vm.named_modules():
            if type(sub).__name__.startswith("Rope2D") and hasattr(sub, "freqs_cis"):
                parts = sub_name.split(".")
                parent = vm
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                target = (parent, parts[-1], sub)
                break
    if target is not None:
        parent, leaf, old = target
        new = Rope2DReal(old).to(device=old.freqs_cis.device)
        setattr(parent, leaf, new)
        if verbose:
            print(f"[patches] swapped {type(old).__name__} -> Rope2DReal at .{leaf}")
    elif verbose:
        print("[patches] WARN: no Rope2DPosEmb instance found; relying on apply_rope_real's complex-input fallback")

    # (d) Belt-and-braces: neutralise 'magi' attention if it's still set.
    cfg = model.config
    if getattr(cfg, "_attn_implementation", None) == "magi":
        cfg._attn_implementation = "sdpa"
    if hasattr(cfg, "text_config") and getattr(cfg.text_config, "_attn_implementation", None) == "magi":
        cfg.text_config._attn_implementation = "sdpa"

    if verbose:
        print("[patches] vision patches applied to loaded model")
