"""Download nvidia/LocateAnything-3B, install shims, load with rehydrated config, patch.

Single entry point: `load_locateanything_3b()`. Returns a fully-prepared (model, tokenizer,
processor, config) tuple ready for export or PyTorch inference.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Tuple

import torch
from huggingface_hub import snapshot_download

from .config import MODEL_ID, REF_DTYPE, WEIGHTS_DIR
from .shims import install_transformers_shims, rehydrate_config
from .patches import apply_vision_patches


def download_weights(model_id: str = MODEL_ID, local_dir: Path | None = None) -> Path:
    local_dir = local_dir or WEIGHTS_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        model_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "*.json", "*.py", "*.txt",
            "*.safetensors", "*.safetensors.index.json",
            "tokenizer*", "chat_template*", "generation_config*",
        ],
    )
    return Path(path)


def load_locateanything_3b(
    local_dir: Path | None = None,
    model_id: str = MODEL_ID,
    dtype: torch.dtype = REF_DTYPE,
    install_shims: bool = True,
    apply_patches: bool = True,
    verbose: bool = True,
) -> Tuple[object, object, object, object, Path]:
    """Load LocateAnything-3B end-to-end with all shims and patches applied.

    Returns
    -------
    model, tokenizer, processor, config, local_dir
    """
    if install_shims:
        install_transformers_shims(verbose=verbose)

    local = Path(local_dir) if local_dir else download_weights(model_id)
    if verbose:
        print(f"[loader] weights at {local}")

    # The vendored modeling files import each other relatively; add the parent to
    # sys.path so `transformers_modules.<dirname>.<module>` resolves.
    import sys
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))

    # transformers' dynamic loader will (re)load modeling files via trust_remote_code.
    from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    # use_fast=False keeps the slow image processor that ships with the model (no fast variant).
    processor = AutoProcessor.from_pretrained(str(local), trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(str(local), trust_remote_code=True)

    # Rehydrate every text/vision_config field from config.json — newer transformers move
    # RoPE params into rope_parameters but the vendored modeling_qwen2 reads them directly.
    with open(local / "config.json") as f:
        raw = json.load(f)
    rehydrate_config(config, raw)
    if verbose:
        print(f"[loader] text_config rope_theta={config.text_config.rope_theta}, "
              f"sliding={config.text_config.use_sliding_window}, block_size={config.text_config.block_size}")

    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.text_config.use_cache = True

    # The torch_dtype kwarg is deprecated in transformers >=4.55 (use `dtype=`) but
    # still accepted; we pass torch_dtype for broad compatibility.
    model = AutoModel.from_pretrained(
        str(local), trust_remote_code=True, torch_dtype=dtype, config=config,
    ).eval().to("cuda" if torch.cuda.is_available() else "cpu")

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"[loader] loaded {n_params/1e9:.2f} B params  "
            f"(vision={sum(p.numel() for p in model.vision_model.parameters())/1e6:.0f} M, "
            f"projector={sum(p.numel() for p in model.mlp1.parameters())/1e6:.1f} M, "
            f"lm={sum(p.numel() for p in model.language_model.parameters())/1e9:.2f} B)"
        )

    if apply_patches:
        apply_vision_patches(model, verbose=verbose)

    return model, tokenizer, processor, config, local


def normalize_image_grid_hws(inputs: dict, pixel_values_key: str = "pixel_values"):
    """Ensure inputs['image_grid_hws'] is a torch int32 tensor.

    The custom processor sometimes returns image_grid_hws as a numpy array whose dtype
    is `numpy.dtypes.Int64DType`, which `torch.zeros(..., dtype=...)` inside the
    vendored modeling_vit.forward rejects.
    """
    gh = inputs.get("image_grid_hws")
    if gh is None:
        return inputs
    if not torch.is_tensor(gh):
        inputs["image_grid_hws"] = torch.as_tensor(
            gh, dtype=torch.int32, device=inputs[pixel_values_key].device
        )
    else:
        inputs["image_grid_hws"] = gh.to(dtype=torch.int32)
    return inputs


def lock_processor_resolution(processor, eng_img_w: int, eng_img_h: int, verbose: bool = True) -> None:
    """Lock the LocateAnything image processor's rescale() to be a no-op for input
    already sized to (eng_img_w, eng_img_h).

    The vendored LocateAnythingImageProcessor is a Kimi-VL-style sizer, not Qwen2-VL.
    Its rescale() does TWO transforms:
      1) token-budget downscale (aspect-preserving) if (w/P)*(h/P) > in_token_limit
      2) snap-resize (NOT pad) to the next multiple of merge_kernel_size * patch_size
    To make rescale() a true no-op on a correctly pre-sized frame, we need:
      - (eng_img_w/P) * (eng_img_h/P) <= in_token_limit         [avoids step 1]
      - eng_img_w and eng_img_h already multiples of mk*P       [avoids step 2]
      - eng_img_w/P < 512 and eng_img_h/P < 512                  [avoids "Exceed pos emb"]
    There is no min_pixels/max_pixels on this processor; do not look for them.
    """
    ip = processor.image_processor
    P = getattr(ip, "patch_size", None)
    mk = getattr(ip, "merge_kernel_size", None)
    tl = getattr(ip, "in_token_limit", None)
    if P is None or mk is None or tl is None:
        if verbose:
            print(f"[loader] WARN: image_processor lacks patch_size/merge_kernel_size/in_token_limit (type={type(ip).__name__}); cannot lock")
        return
    mk_h, mk_w = int(mk[0]), int(mk[1])
    grid_w = eng_img_w // P
    grid_h = eng_img_h // P
    if eng_img_w % (mk_w * P) != 0 or eng_img_h % (mk_h * P) != 0:
        raise RuntimeError(
            f"engine resolution ({eng_img_w}x{eng_img_h}) is not a multiple of "
            f"merge_kernel_size*patch_size ({mk_w*P}x{mk_h*P}); the processor will "
            f"snap-resize and break the baked grid_hws"
        )
    if grid_w >= 512 or grid_h >= 512:
        raise RuntimeError(
            f"engine grid ({grid_h}x{grid_w}) exceeds processor pos-emb ceiling 512"
        )
    needed_tl = grid_w * grid_h
    changed = []
    if tl < needed_tl:
        ip.in_token_limit = needed_tl
        changed.append(f"in_token_limit: {tl} -> {needed_tl}")
    if verbose:
        msg = (f"[loader] processor resolution lock OK "
               f"(P={P}, mk={mk_h}x{mk_w}, grid={grid_h}x{grid_w}, in_token_limit={ip.in_token_limit})")
        if changed:
            msg += f"  raised: {'; '.join(changed)}"
        print(msg)
