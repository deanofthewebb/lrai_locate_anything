"""Adapter from our MoonViT/projector output to TRT-LLM's prompt-embedding input.

Two modes:
    vision_mode='pt'  -- use the Phase D PT MoonViTVisionModel (verified parity
                         within 1.5e-3 of HF). This is the runtime default for
                         now because vision_proj.engine has not been built yet.
    vision_mode='trt' -- use the fused vision_proj.engine (export_prod path).
                         Preserved for when the engine is built.

The vision_proj engine (built by export_prod) emits visual features in
LocateAnything's native layout: (L_post, hidden), where L_post is the
post-projector token count for a single image and hidden matches the
Qwen2 text-embedding dim.

TRT-LLM's ModelRunner.generate accepts a `prompt_table` argument of
shape (batch=1, L_post, hidden) plus a `prompt_tasks` index and a
`prompt_vocab_size` configured at build time. This adapter reshapes
and positions the features so they land in the virtual-token slots
reserved by build.build_llm_engine.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from lrai_locate_anything.trt.engine import TRTEngine

if TYPE_CHECKING:
    pass


class MoonViTAdapter:
    """Vision encoder wrapper feeding (1, L_post, hidden) bf16 prompt_table to LLM.

    Two modes:
      vision_mode='pt'  -- use Phase D PT MoonViTVisionModel (verified parity
                            within 1.5e-3 of HF).
      vision_mode='trt' -- use fused vision_proj.engine (preserved path).

    PT mode is the runtime default because vision_proj.engine doesn't exist yet.
    Once it's built, callers can switch to 'trt' explicitly.

    TRT-mode format conversion:
        ours:     vision_proj.engine -> Tensor[L_post, hidden]   (bf16, CUDA)
        theirs:   prompt_table       -> Tensor[1, L_post, hidden] with the
                  same dtype as the LLM engine and an additive positional
                  offset baked in (TRT-LLM expects raw token embeddings,
                  not embeddings + RoPE -- RoPE is applied inside the engine).
    """

    # Engine input/output binding names emitted by export_prod.vision_proj.export_vision_proj.
    # See lrai_locate_anything/export_prod/vision_proj.py:101-105.
    _INPUT_NAME = "pixel_values"
    _OUTPUT_NAME = "visual_features"

    # Patch geometry baked into the export_prod graph: (L_pre, 3, 14, 14)
    # where L_pre = grid_h * grid_w and patch_size = 14.
    _PATCH_SIZE = 14

    # Imagenet-style normalization used by the MoonViT processor.
    # OpenCLIP / SigLIP convention used by MoonViT (lrai_locate_anything
    # vendor: processing_locateanything.py uses 0.5 mean/std).
    _MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    _STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(
        self,
        vision_mode: str,
        *,
        # PT-mode args
        weights_dir: Optional[str] = None,
        grid_h: int = 36,
        grid_w: int = 46,
        # TRT-mode args
        vision_proj_engine_path: Optional[str] = None,
        # Common
        llm_dtype: str = "bfloat16",
    ):
        """Initialize the adapter in either PT or TRT mode.

        Args:
            vision_mode: 'pt' or 'trt' -- REQUIRED, no implicit default.
            weights_dir: required if vision_mode='pt'. Directory holding the
                MoonViT/LocateAnything safetensors shards.
            grid_h: patch-grid height (PT mode init default).
            grid_w: patch-grid width (PT mode init default).
            vision_proj_engine_path: required if vision_mode='trt'. Path to the
                export_prod-built TRT engine that fuses MoonViT + projector.
            llm_dtype: name of the downstream LLM engine prompt-embedding dtype
                ('bfloat16', 'float16', or 'float32'). Default 'bfloat16'.
        """
        if vision_mode not in ("pt", "trt"):
            raise RuntimeError(
                f"vision_mode must be 'pt' or 'trt', got {vision_mode!r}"
            )
        self.vision_mode = vision_mode
        self.llm_dtype = llm_dtype
        self.grid_h = grid_h
        self.grid_w = grid_w

        if vision_mode == "pt":
            if weights_dir is None:
                raise RuntimeError("vision_mode='pt' requires weights_dir")
            self.weights_dir = weights_dir
            self._init_pt(weights_dir, grid_h, grid_w)
        else:  # 'trt'
            if vision_proj_engine_path is None:
                raise RuntimeError(
                    "vision_mode='trt' requires vision_proj_engine_path"
                )
            self._init_trt(vision_proj_engine_path)

    # ------------------------------------------------------------------ PT mode

    def _init_pt(self, weights_dir: str, grid_h: int, grid_w: int) -> None:
        """Load the Phase D PT MoonViTVisionModel and install rope freqs."""
        import os
        from safetensors.torch import load_file
        from transformers import AutoModel
        from lrai_locate_anything.trtllm_prod.modeling_moonvit import (
            MoonViTVisionModel,
        )

        sd: dict = {}
        for fn in sorted(os.listdir(weights_dir)):
            if fn.endswith(".safetensors"):
                sd.update(load_file(os.path.join(weights_dir, fn)))
        vision_sd = {
            k: v
            for k, v in sd.items()
            if k.startswith("vision_model.") or k.startswith("mlp1.")
        }

        self.vision_model = MoonViTVisionModel.from_moonvit_state_dict(
            vision_sd, grid_h=grid_h, grid_w=grid_w, use_bf16=False
        )

        # Harvest freqs_source from HF Rope2DPosEmb (one-time)
        hf_full = AutoModel.from_pretrained(
            weights_dir, trust_remote_code=True, torch_dtype=torch.float32
        )
        hf_vision = hf_full.vision_model
        rope = hf_vision.encoder.rope_2d
        rope.get_freqs_cis(torch.tensor([[64, 64]]))  # populate to max
        freqs_cis = rope.freqs_cis
        freqs_source = torch.stack(
            [freqs_cis.real, freqs_cis.imag], dim=-1
        ).to(torch.float32)
        init_packed = (
            freqs_source[:grid_h, :grid_w]
            .reshape(grid_h * grid_w, freqs_source.shape[2], 2)
            .contiguous()
        )
        self.vision_model.install_pt_attention_swap(
            freqs_packed=init_packed, freqs_source=freqs_source
        )
        del hf_full, hf_vision  # free RAM

        self.vision_model = self.vision_model.cuda().eval()
        # PT mode does no letterboxing inside the adapter; the processor does
        # it upstream. Adopt the same image-prep as orchestrator's
        # _processor_call for parity.

    # ----------------------------------------------------------------- TRT mode

    def _init_trt(self, vision_proj_engine_path: str) -> None:
        """Load the prebuilt vision_proj.engine (preserved path)."""
        vision_proj_engine = Path(vision_proj_engine_path)
        if not vision_proj_engine.exists():
            raise FileNotFoundError(
                f"vision_proj engine not found at {vision_proj_engine}. "
                f"Build it first via lrai_locate_anything.export_prod.vision_proj.export_vision_proj "
                f"+ trtexec."
            )
        self.engine = TRTEngine(vision_proj_engine)
        # The engine has fixed input shape (L_pre, 3, P, P) -- bound at export time.
        # Recover L_pre + (grid_h, grid_w) from the engine's profile so the
        # adapter knows what resolution to letterbox the PIL image to.
        in_shape = self.engine.engine.get_tensor_shape(self._INPUT_NAME)
        if tuple(in_shape[1:]) != (3, self._PATCH_SIZE, self._PATCH_SIZE):
            raise RuntimeError(
                f"vision_proj engine input shape {tuple(in_shape)} does not match "
                f"expected (L_pre, 3, {self._PATCH_SIZE}, {self._PATCH_SIZE})."
            )
        self.L_pre = int(in_shape[0])
        # The export bakes a square-ish grid; we don't assume aspect ratio, but
        # the export contract is grid_h * grid_w == L_pre. Default to the
        # canonical 36x46 (LocateAnything thumbnail target) when both factor;
        # otherwise force a near-square that divides cleanly.
        self.grid_h, self.grid_w = self._derive_grid(self.L_pre)
        self.img_h = self.grid_h * self._PATCH_SIZE
        self.img_w = self.grid_w * self._PATCH_SIZE

    @staticmethod
    def _derive_grid(L_pre: int) -> Tuple[int, int]:
        """Pick (grid_h, grid_w) such that grid_h*grid_w == L_pre. Prefers the
        canonical LocateAnything thumbnail grid (36, 46) when L_pre matches."""
        if L_pre == 36 * 46:
            return 36, 46
        # Find the factor pair closest to square.
        best = (1, L_pre)
        for h in range(1, int(L_pre ** 0.5) + 1):
            if L_pre % h == 0:
                w = L_pre // h
                if abs(w - h) < abs(best[1] - best[0]):
                    best = (h, w)
        return best

    def _letterbox(self, image: "Image.Image") -> Tuple[np.ndarray, float, int, int]:
        """Aspect-preserving resize + center-pad to (img_w, img_h).

        Returns (chw_float_normalized, scale, pad_x, pad_y) where the array is
        (3, img_h, img_w) float32 in [-1, 1] after Imagenet-style normalization.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.img_w / orig_w, self.img_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.img_w, self.img_h), (128, 128, 128))
        pad_x = (self.img_w - new_w) // 2
        pad_y = (self.img_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        arr = (arr - self._MEAN) / self._STD
        return arr.astype(np.float32), scale, pad_x, pad_y

    def _patchify(self, chw: np.ndarray) -> np.ndarray:
        """Cut (1, 3, img_h, img_w) -> (L_pre, 3, 14, 14) flat patch sequence
        in row-major (grid_h, grid_w) layout. Matches the layout the canonical
        MoonViT processor emits (vendor processing_locateanything.py).
        """
        _, c, H, W = chw.shape
        P = self._PATCH_SIZE
        gh, gw = self.grid_h, self.grid_w
        if H != gh * P or W != gw * P:
            raise RuntimeError(
                f"patchify: input shape ({H},{W}) does not match grid "
                f"({gh}x{gw}) * patch ({P})"
            )
        # (1, 3, gh, P, gw, P) -> (gh, gw, 3, P, P) -> (gh*gw, 3, P, P)
        x = chw.reshape(1, c, gh, P, gw, P)
        x = x.transpose(0, 2, 4, 1, 3, 5)  # (1, gh, gw, 3, P, P)
        x = x.reshape(gh * gw, c, P, P)
        return np.ascontiguousarray(x)

    # ------------------------------------------------------------------ forward

    @torch.no_grad()
    def forward(self, image, processor=None):
        """Run vision encoding for a single PIL image.

        PT mode returns a (1, L_post, hidden) prompt_table tensor on CUDA.
        TRT mode returns (prompt_table, scale, pad_x, pad_y) -- preserved
        contract for callers that need letterbox params to un-warp box coords.
        """
        if self.vision_mode == "pt":
            return self._forward_pt(image, processor)
        return self._forward_trt(image)

    def _forward_pt(self, image_pil, processor=None):
        """PT-mode forward: HF AutoProcessor -> Phase D MoonViTVisionModel."""
        if processor is None:
            if not hasattr(self, "_processor"):
                raise RuntimeError(
                    "_forward_pt requires a processor; pass via processor="
                )
            processor = self._processor

        # The processor __call__ requires a non-None text argument; line 480 of
        # processing_locateanything.py does `text[0]` unconditionally when text
        # is not a str, which crashes when text=None.  Pass the standard
        # single-image placeholder so the processor substitutes it with the
        # visual token sequence and also returns image_grid_hws.
        image_placeholder_text = getattr(processor, "image_placeholder", "image")
        inputs = processor(images=image_pil, text=f"<{image_placeholder_text}-1>", return_tensors="pt")
        pixel_values = inputs["pixel_values"].cuda()
        grid_hws = inputs.get("image_grid_hws", None)
        if grid_hws is None:
            raise RuntimeError(
                "processor did not return image_grid_hws; cannot drive set_grid"
            )

        # The processor typically returns (L_pre, 3, 14, 14); if (1, L_pre, 3, 14, 14)
        # appears with a batch dim, squeeze it.
        if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)

        if grid_hws.ndim == 2 and grid_hws.shape[0] == 1:
            h, w = int(grid_hws[0, 0]), int(grid_hws[0, 1])
        else:
            raise RuntimeError(
                f"unexpected grid_hws shape {tuple(grid_hws.shape)}; expected (1, 2)"
            )

        # Drive set_grid if the processor handed us a different layout.
        if (h, w) != (
            self.vision_model.embeddings.grid_h,
            self.vision_model.embeddings.grid_w,
        ):
            self.vision_model.set_grid(h, w)

        tokens = self.vision_model(pixel_values)  # (L_post, 2048)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(self.llm_dtype, torch.bfloat16)
        return tokens.unsqueeze(0).to(target_dtype).contiguous()

    def _forward_trt(self, image: "Image.Image") -> Tuple["torch.Tensor", float, int, int]:
        """TRT-mode forward (preserved): vision_proj.engine path.

        Pipeline:
            PIL.Image
              -> letterbox + normalize (matches export_prod's input contract)
              -> patchify into (L_pre, 3, 14, 14)
              -> vision_proj.engine.execute() -> (L_post, hidden)
              -> unsqueeze(0)                 -> (1, L_post, hidden)
              -> contiguous, dtype-aligned to the LLM engine

        Returns:
            (prompt_table, scale, pad_x, pad_y) where prompt_table is a
            torch.Tensor of shape (1, L_post, hidden) on CUDA in llm_dtype,
            and (scale, pad_x, pad_y) are the letterbox params needed by the
            caller to un-letterbox box coordinates back to the original image.
        """
        chw, scale, pad_x, pad_y = self._letterbox(image)
        patches = self._patchify(chw)  # (L_pre, 3, P, P) float32
        # The engine input is fp16 (export_prod export_vision_proj default dtype).
        # See lrai_locate_anything/export_prod/vision_proj.py:96.
        patches_in = patches.astype(np.float16)
        out = self.engine({self._INPUT_NAME: patches_in})
        vis = out[self._OUTPUT_NAME]  # (L_post, hidden) -- numpy, fp16 OR uint16(bf16)
        # If the binding is BF16, vis comes back as uint16 (byte storage). View
        # as bf16, then upcast to llm_dtype. Otherwise fp16 -> llm_dtype via
        # a real precision cast through fp32.
        if self.engine.is_bf16.get(self._OUTPUT_NAME, False):
            t = torch.from_numpy(vis).view(torch.bfloat16)
        else:
            t = torch.from_numpy(vis.astype(np.float32))
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = (
            dtype_map.get(self.llm_dtype, torch.bfloat16)
            if isinstance(self.llm_dtype, str)
            else self.llm_dtype
        )
        t = t.to(device="cuda", dtype=target_dtype).contiguous()
        if t.ndim != 2:
            raise RuntimeError(
                f"vision_proj output expected 2D (L_post, hidden); got shape={tuple(t.shape)}"
            )
        prompt_table = t.unsqueeze(0).contiguous()  # (1, L_post, hidden)
        return prompt_table, scale, pad_x, pad_y
