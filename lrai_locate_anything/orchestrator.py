"""LocateAnythingRunner — the public API.

Loads the model, optionally exports + builds engines, exposes `.detect()` for
single-image inference and provides the TRT-orchestrator generate-loop that
mirrors the canonical generate() with MTP↔AR hybrid decoding.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

import re
from .config import MODEL_ID, REF_DTYPE, WORK, ONNX_DIR, TRT_DIR, enable_llm_trt, ensure_nvidia_stack
from .model_loader import load_locateanything_3b, normalize_image_grid_hws, lock_processor_resolution
from .parse import parse_boxes, python_patch_merger


# ---------------------------------------------------------------------------
# Letterbox (aspect-preserving resize + pad) for the fixed-shape TRT engine.
# NOTE: a naive BICUBIC-stretch to (eng_w, eng_h) distorts aspect ratio and
# produces visual features MoonViT cannot interpret — the LM degenerates into
# the <ref>$$$$$ / <ref>""""" mode-collapse loop on ALL generation paths.
# Letterbox preserves aspect ratio at the cost of gray pad bars (which the
# vision tower handles fine — they are a small constant region).
# ---------------------------------------------------------------------------
def _letterbox(img: Image.Image, target_w: int, target_h: int,
               color: Tuple[int, int, int] = (128, 128, 128)
               ) -> Tuple[Image.Image, float, int, int]:
    """Aspect-preserving resize + center-pad to exactly (target_w, target_h).

    Returns (letterboxed_image, scale, pad_x, pad_y) where
        original_x = (letterbox_x - pad_x) / scale
        original_y = (letterbox_y - pad_y) / scale
    """
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


# ---------------------------------------------------------------------------
# Prompt auto-canonicalization
# ---------------------------------------------------------------------------
_CANONICAL_PREFIX = "Locate all the instances that matches the following description: "

def canonicalize_prompt(prompt: str) -> Tuple[str, bool]:
    """Rewrite natural-language prompts to the model's canonical training phrasing.

    LocateAnything-3B was trained exclusively on prompts of the form
        "Locate all the instances that matches the following description: X."
    where X is a single category or "<cat1></c><cat2>" for multi-class. OOD
    phrasings like "Detect all cats. Return bounding boxes." cause the MTP head
    to mode-collapse on the mask token. This helper extracts the object
    description from common natural-language phrasings and reformats to canonical.

    Returns (rewritten_prompt, was_rewritten).
    """
    p = prompt.strip()
    if "matches the following description:" in p.lower():
        return p, False
    # Strip boilerplate suffix ("Return bounding boxes", "Please provide", etc.)
    p = re.sub(r"[\.,]?\s*(return|provide|give|find|with|please)\s+(bounding\s+box(es)?|boxes|coordinates)\s*\.?\s*$",
               "", p, flags=re.I)
    p = re.sub(r"\s*\.\s*$", "", p)
    # Extract the object after a detection verb
    m = re.match(
        r"^(?:please\s+)?"
        r"(?:detect|find|locate|bound|identify|ground|point\s+(?:to|at)|where\s+(?:are|is)|show\s+me|spot)"
        r"\s+(?:all\s+|every\s+|the\s+|any\s+|each\s+|me\s+the\s+)?(.+)$",
        p, re.I,
    )
    if not m:
        return prompt, False
    target = m.group(1).strip()
    # Multi-class join: "cats and dogs", "people, cars and bikes" -> "cat</c>dog", etc.
    target = re.sub(r",?\s+and\s+|,\s*", "</c>", target)
    # Trim trailing punctuation
    target = re.sub(r"[\.\?!]+\s*$", "", target).strip()
    if not target:
        return prompt, False
    return _CANONICAL_PREFIX + target + ".", True


# ---------------------------------------------------------------------------
# TRT-engine-based generate-loop
# ---------------------------------------------------------------------------
class _TRTGenerator:
    """Mirrors the canonical PBD generate() over TRT engines. AR-only when only
    vision/projector engines are available; full MTP+AR when LLM engines exist.
    """

    def __init__(self, runner: "LocateAnythingRunner"):
        self.r = runner

    # ---- atomic ops over engines ----
    def _vision(self, px: np.ndarray) -> np.ndarray:
        # The vision engine is fixed-resolution; px MUST be (L_pre_fixed, 3, 14, 14).
        pre = self.r.vit_engine({"pixel_values": px})["vit_feats"]  # (L_pre_fixed, 1152)
        gh = np.array([[self.r.grid_h, self.r.grid_w]], dtype=np.int32)
        return python_patch_merger(pre, gh, kh=2, kw=2).astype(np.float16)

    def _project(self, x: np.ndarray) -> np.ndarray:
        return self.r.proj_engine({"vit_feats_4x": x})["proj_feats"]

    def _prefill_trt(self, ids, vf, pos, att):
        o = self.r.prefill_engine({
            "input_ids":       ids.astype(np.int64),
            "visual_features": vf.astype(np.float16),
            "position_ids":    pos.astype(np.int64),
            "attention_mask":  att.astype(np.int64),
        })
        past = []
        for i in range(self.r.n_layers):
            past.append(o[f"present_k_{i}"])
            past.append(o[f"present_v_{i}"])
        return o["logits"], past

    def _decode_trt(self, ids, pos, att, past):
        feed = {
            "input_ids":      ids.astype(np.int64),
            "position_ids":   pos.astype(np.int64),
            "attention_mask": att.astype(np.int64),
        }
        for i in range(self.r.n_layers):
            feed[f"past_k_{i}"] = past[2 * i]
            feed[f"past_v_{i}"] = past[2 * i + 1]
        o = self.r.decode_engine(feed)
        nxt = []
        for i in range(self.r.n_layers):
            nxt.append(o[f"present_k_{i}"])
            nxt.append(o[f"present_v_{i}"])
        return o["logits"], nxt

    def _prefill_pt(self, ids, vf, pos, att):
        with torch.inference_mode():
            i = torch.from_numpy(ids).long().cuda()
            v = torch.from_numpy(vf).cuda().to(REF_DTYPE)
            p = torch.from_numpy(pos).long().cuda()
            m = torch.from_numpy(att).long().cuda()
            o = self.r.model.language_model.model(
                input_ids=i, visual_features=v,
                image_token_index=int(self.r.config.image_token_index),
                position_ids=p, attention_mask=m,
                use_cache=True, return_dict=True,
            )
            logits = self.r.model.language_model.lm_head(o.last_hidden_state).float().cpu().numpy().astype(np.float16)
            past = []
            for (k, vv) in o.past_key_values:
                past.append(k.cpu().numpy().astype(np.float16))
                past.append(vv.cpu().numpy().astype(np.float16))
        return logits, past

    def _decode_pt(self, ids, pos, att, past):
        with torch.inference_mode():
            i = torch.from_numpy(ids).long().cuda()
            p = torch.from_numpy(pos).long().cuda()
            m = torch.from_numpy(att).long().cuda()
            pkv = tuple(
                (torch.from_numpy(past[2 * j]).cuda().to(REF_DTYPE),
                 torch.from_numpy(past[2 * j + 1]).cuda().to(REF_DTYPE))
                for j in range(self.r.n_layers)
            )
            o = self.r.model.language_model.model(
                input_ids=i, position_ids=p, attention_mask=m,
                past_key_values=pkv, use_cache=True, return_dict=True,
            )
            logits = self.r.model.language_model.lm_head(o.last_hidden_state).float().cpu().numpy().astype(np.float16)
            nxt = []
            for (k, vv) in o.past_key_values:
                nxt.append(k.cpu().numpy().astype(np.float16))
                nxt.append(vv.cpu().numpy().astype(np.float16))
        return logits, nxt

    def _prefill(self, *a):
        return (self._prefill_trt if self.r.prefill_engine else self._prefill_pt)(*a)

    def _decode(self, *a):
        return (self._decode_trt if self.r.decode_engine else self._decode_pt)(*a)

    # ---- main loop ----
    def generate(self, pixel_values, input_ids,
                 max_new_tokens=128, generation_mode="hybrid",
                 temperature=0.0, top_p=1.0, repetition_penalty=1.0):
        from generate_utils import sample_tokens, sample_tokens_ar, handle_pattern

        # 1) vision (pre-merger -> Python patch_merger) -> projector
        vit = self._vision(pixel_values.astype(np.float16))
        proj = self._project(vit)
        # 2) prefill via canonical contract: input_ids + visual_features
        ids = input_ids.astype(np.int64)
        S = ids.shape[1]
        pos = np.arange(S, dtype=np.int64)[None, :]
        att = np.ones((1, S), dtype=np.int64)
        n_img_in_ids = int((ids == self.r.TID["image_token_index"]).sum())
        logits, past = self._prefill(ids, proj[:n_img_in_ids], pos, att)
        generated = ids.copy()
        use_mtp = (generation_mode != "slow")
        out_tokens: List[int] = []
        BLOCK = self.r.config.text_config.block_size

        for _ in range(max_new_tokens):
            P = past[0].shape[2]
            if use_mtp and generation_mode != "slow":
                last = generated[:, -1:]
                mask_ids = np.full((1, BLOCK - 1), int(self.r.TID["default_mask_token_id"]), dtype=np.int64)
                mtp_ids = np.concatenate([last, mask_ids], axis=1)
                pos = np.arange(P, P + BLOCK, dtype=np.int64)[None, :]
                pos[:, -BLOCK:] -= 1
                att = np.ones((1, P + BLOCK), dtype=np.int64)
                logits_o, nxt = self._decode(mtp_ids, pos, att, past)
                lt = torch.from_numpy(logits_o.astype(np.float32))
                gt = torch.from_numpy(generated).long()
                probs, conf, x0, box_avg = sample_tokens(
                    lt, gt, self.r.TID,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generation_mode=generation_mode, keep_k=5,
                )
                # MTP failed to decode a valid box/ref pattern: box_avg is None
                # (seq_len==1 short-circuit in sample_tokens, shouldn't happen with
                # BLOCK>1 but defensive) or the b=0 slot is the length-1 fallback_box
                # returned by sample_tokens when neither decode_bbox_avg nor decode_ref
                # succeeded. Canonical recovery: take one AR step, do NOT commit MTP's
                # raw per-position argmax (which would land in handle_pattern's ref
                # branch as garbage).
                if box_avg is None or box_avg[0].numel() < 6 or (box_avg[0] == 0).all().item():
                    if generation_mode == "hybrid":
                        use_mtp = False
                        continue
                    # 'fast' mode has no AR fallback; commit raw argmax as before.
                    new_tokens = x0[0]
                else:
                    new_tokens = box_avg[0]
                pat = handle_pattern(new_tokens, self.r.TID, generation_mode)
                if pat.get("need_switch_to_ar") and generation_mode == "hybrid":
                    use_mtp = False
                    continue
                toks = list(pat["tokens"])
                if not toks:
                    # handle_pattern cannot return empty tokens for any canonical input;
                    # treat as a degenerate state and fall back to AR rather than break,
                    # otherwise we drop the stream silently.
                    if generation_mode == "hybrid":
                        use_mtp = False
                        continue
                    break
                tnp = np.asarray(toks, dtype=np.int64)[None, :]
                generated = np.concatenate([generated, tnp], axis=1)
                out_tokens.extend(toks)
                k = tnp.shape[1]
                past = [nxt[2 * i][:, :, :P + k, :] for i in range(self.r.n_layers)] + \
                       [nxt[2 * i + 1][:, :, :P + k, :] for i in range(self.r.n_layers)]
                past = [t for pair in zip(past[:self.r.n_layers], past[self.r.n_layers:]) for t in pair]
                if pat.get("is_terminal") or int(self.r.TID["im_end_token_id"]) in toks:
                    break
            else:
                last_id = generated[:, -1:]
                pos = np.array([[P]], dtype=np.int64)
                att = np.ones((1, P + 1), dtype=np.int64)
                logits_o, nxt = self._decode(last_id, pos, att, past)
                lt = torch.from_numpy(logits_o.astype(np.float32))
                gt = torch.from_numpy(generated).long()
                probs, conf, x0, *_ = sample_tokens_ar(
                    lt, gt, self.r.TID,
                    temperature=temperature, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                tok = int(x0[0, 0].item())
                generated = np.concatenate([generated, [[tok]]], axis=1)
                out_tokens.append(tok)
                past = nxt
                if generation_mode == "hybrid" and tok == int(self.r.TID["box_end_token_id"]):
                    use_mtp = True
                if tok == int(self.r.TID["im_end_token_id"]):
                    break
        return generated, out_tokens


# ---------------------------------------------------------------------------
# Public Runner
# ---------------------------------------------------------------------------
class LocateAnythingRunner:
    """High-level API: load model, export + build engines on demand, run inference."""

    def __init__(self, model, tokenizer, processor, config, local_dir: Path, patches_snapshot=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.local_dir = local_dir
        # Snapshot returned by apply_vision_patches — enables temporary patch revert
        # for the A/B diagnostic fallback inside detect().
        self.patches_snapshot = patches_snapshot

        # LLM shape constants
        tc = config.text_config
        self.hidden_size = tc.hidden_size
        self.n_layers = tc.num_hidden_layers
        self.n_kv_heads = tc.num_key_value_heads
        self.head_dim = tc.hidden_size // tc.num_attention_heads
        self.vocab_size = tc.vocab_size

        # Token ID dict (both suffixed canonical keys and short aliases the orchestrator uses)
        self.TID = {
            "box_start_token_id":    config.box_start_token_id,
            "box_end_token_id":      config.box_end_token_id,
            "coord_start_token_id":  config.coord_start_token_id,
            "coord_end_token_id":    config.coord_end_token_id,
            "ref_start_token_id":    config.ref_start_token_id,
            "ref_end_token_id":      config.ref_end_token_id,
            "none_token_id":         config.none_token_id,
            "null_token_id":         tc.null_token_id,
            "switch_token_id":       tc.switch_token_id,
            "default_mask_token_id": tc.text_mask_token_id,
            "im_end_token_id":       tc.eos_token_id,
            "image_token_index":     config.image_token_index,
            # short aliases
            "box_start": config.box_start_token_id, "box_end": config.box_end_token_id,
            "coord_start": config.coord_start_token_id, "coord_end": config.coord_end_token_id,
            "ref_start": config.ref_start_token_id,  "ref_end": config.ref_end_token_id,
            "none": config.none_token_id, "null": tc.null_token_id,
            "switch": tc.switch_token_id, "mask": tc.text_mask_token_id,
            "im_end": tc.eos_token_id,    "image": config.image_token_index,
        }

        # Engines (populated by .build_engines())
        self.vit_engine = None
        self.proj_engine = None
        self.prefill_engine = None
        self.decode_engine = None

        # Resolution (populated by .export_engines())
        self.grid_h: Optional[int] = None
        self.grid_w: Optional[int] = None
        self.eng_img_w: Optional[int] = None
        self.eng_img_h: Optional[int] = None

        self._gen = _TRTGenerator(self)

    # ----- factory -----
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        local_dir: Optional[Path] = None,
        auto_export: bool = False,
        sample_image: Optional[Path] = None,
        sample_prompt: str = "Locate all the instances that matches the following description: cat.",
    ) -> "LocateAnythingRunner":
        """Load model + (optionally) export and build engines on first run.

        For auto_export=True, supply a sample_image whose resolution will determine
        the baked engine size. If None, a tiny default image is used (grid 36x46).
        """
        ensure_nvidia_stack(verbose=False)
        model, tokenizer, processor, config, local, snap = load_locateanything_3b(
            local_dir=local_dir, model_id=model_id,
        )
        runner = cls(model, tokenizer, processor, config, local, patches_snapshot=snap)
        if auto_export:
            runner.export_engines(sample_image=sample_image, sample_prompt=sample_prompt)
            runner.build_engines()
            runner.load_engines()
        return runner

    # ----- export + build -----
    def _processor_call(self, image: Image.Image, prompt: str) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        enc = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        normalize_image_grid_hws(enc)
        enc["pixel_values"] = enc["pixel_values"].to(REF_DTYPE)
        return enc

    def export_engines(self, sample_image: Optional[Path] = None, sample_prompt: str = "Locate all the instances that matches the following description: cat."):
        """Run the four exports (vision, projector, llm_prefill, llm_decode).

        The vision engine bakes pos_emb at the resolution implied by `sample_image`.
        Pass a sample image representative of your downstream workload (resolution-wise).
        """
        from .export import (
            export_vision, export_projector,
            export_llm_prefill, export_llm_decode,
        )

        # Use the sample to determine the bake resolution.
        if sample_image is not None:
            img = Image.open(sample_image).convert("RGB")
        else:
            # Default: 672x672 thumbnail → grid (36, 46) for the demo cats image.
            img = Image.new("RGB", (672, 672), color=(128, 128, 128))
        img.thumbnail((672, 672))
        enc = self._processor_call(img, sample_prompt)
        gh_t = enc["image_grid_hws"]
        self.grid_h, self.grid_w = int(gh_t[0, 0]), int(gh_t[0, 1])
        self.eng_img_w = self.grid_w * 14
        self.eng_img_h = self.grid_h * 14
        n_img_tokens = (self.grid_h * self.grid_w) // 4
        print(f"[runner] baking engines for grid_hws=({self.grid_h},{self.grid_w})  "
              f"L_pre={self.grid_h*self.grid_w}  L_post={n_img_tokens}")

        vit_h = self.config.vision_config.hidden_size
        vit_feat_dim = vit_h * 4

        export_vision(self.model.vision_model, self.grid_h, self.grid_w, ONNX_DIR / "vision.onnx")
        export_projector(self.model.mlp1, vit_feat_dim, ONNX_DIR / "projector.onnx")

        lm_main = self.model.language_model.model
        lm_head = self.model.language_model.lm_head
        export_llm_prefill(
            lm_main, lm_head,
            image_token_index=int(self.config.image_token_index),
            n_layers=self.n_layers, hidden_size=self.hidden_size,
            n_img_tokens=n_img_tokens,
            onnx_path=ONNX_DIR / "llm_prefill.onnx",
        )
        export_llm_decode(
            lm_main, lm_head,
            n_layers=self.n_layers, hidden_size=self.hidden_size,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            text_mask_token_id=int(self.config.text_config.text_mask_token_id),
            onnx_path=ONNX_DIR / "llm_decode.onnx",
        )

    def build_engines(self, llm: bool = True):
        from .trt.build import build_vision, build_projector, build_llm
        assert self.grid_h is not None, "call export_engines() first"
        L_pre = self.grid_h * self.grid_w
        L_post = L_pre // 4
        build_vision(ONNX_DIR / "vision.onnx", TRT_DIR / "vision.engine", L_pre)
        build_projector(ONNX_DIR / "projector.onnx", TRT_DIR / "projector.engine", L_post)
        if llm and enable_llm_trt():
            build_llm(
                ONNX_DIR / "llm_prefill.onnx", ONNX_DIR / "llm_decode.onnx",
                TRT_DIR / "llm_prefill.engine", TRT_DIR / "llm_decode.engine",
                hidden_size=self.hidden_size, n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            )
        elif llm:
            print("[runner] LLM TRT build skipped (insufficient VRAM); orchestrator will use PyTorch fallback")

    def load_engines(self):
        from .trt.engine import TRTEngine
        if (TRT_DIR / "vision.engine").exists():
            self.vit_engine = TRTEngine(TRT_DIR / "vision.engine")
        if (TRT_DIR / "projector.engine").exists():
            self.proj_engine = TRTEngine(TRT_DIR / "projector.engine")
        if (TRT_DIR / "llm_prefill.engine").exists():
            self.prefill_engine = TRTEngine(TRT_DIR / "llm_prefill.engine")
        if (TRT_DIR / "llm_decode.engine").exists():
            self.decode_engine = TRTEngine(TRT_DIR / "llm_decode.engine")
        # Lock the processor's resize to match the baked engine resolution.
        if self.eng_img_w is not None:
            lock_processor_resolution(self.processor, self.eng_img_w, self.eng_img_h)
        print(f"[runner] engines loaded: "
              f"vit={'yes' if self.vit_engine else 'no'}, "
              f"proj={'yes' if self.proj_engine else 'no'}, "
              f"prefill={'yes' if self.prefill_engine else 'no (PT fallback)'}, "
              f"decode={'yes' if self.decode_engine else 'no (PT fallback)'}")

    # ----- inference -----
    def _detect_via_trt(self, image: Image.Image, prompt: str,
                         max_new_tokens: int, generation_mode: str) -> Tuple[List, str]:
        """TRT path: vision→projector engines, then LLM via prefill+decode engines (or
        PyTorch LM fallback when LLM engines absent).

        The TRT vision engine has a fixed input shape baked at export time, so we
        cannot hand it a native-resolution PIL. We letterbox (aspect-preserving
        resize + center-pad) to the engine resolution, then undo the letterbox
        transform on the box coordinates so callers see boxes in ORIGINAL image
        pixel space. Returns boxes in original-image coords (NOT engine coords).
        """
        if self.eng_img_w is None:
            raise RuntimeError("Runner not initialised; call .export_engines() + .build_engines() + .load_engines() first")
        img_eng, scale, pad_x, pad_y = _letterbox(image, self.eng_img_w, self.eng_img_h)
        enc = self._processor_call(img_eng, prompt)
        gh = enc["image_grid_hws"]
        if (int(gh[0, 0]), int(gh[0, 1])) != (self.grid_h, self.grid_w):
            raise RuntimeError(
                f"processor produced grid_hws=({int(gh[0,0])},{int(gh[0,1])}) but engine baked for "
                f"({self.grid_h},{self.grid_w}). Check that eng_img_w/eng_img_h are multiples of "
                f"merge_kernel_size*patch_size and that processor.image_processor.in_token_limit "
                f">= {self.grid_h * self.grid_w} (see lock_processor_resolution())."
            )
        px_np = enc["pixel_values"].detach().cpu().numpy().astype(np.float16)
        ids_np = enc["input_ids"].detach().cpu().numpy().astype(np.int64)
        _, toks = self._gen.generate(
            px_np, ids_np,
            max_new_tokens=max_new_tokens, generation_mode=generation_mode,
            temperature=0.0, top_p=1.0, repetition_penalty=1.1,
        )
        text = self.tokenizer.decode(toks, skip_special_tokens=False)
        boxes_lb = parse_boxes(text, self.eng_img_w, self.eng_img_h)
        # Undo letterbox: letterbox_xy -> orig_xy
        boxes_orig = [
            ((x1 - pad_x) / scale, (y1 - pad_y) / scale,
             (x2 - pad_x) / scale, (y2 - pad_y) / scale)
            for (x1, y1, x2, y2) in boxes_lb
        ]
        return boxes_orig, text

    def _detect_via_pt(self, image: Image.Image, prompt: str,
                        max_new_tokens: int, generation_mode: str = "hybrid",
                        unpatched: bool = False) -> Tuple[List, str]:
        """PyTorch canonical generate(). Used as auto-fallback when TRT returns 0 boxes
        (only available if `model` is still resident on GPU).

        unpatched=True temporarily reverts apply_vision_patches() so the model runs
        with the ORIGINAL canonical vision functions. Used as the 3rd-tier fallback
        when both patched-TRT and patched-PT produce degenerate output, to isolate
        whether the patches are responsible.

        Unlike the TRT path, `model.generate()` is fully dynamic — we hand the
        native-resolution PIL straight to the processor and let it decide the grid.
        Returns boxes in ORIGINAL image pixel space.
        """
        if self.model is None:
            return [], ""
        from .patches import restore_vision_patches, apply_vision_patches
        new_snap = None
        if unpatched and self.patches_snapshot is not None:
            restore_vision_patches(self.model, self.patches_snapshot)
        try:
            enc = self._processor_call(image, prompt)
            with torch.inference_mode():
                out = self.model.generate(
                    pixel_values=enc["pixel_values"], input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], image_grid_hws=enc["image_grid_hws"],
                    tokenizer=self.tokenizer, max_new_tokens=max_new_tokens, use_cache=True,
                    generation_mode=generation_mode, do_sample=False, repetition_penalty=1.1,
                    verbose=False,
                )
            ot = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(ot):
                text = self.tokenizer.decode(ot[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            else:
                text = str(ot)
        finally:
            # Re-apply patches so subsequent TRT calls (which expect patched modules) work.
            if unpatched and self.patches_snapshot is not None:
                new_snap = apply_vision_patches(self.model, verbose=False)
                self.patches_snapshot = new_snap
        orig_w, orig_h = image.size
        return parse_boxes(text, orig_w, orig_h), text

    def detect(self, image, prompt: str = "Locate all the instances that matches the following description: object.",
               max_new_tokens: int = 128, generation_mode: str = "hybrid",
               diagnostic: bool = True, auto_fallback_to_pt: bool = True,
               verbose: bool = False) -> Tuple[List, str]:
        """Single-image inference. Returns (boxes, raw_decoded_text).

        boxes are (x1, y1, x2, y2) in original image pixel space.

        Parameters
        ----------
        diagnostic         : write WORK/'last_inference.txt' (raw text + metadata).
                             On 0 boxes also prints a one-line preview to stdout so
                             notebook staleness can't hide the diagnostic.
        auto_fallback_to_pt: when TRT returns 0 boxes AND `self.model` is resident,
                             automatically retry via canonical model.generate(). If
                             PT returns boxes, use that result. This converts the
                             two-step manual diagnose-and-fall-back I was previously
                             pushing into notebook cells into a single transparent call.
        verbose            : always print raw text + token count, not just on failure.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        orig_w, orig_h = image.size

        # Auto-canonicalize OOD prompts. The model is trained on a specific phrasing;
        # natural-language alternatives like "Detect all cats. Return bounding boxes."
        # cause MTP mode-collapse. We rewrite transparently.
        canonical_prompt, rewritten = canonicalize_prompt(prompt)
        if rewritten and (diagnostic or verbose):
            print(f"[detect] prompt auto-canonicalized:")
            print(f"          input:     {prompt!r}")
            print(f"          rewritten: {canonical_prompt!r}")
        elif not rewritten and "matches the following description:" not in prompt.lower():
            print(f"[detect] WARN: prompt is not in canonical form; model may produce")
            print(f"          degenerate output. Canonical form is:")
            print(f"          'Locate all the instances that matches the following description: <X>.'")
        prompt = canonical_prompt

        # 1) TRT path
        boxes_eng, text = self._detect_via_trt(image, prompt, max_new_tokens, generation_mode)
        path_used = "trt"

        # 2) Auto-fallback to PyTorch (patched, same vision-patch code as TRT) if TRT 0
        pt_text = None
        unpatched_text = None
        if len(boxes_eng) == 0 and auto_fallback_to_pt and self.model is not None:
            if verbose or diagnostic:
                print(f"[detect] TRT returned 0 boxes; auto-falling back to PyTorch canonical generate()...")
            pt_boxes_eng, pt_text = self._detect_via_pt(image, prompt, max_new_tokens, generation_mode)
            if len(pt_boxes_eng) > 0:
                if verbose or diagnostic:
                    print(f"[detect] PyTorch (patched) returned {len(pt_boxes_eng)} boxes. Using PT result.")
                boxes_eng, text = pt_boxes_eng, pt_text
                path_used = "pt_patched"
            else:
                # 3) Both patched paths failed — try unpatched canonical PyTorch.
                # If THAT works, our vision patches are the culprit; if not, the
                # issue is upstream of the patches entirely (prompt/template/model).
                if self.patches_snapshot is not None:
                    if verbose or diagnostic:
                        print(f"[detect] PT (patched) also 0. Trying CANONICAL (patches reverted)...")
                    unpatched_boxes_eng, unpatched_text = self._detect_via_pt(
                        image, prompt, max_new_tokens, generation_mode, unpatched=True,
                    )
                    if len(unpatched_boxes_eng) > 0:
                        if verbose or diagnostic:
                            print(f"[detect] CANONICAL (unpatched) returned {len(unpatched_boxes_eng)} boxes!")
                            print(f"[detect] ⚠️  Vision patches are degrading the model.  "
                                  f"Patched ONNX/TRT engines were built from a broken state — re-export needed.")
                        boxes_eng, text = unpatched_boxes_eng, unpatched_text
                        path_used = "pt_unpatched_canonical"
                    else:
                        if verbose or diagnostic:
                            print(f"[detect] CANONICAL (unpatched) also 0. Issue is upstream of patches "
                                  f"(prompt / chat template / image preprocessing / model).")
                # Compose the diagnostic-dump text from all attempted paths
                segments = [f"### TRT (path=trt, patched) ###\n{text}"]
                if pt_text is not None:
                    segments.append(f"### PyTorch (patched canonical) ###\n{pt_text}")
                if unpatched_text is not None:
                    segments.append(f"### PyTorch (CANONICAL, patches reverted) ###\n{unpatched_text}")
                text = "\n\n".join(segments) + "\n"

        # Boxes are already in original-image pixel space (TRT undoes letterbox,
        # PT parses with orig_w/orig_h). No further scaling needed.
        boxes = boxes_eng

        # 4) Diagnostics — record what the PT-canonical path actually sees so we can
        # verify against canonical inference. Native image (NO pre-resize); whatever
        # grid_hws the processor produces is what the model saw.
        if diagnostic or verbose:
            try:
                rendered_text = ""
                token_ids = []
                n_img_tokens_actual = 0
                diag_grid_h, diag_grid_w = 0, 0
                try:
                    msg = [{"role":"user","content":[{"type":"image","image":image},
                                                       {"type":"text","text":prompt}]}]
                    rendered_text = self.processor.py_apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    imgs_d, vids_d = self.processor.process_vision_info(msg)
                    enc_d = self.processor(text=[rendered_text], images=imgs_d, videos=vids_d, return_tensors="pt")
                    token_ids = enc_d["input_ids"][0].tolist()
                    n_img_tokens_actual = int(
                        (enc_d["input_ids"] == int(self.config.image_token_index)).sum()
                    )
                    gh_d = enc_d.get("image_grid_hws")
                    if gh_d is not None:
                        gh_arr = gh_d.tolist() if hasattr(gh_d, "tolist") else gh_d
                        diag_grid_h, diag_grid_w = int(gh_arr[0][0]), int(gh_arr[0][1])
                except Exception as _e:
                    rendered_text = f"<failed to render prompt: {_e}>"

                (WORK / "last_inference.txt").write_text(
                    f"# prompt:           {prompt}\n"
                    f"# path:             {path_used}\n"
                    f"# boxes:            {len(boxes)}\n"
                    f"# generation_mode:  {generation_mode}\n"
                    f"# orig_img:         {orig_w}x{orig_h}\n"
                    f"# eng_img (TRT):    {self.eng_img_w}x{self.eng_img_h}\n"
                    f"# diag_grid_hws:    ({diag_grid_h},{diag_grid_w})  [from processor on NATIVE image]\n"
                    f"# eng_grid_hws:     ({self.grid_h},{self.grid_w})  [baked into TRT engine]\n"
                    f"# image_token_idx:  {int(self.config.image_token_index)}\n"
                    f"# image_tokens_in_input_ids: {n_img_tokens_actual}\n"
                    f"# input_ids_len:    {len(token_ids)}\n"
                    f"# input_ids_first_20:  {token_ids[:20]}\n"
                    f"# input_ids_last_20:   {token_ids[-20:]}\n"
                    f"\n=== rendered prompt (chat template applied) ===\n"
                    f"{rendered_text}\n"
                    f"\n=== model output ===\n{text}\n"
                )
            except Exception:
                pass

        if verbose:
            preview = text[:600].replace("\n", "  ")
            print(f"[detect] path={path_used} boxes={len(boxes)} chars={len(text)}")
            print(f"  raw: {preview}{'...' if len(text)>600 else ''}")
        elif diagnostic and len(boxes) == 0:
            preview = text[:400].replace("\n", "  ")
            print(f"[detect] 0 boxes detected (path={path_used}).  Raw output ({len(text)} chars):")
            print(f"  {preview}{'...' if len(text)>400 else ''}")
            if "<box>" not in text:
                print(f"[detect] No <box> tags in output. Full dump at {WORK / 'last_inference.txt'}")
            else:
                print(f"[detect] WARN: <box> present in raw but parse_boxes returned []. Check the regex.")

        # On Colab, auto-trigger a browser download of the diagnostic dump when
        # detection fails so the user can share it with the maintainer without
        # having to navigate the Files panel manually.
        if diagnostic and len(boxes) == 0:
            self._maybe_trigger_colab_download(WORK / "last_inference.txt")

        return boxes, text

    @staticmethod
    def _maybe_trigger_colab_download(path) -> None:
        """If running on Colab, trigger a browser download of `path`. No-op elsewhere."""
        try:
            from google.colab import files as _gfiles  # type: ignore
        except ImportError:
            return
        try:
            _gfiles.download(str(path))
            print(f"[detect] auto-downloaded {path} to your local Downloads folder")
        except Exception as e:
            print(f"[detect] download trigger failed (file still at {path}): {e}")
