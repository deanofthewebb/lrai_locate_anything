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

# Common irregular plurals. LocateAnything-3B is sensitive to singular vs plural
# phrasing — "cat" returns 1 box (the dominant cat), "cats" returns all of them.
# We default to plural for higher recall on every detect() call.
_IRREGULAR_PLURALS = {
    "person": "people", "child": "children", "mouse": "mice",
    "man": "men", "woman": "women", "foot": "feet", "tooth": "teeth",
    "goose": "geese", "sheep": "sheep", "deer": "deer",
    "fish": "fish", "ox": "oxen",
    "luggage": "luggage", "equipment": "equipment", "furniture": "furniture",
}


def _pluralize_one(word: str) -> str:
    """English heuristic pluralization for a single noun. Preserves
    already-plural words and known mass nouns / irregulars. Falls back to
    appending 's' for the common case."""
    if not word:
        return word
    lower = word.lower()
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    # Already plural-shaped (ends in s/es/ies/ces or known plural suffixes)
    if (lower.endswith("s") or lower.endswith("ies") or lower.endswith("es")
            or lower in _IRREGULAR_PLURALS.values()):
        return word
    # consonant + y -> -ies (city -> cities, but not toy -> toys)
    if len(word) >= 2 and word[-1].lower() == "y" and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    # x/s/z/ch/sh -> +es (box -> boxes, bus -> buses, watch -> watches)
    if lower.endswith(("x", "ch", "sh", "ss", "z")):
        return word + "es"
    # Default: +s
    return word + "s"


def pluralize_target(target: str) -> str:
    """Pluralize the target portion of a canonical prompt. Handles multi-class
    targets joined by '</c>'."""
    if "</c>" in target:
        return "</c>".join(_pluralize_one(t.strip()) for t in target.split("</c>"))
    # Single-class: target may itself be a phrase like "the red cars" — only
    # pluralize the head noun (last word) to avoid corrupting modifiers.
    parts = target.strip().split()
    if not parts:
        return target
    parts[-1] = _pluralize_one(parts[-1])
    return " ".join(parts)


def canonicalize_prompt(prompt: str) -> Tuple[str, bool]:
    """Rewrite natural-language prompts to the model's canonical training phrasing.

    LocateAnything-3B was trained exclusively on prompts of the form
        "Locate all the instances that matches the following description: X."
    where X is a single category or "<cat1></c><cat2>" for multi-class. OOD
    phrasings like "Detect all cats. Return bounding boxes." cause the MTP head
    to mode-collapse on the mask token. This helper extracts the object
    description from common natural-language phrasings and reformats to canonical.

    By default we also PLURALIZE the target so 'cat' becomes 'cats' — the model
    is plural-sensitive and singular forms return only the dominant instance.
    For locate-the-single-X intent, pre-canonicalize manually and skip this
    helper. See pluralize_target() for the heuristic.

    Returns (rewritten_prompt, was_rewritten).
    """
    p = prompt.strip()
    if "matches the following description:" in p.lower():
        # Already in canonical shape; we ONLY pluralize the target for recall.
        m = re.search(r"matches the following description:\s*(.+?)\.?\s*$", p, re.I)
        if not m:
            return p, False
        target = m.group(1).strip().rstrip(".")
        plural = pluralize_target(target)
        if plural == target:
            return p, False
        rewritten = re.sub(
            r"(matches the following description:\s*)(.+?)\.?\s*$",
            lambda mm: mm.group(1) + plural + ".",
            p, flags=re.I,
        )
        return rewritten, True
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
    # Pluralize for higher recall — see docstring.
    target = pluralize_target(target)
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
    # BF16 binding helpers: when an engine binding is bf16, pass a torch
    # bf16 tensor or a uint16 buffer (NEVER np.float16 — that would
    # silently reinterpret 1-5-10 fp16 bytes as 1-8-7 bf16 bytes and
    # produce garbage). engine.py's _coerce_input enforces this.
    @staticmethod
    def _np16_to_bf16_torch(arr_np):
        """View a np.float16 buffer as torch.bfloat16 byte-equivalent? No —
        the bit layouts differ. We must go through a real precision cast
        via float32. Use this for the legacy fp16 → bf16 boundary."""
        return torch.from_numpy(arr_np.astype(np.float32)).to(torch.bfloat16)

    @staticmethod
    def _bf16_uint16_to_torch(arr_u16):
        """Reinterpret a uint16 buffer as torch.bfloat16 (zero-copy bytes)."""
        return torch.from_numpy(arr_u16).view(torch.bfloat16)

    def _vision(self, px: np.ndarray) -> np.ndarray:
        # The vision engine is fixed-resolution; px MUST be (L_pre_fixed, 3, 14, 14).
        # Currently vit_engine I/O is fp16 (we left vision/projector at fp16 to
        # keep build time down). If the binding is BF16 we accept uint16 storage.
        pre = self.r.vit_engine({"pixel_values": px})["vit_feats"]  # (L_pre_fixed, 1152)
        gh = np.array([[self.r.grid_h, self.r.grid_w]], dtype=np.int32)
        return python_patch_merger(pre, gh, kh=2, kw=2).astype(np.float16)

    def _project(self, x: np.ndarray) -> np.ndarray:
        return self.r.proj_engine({"vit_feats_4x": x})["proj_feats"]

    def _prefill_trt(self, ids, vf, pos, att):
        # If the prefill engine's visual_features binding is BF16, pass a
        # torch.bfloat16 tensor (engine.py views it as uint16 zero-copy).
        # vf comes from _project (fp16 numpy from the fp16 projector engine).
        # Bridge via real cast through fp32.
        if self.r.prefill_engine.is_bf16.get("visual_features", False):
            vf_in = self._np16_to_bf16_torch(vf)
        else:
            vf_in = vf.astype(np.float16)
        o = self.r.prefill_engine({
            "input_ids":       ids.astype(np.int64),
            "visual_features": vf_in,
            "position_ids":    pos.astype(np.int64),
            "attention_mask":  att.astype(np.int64),
        })
        past = []
        for i in range(self.r.n_layers):
            past.append(o[f"present_k_{i}"])
            past.append(o[f"present_v_{i}"])
        return o["logits"], past

    def _decode_trt(self, ids, pos, att, past, branch: str = "mtp"):
        """Run the TRT decode engine. `branch` selects which of the two engines
        to use — the AR/MTP attention branch is constant-folded into each ONNX
        graph at trace time (modeling_qwen2.py:1279 / dynamo=False), so we must
        route by the input's semantic role:
          branch='mtp' → llm_decode.engine, traced with `[real_last, mask×5]`
                          (SDLM block-mask path). Use for the MTP forward.
          branch='ar'  → llm_decode_ar.engine, traced with non-mask input_ids
                          (canonical AR path). Use for KV-rebuild + AR fallback.
        Picking the wrong engine produces correct STRUCTURE tokens but corrupt
        content because the attention mask topology baked into the engine
        doesn't match the runtime input semantics.

        past_k_*/past_v_* may be uint16 (BF16 storage) or fp16 numpy depending
        on the engine binding. We accept both forms verbatim — engine.py's
        _coerce_input enforces type safety.
        """
        if branch == "mtp":
            engine = self.r.decode_engine
        elif branch == "ar":
            engine = self.r.decode_engine_ar
            if engine is None:
                raise RuntimeError(
                    "_decode_trt(branch='ar') called but decode_engine_ar is None. "
                    "Re-export with the dual-engine pipeline: "
                    "runner._wipe_stale_artifacts('add decode_ar'); "
                    "runner.export_engines(); runner.build_engines(); runner.load_engines()."
                )
        else:
            raise ValueError(f"branch must be 'mtp' or 'ar'; got {branch!r}")
        feed = {
            "input_ids":      ids.astype(np.int64),
            "position_ids":   pos.astype(np.int64),
            "attention_mask": att.astype(np.int64),
        }
        # past tensors already carry the right dtype from the previous engine
        # call (uint16 for BF16 bindings, fp16 numpy otherwise). Pass through.
        for i in range(self.r.n_layers):
            feed[f"past_k_{i}"] = past[2 * i]
            feed[f"past_v_{i}"] = past[2 * i + 1]
        o = engine(feed)
        nxt = []
        for i in range(self.r.n_layers):
            nxt.append(o[f"present_k_{i}"])
            nxt.append(o[f"present_v_{i}"])
        return o["logits"], nxt

    @staticmethod
    def _kv_np_to_torch(arr_np):
        """Past KV may come from a TRT engine as uint16 (BF16 storage) OR
        fp16 numpy. Reconstruct a torch tensor on the model's device + dtype.
        """
        if arr_np.dtype == np.uint16:
            # BF16 byte storage → torch.bfloat16 (zero-copy view)
            return torch.from_numpy(arr_np).view(torch.bfloat16).cuda().to(REF_DTYPE)
        # fp16 numpy: real precision cast through float32 to land at REF_DTYPE
        return torch.from_numpy(arr_np.astype(np.float32)).cuda().to(REF_DTYPE)

    @staticmethod
    def _kv_torch_to_engine(t, want_uint16: bool):
        """Convert a PT kv tensor to the engine's I/O format. want_uint16=True
        means the next engine expects BF16 storage as uint16."""
        if want_uint16:
            return t.contiguous().view(torch.uint16).cpu().numpy()
        return t.float().cpu().numpy().astype(np.float16)

    @staticmethod
    def _logits_to_torch_float(arr_np):
        """Convert engine-output logits to torch float32 ready for sample_tokens.
        Handles both BF16 (uint16 storage) and FP16 numpy buffers."""
        if arr_np.dtype == np.uint16:
            # BF16 byte storage → torch.bfloat16 view → float32 cast
            return torch.from_numpy(arr_np).view(torch.bfloat16).float()
        return torch.from_numpy(arr_np.astype(np.float32))

    def _prefill_pt(self, ids, vf, pos, att):
        with torch.inference_mode():
            i = torch.from_numpy(ids).long().cuda()
            # vf can be fp16 numpy (from fp16 projector) — cast through fp32
            # to land at the model's bf16 cleanly.
            if isinstance(vf, np.ndarray) and vf.dtype == np.uint16:
                v = torch.from_numpy(vf).view(torch.bfloat16).cuda().to(REF_DTYPE)
            else:
                v = torch.from_numpy(np.asarray(vf, dtype=np.float32)).cuda().to(REF_DTYPE)
            p = torch.from_numpy(pos).long().cuda()
            m = torch.from_numpy(att).long().cuda()
            o = self.r.model.language_model.model(
                input_ids=i, visual_features=v,
                image_token_index=int(self.r.config.image_token_index),
                position_ids=p, attention_mask=m,
                use_cache=True, return_dict=True,
            )
            logits = self.r.model.language_model.lm_head(o.last_hidden_state).float().cpu().numpy().astype(np.float16)
            # Match the engine binding format the NEXT call will use. If TRT
            # decode engines are bf16-bound, store kv as uint16; else fp16.
            want_uint16 = bool(self.r.decode_engine and self.r.decode_engine.is_bf16.get("past_k_0", False))
            past = []
            for (k, vv) in o.past_key_values:
                past.append(self._kv_torch_to_engine(k, want_uint16))
                past.append(self._kv_torch_to_engine(vv, want_uint16))
        return logits, past

    def _decode_pt(self, ids, pos, att, past):
        with torch.inference_mode():
            i = torch.from_numpy(ids).long().cuda()
            p = torch.from_numpy(pos).long().cuda()
            m = torch.from_numpy(att).long().cuda()
            pkv = tuple(
                (self._kv_np_to_torch(past[2 * j]),
                 self._kv_np_to_torch(past[2 * j + 1]))
                for j in range(self.r.n_layers)
            )
            o = self.r.model.language_model.model(
                input_ids=i, position_ids=p, attention_mask=m,
                past_key_values=pkv, use_cache=True, return_dict=True,
            )
            logits = self.r.model.language_model.lm_head(o.last_hidden_state).float().cpu().numpy().astype(np.float16)
            want_uint16 = bool(self.r.decode_engine and self.r.decode_engine.is_bf16.get("past_k_0", False))
            nxt = []
            for (k, vv) in o.past_key_values:
                nxt.append(self._kv_torch_to_engine(k, want_uint16))
                nxt.append(self._kv_torch_to_engine(vv, want_uint16))
        return logits, nxt

    def _prefill(self, *a):
        return (self._prefill_trt if self.r.prefill_engine else self._prefill_pt)(*a)

    def _decode_mtp(self, ids, pos, att, past):
        """MTP-branch decode (input ends in text_mask_token_id). Uses the
        TRT decode_engine when available, else PT (PT re-evaluates the branch
        from runtime input so it auto-routes correctly)."""
        if self.r.decode_engine is not None:
            return self._decode_trt(ids, pos, att, past, branch="mtp")
        return self._decode_pt(ids, pos, att, past)

    def _decode_ar(self, ids, pos, att, past):
        """AR-branch decode (input has no text_mask_token_id at the last
        position). Uses the TRT decode_engine_ar when available, else PT."""
        if self.r.decode_engine_ar is not None:
            return self._decode_trt(ids, pos, att, past, branch="ar")
        return self._decode_pt(ids, pos, att, past)

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
                # MTP forward: input ends in mask_ids[-1] = text_mask_token_id,
                # so we MUST use the mtp-baked engine (block-mask attention).
                logits_o, nxt = self._decode_mtp(mtp_ids, pos, att, past)
                lt = self._logits_to_torch_float(logits_o)
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
                # CRITICAL: discard ALL MTP-block kv entries (computed under MASK
                # token inputs) and rebuild kv for the committed tokens via a
                # short AR forward pass through the decode engine. Without this,
                # subsequent iterations attend to mask-poisoned kv states and
                # produce multilingual garbage.
                #
                # Canonical (modeling_locateanything.py:_sample_token_in_mtp -> line
                # 484): `past_key_values = kv[:, :, :generated.shape[1], :]`
                # which truncates past_key_values back to the PRE-MTP length, then
                # the next iteration's prepare_inputs_for_generation feeds the
                # newly-committed tokens through the model to recompute their kv.
                # We replicate that here by truncating to P (pre-MTP) and re-doing
                # an AR forward over `toks` to populate kv at positions [P, P+k).
                past = [t[:, :, :P, :] for t in nxt]  # truncate to pre-MTP
                if pat.get("is_terminal") or int(self.r.TID["im_end_token_id"]) in toks:
                    break
                # Re-run committed tokens through the AR-branch decode engine to
                # refill kv with the correct (non-mask) attention states. The
                # mtp engine has the SDLM block-mask path baked in (export/llm.py
                # traces with text_mask_token_id at the last position); feeding
                # it real committed tokens would corrupt the KV. The ar engine
                # is traced with non-mask input_ids so modeling_qwen2.py:1279's
                # data-dependent branch evaluates to the canonical AR path at
                # trace time, baking that branch into the ONNX graph.
                rebuild_pos = np.arange(P, P + len(toks), dtype=np.int64)[None, :]
                rebuild_att = np.ones((1, P + len(toks)), dtype=np.int64)
                _, past = self._decode_ar(tnp, rebuild_pos, rebuild_att, past)
            else:
                last_id = generated[:, -1:]
                pos = np.array([[P]], dtype=np.int64)
                att = np.ones((1, P + 1), dtype=np.int64)
                # AR fallback: single real (non-mask) token → AR-branch engine.
                logits_o, nxt = self._decode_ar(last_id, pos, att, past)
                lt = self._logits_to_torch_float(logits_o)
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
        # decode_engine:    SDLM block-mask branch (correct for MTP `[last, mask×5]`).
        # decode_engine_ar: AR / non-block-mask branch (correct for AR steps and KV
        #                   rebuilds where input_ids has no text_mask_token_id at the
        #                   last position).
        # modeling_qwen2.py:1279's data-dependent branch is constant-folded into
        # each ONNX graph at trace time (dynamo=False), so we need BOTH engines.
        # _TRTGenerator routes to the correct one based on the input's last token.
        self.decode_engine = None
        self.decode_engine_ar = None

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
        force_reexport: bool = False,
    ) -> "LocateAnythingRunner":
        """Load model + (optionally) export and build engines on first run.

        For auto_export=True, supply a sample_image whose resolution will determine
        the baked engine size. If None, a tiny default image is used (grid 36x46).

        If `force_reexport=True` OR the model loader had to rescue lm_head (any
        cached engines on disk were built from a random lm_head and are corrupt),
        all stale ONNX + TRT artifacts are wiped before re-export.
        """
        ensure_nvidia_stack(verbose=False)
        model, tokenizer, processor, config, local, snap = load_locateanything_3b(
            local_dir=local_dir, model_id=model_id,
        )
        runner = cls(model, tokenizer, processor, config, local, patches_snapshot=snap)
        rescued = bool(getattr(model, "_locany_lm_head_was_rescued", False))
        if auto_export:
            current_fp = runner._model_fingerprint()
            stale_fp = runner._cached_artifacts_fingerprint()
            engines_on_disk = any(
                (TRT_DIR / f).exists()
                for f in ("vision.engine", "projector.engine", "llm_prefill.engine", "llm_decode.engine")
            )
            # Wipe if ANY of these is true:
            #   - explicit force_reexport
            #   - lm_head was rescued this load (model state changed)
            #   - cached fingerprint doesn't match the current model
            #   - engines exist on disk WITHOUT a fingerprint stamp (pre-fingerprint
            #     artifacts — assume stale because we have no way to verify they
            #     match the current model)
            reasons = []
            if force_reexport: reasons.append("force_reexport")
            if rescued:        reasons.append("lm_head rescued")
            if stale_fp is not None and stale_fp != current_fp:
                reasons.append(f"fingerprint mismatch (cached={stale_fp[:12]} current={current_fp[:12]})")
            elif stale_fp is None and engines_on_disk:
                reasons.append("engines on disk without fingerprint (pre-fingerprint artifacts, assumed stale)")
            if reasons:
                runner._wipe_stale_artifacts(reason="; ".join(reasons))
            runner.export_engines(sample_image=sample_image, sample_prompt=sample_prompt)
            runner.build_engines()
            runner.load_engines()
            runner._write_artifacts_fingerprint(current_fp)
        return runner

    # Bump this whenever the engine build pipeline changes in a way that
    # invalidates pre-existing engines (dtype flags, profile shapes, ONNX
    # graph topology, etc.). Combined with the model-state hash to form the
    # full fingerprint, so engines built with old flags get wiped on next
    # auto_export even when the model weights haven't changed.
    _ENGINE_BUILD_VERSION = "v4:strongly-typed-bf16"

    def _model_fingerprint(self) -> str:
        """Stable hash of the in-memory model state PLUS the engine build version.

        We hash:
          - embed_tokens.weight (changes on every load if loading is broken,
            and is the most important signal — random init has std=0.02,
            trained has std=0.024)
          - lm_head.weight       (same; tied or not, its values matter)
          - mlp1 projector weights (vision-to-LM bridge)
          - vision_model.patch_embed conv weight (entry point of vision tower)
          - _ENGINE_BUILD_VERSION (forces invalidation when build pipeline changes)
        We do NOT hash every param (too expensive); these are the 4 layers
        whose values would change between a broken random-init load and a
        good trained load. SHA1 over the first 1024 bytes of each tensor's
        raw bytes is enough to discriminate; full hash would be wasteful.
        """
        import hashlib
        h = hashlib.sha1()
        tensors_to_hash = [
            ("embed_tokens", self.model.language_model.model.embed_tokens.weight),
            ("lm_head",      self.model.language_model.lm_head.weight),
            ("mlp1",         next(self.model.mlp1.parameters())),
            ("vit_patch",    self.model.vision_model.patch_embed.proj.weight),
        ]
        for name, t in tensors_to_hash:
            h.update(name.encode())
            # First 1024 bytes via CPU float view (bf16 has no numpy dtype)
            view = t.detach().float().cpu().flatten()[:512].contiguous().numpy().tobytes()
            h.update(view)
        h.update(f"|build={self._ENGINE_BUILD_VERSION}".encode())
        return h.hexdigest()

    def _cached_artifacts_fingerprint(self) -> Optional[str]:
        """Read the fingerprint stamp written next to the engine artifacts on
        a prior successful export. Returns None if no stamp exists."""
        fp_path = TRT_DIR / ".model_fingerprint"
        if fp_path.exists():
            try:
                return fp_path.read_text().strip()
            except Exception:
                return None
        return None

    def _write_artifacts_fingerprint(self, fingerprint: str) -> None:
        """Stamp the current model fingerprint next to the engines so the next
        session can detect if the cached artifacts came from a different model state."""
        TRT_DIR.mkdir(parents=True, exist_ok=True)
        (TRT_DIR / ".model_fingerprint").write_text(fingerprint)

    def _wipe_stale_artifacts(self, reason: str = "") -> None:
        """Remove cached ONNX + TRT artifacts. Used when the model state has changed
        in a way that invalidates anything baked from a prior session.
        """
        for d in (ONNX_DIR, TRT_DIR):
            if d.exists():
                # Wipe files AND the fingerprint stamp
                items = [p for p in d.iterdir() if p.is_file() or p.name.startswith(".")]
                if items:
                    print(f"[runner] wiping {len(items)} stale artifact(s) in {d} ({reason})")
                    for p in items:
                        p.unlink()
            d.mkdir(parents=True, exist_ok=True)

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

        Dtype contract:
          - Model is loaded as REF_DTYPE (bfloat16, canonical for PT inference).
          - TRT engines are fp16-built (numpy lacks bf16 for engine I/O).
          - We TEMPORARILY cast the model to fp16 for the duration of the ONNX
            exports, then restore the original dtype. PT inference (model.generate)
            runs at REF_DTYPE; only the trace-graph capture is fp16.
        """
        from .export import (
            export_vision, export_projector,
            export_llm_prefill, export_llm_decode, export_llm_decode_ar,
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

        # Sanity check the LM weights BEFORE we trace them into ONNX. If we
        # bake random-init weights into the engine, every TRT inference will
        # mode-collapse forever — and you can't tell from the engine file alone.
        # This guard ensures ONNX/TRT inherit only the real trained weights.
        # Trained Qwen2.5-3B embed_tokens has std ~0.024; Qwen2 random init
        # uses initializer_range=0.02. The gap is ~20% so the threshold is safe.
        embed_std = self.model.language_model.model.embed_tokens.weight.std().item()
        if embed_std < 0.022:
            raise RuntimeError(
                f"export_engines refusing to run: embed_tokens.weight.std()={embed_std:.5f} "
                f"is too close to Qwen2's initializer_range (0.020). The model is at "
                f"random init — exporting now would bake garbage into the ONNX/TRT "
                f"engines. Most likely cause: transformers>=5.0 silently failed to load "
                f"the checkpoint. Verify with:\n"
                f"  import transformers; print(transformers.__version__)\n"
                f"and ensure transformers<5.0 (e.g. 4.57.6)."
            )

        vit_h = self.config.vision_config.hidden_size
        vit_feat_dim = vit_h * 4

        # Mixed-precision export contract:
        #   - vision_model + mlp1 (projector): traced as fp16, engines built fp16
        #     (their cos_sim against PT is already 0.989/0.996 — not precision-
        #     sensitive enough to justify bf16 build complexity)
        #   - language_model (prefill, decode_mtp, decode_ar): traced as bf16,
        #     engines built bf16+fp16 with BuilderFlag.BF16 (Qwen2 attention
        #     scores need fp32's exponent range to match PT bf16)
        # Cast each submodule to its export precision around its own export,
        # then restore. PT inference still runs at the original REF_DTYPE (bf16).
        orig_dtype = next(self.model.parameters()).dtype
        try:
            # Vision + projector: cast to fp16 for trace
            if orig_dtype != torch.float16:
                print(f"[runner] casting vision_model + mlp1 {orig_dtype} -> fp16 for ONNX export")
                self.model.vision_model.to(dtype=torch.float16)
                self.model.mlp1.to(dtype=torch.float16)
            export_vision(self.model.vision_model, self.grid_h, self.grid_w, ONNX_DIR / "vision.onnx")
            export_projector(self.model.mlp1, vit_feat_dim, ONNX_DIR / "projector.onnx")
            if orig_dtype != torch.float16:
                print(f"[runner] restoring vision_model + mlp1 fp16 -> {orig_dtype}")
                self.model.vision_model.to(dtype=orig_dtype)
                self.model.mlp1.to(dtype=orig_dtype)

            lm_main = self.model.language_model.model
            lm_head = self.model.language_model.lm_head
            export_llm_prefill(
                lm_main, lm_head,
                image_token_index=int(self.config.image_token_index),
                n_layers=self.n_layers, hidden_size=self.hidden_size,
                n_img_tokens=n_img_tokens,
                onnx_path=ONNX_DIR / "llm_prefill.onnx",
            )
            # MTP-branch decode: SDLM block-mask path baked in by tracing with
            # text_mask_token_id at input_ids[0, -1].
            export_llm_decode(
                lm_main, lm_head,
                n_layers=self.n_layers, hidden_size=self.hidden_size,
                n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
                text_mask_token_id=int(self.config.text_config.text_mask_token_id),
                onnx_path=ONNX_DIR / "llm_decode.onnx",
            )
            # AR-branch decode: non-mask input_ids baked into the trace so
            # modeling_qwen2.py:1279's branch evaluates to the canonical AR path.
            # Required for the post-MTP KV-rebuild and the AR fallback step;
            # routing handled by _TRTGenerator.
            export_llm_decode_ar(
                lm_main, lm_head,
                n_layers=self.n_layers, hidden_size=self.hidden_size,
                n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
                onnx_path=ONNX_DIR / "llm_decode_ar.onnx",
            )
        finally:
            # Defensive: ensure both submodules are restored to the original
            # dtype even if an export raised mid-stream. PT inference relies
            # on the full model being at REF_DTYPE (bf16).
            if next(self.model.vision_model.parameters()).dtype != orig_dtype:
                self.model.vision_model.to(dtype=orig_dtype)
            if next(self.model.mlp1.parameters()).dtype != orig_dtype:
                self.model.mlp1.to(dtype=orig_dtype)

    def build_engines(self, llm: bool = True):
        """Build vision + projector engines, and optionally the LLM prefill/decode
        engines. If llm=True and enable_llm_trt() returns False (insufficient VRAM),
        we RAISE — no silent skip. Pass llm=False to explicitly opt out and run
        the LM via PyTorch on the same device.
        """
        from .trt.build import build_vision, build_projector, build_llm
        assert self.grid_h is not None, "call export_engines() first"
        L_pre = self.grid_h * self.grid_w
        L_post = L_pre // 4
        build_vision(ONNX_DIR / "vision.onnx", TRT_DIR / "vision.engine", L_pre)
        build_projector(ONNX_DIR / "projector.onnx", TRT_DIR / "projector.engine", L_post)
        if not llm:
            self._llm_engines_rebuilt_this_session = False
            return
        if not enable_llm_trt():
            raise RuntimeError(
                "build_engines: enable_llm_trt() returned False (insufficient VRAM "
                "headroom for LLM TRT build). Pass llm=False to explicitly skip the "
                "LLM engine and run the language model via PyTorch instead. "
                "Silent-skip is disabled — the choice must be intentional."
            )
        build_llm(
            ONNX_DIR / "llm_prefill.onnx", ONNX_DIR / "llm_decode.onnx",
            TRT_DIR / "llm_prefill.engine", TRT_DIR / "llm_decode.engine",
            hidden_size=self.hidden_size, n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            decode_ar_onnx=ONNX_DIR / "llm_decode_ar.onnx",
            decode_ar_engine=TRT_DIR / "llm_decode_ar.engine",
        )
        # Stamp the runner so load_engines() knows the .engine files on disk
        # were built FROM the in-memory (rescued or not) model this session,
        # not inherited from a prior session with a broken lm_head.
        self._llm_engines_rebuilt_this_session = True

    def load_engines(self, device_map: dict | None = None):
        """Load TRT engines into memory.

        device_map: optional dict mapping engine key -> CUDA device id, used to
            place engines across multiple GPUs when no single card has enough
            VRAM for all of them. Keys: "vision", "projector", "prefill",
            "decode_ar", "decode_mtp". Missing keys default to the current
            CUDA device (single-GPU legacy behavior).

            Example on a 10 GB + 12 GB box:
                runner.load_engines(device_map={"vision": 0, "projector": 0,
                                                "decode_ar": 0, "decode_mtp": 0,
                                                "prefill": 1})
            Cross-device data movement is automatic: TRTEngine round-trips I/O
            through host buffers between calls, so feeding output from one
            engine to another on a different GPU just works.
        """
        from .trt.engine import TRTEngine
        device_map = device_map or {}
        # Guard against loading STALE engines: if the loader rescued lm_head this
        # session AND we did NOT also rebuild the LLM engines this session, the
        # .engine files on disk came from a prior session with a broken lm_head.
        rescued = bool(getattr(self.model, "_locany_lm_head_was_rescued", False))
        rebuilt_this_session = bool(getattr(self, "_llm_engines_rebuilt_this_session", False))
        llm_engines_on_disk = any((TRT_DIR / f).exists() for f in ("llm_prefill.engine", "llm_decode.engine"))
        # The AR-branch decode engine is required since the bc7c6bd era — an old
        # cache may have llm_prefill+llm_decode but not llm_decode_ar.
        decode_ar_present = (TRT_DIR / "llm_decode_ar.engine").exists()
        if rescued and llm_engines_on_disk and not rebuilt_this_session:
            raise RuntimeError(
                "Cached TRT LLM engines exist on disk but the loaded model required "
                "lm_head tying (random-init lm_head detected) AND the LLM engines were "
                "NOT rebuilt in this session. The cached engines were built from the "
                "broken model and would produce mode-collapse. To recover:\n"
                "    runner._wipe_stale_artifacts('manual reset')\n"
                "    runner.export_engines()\n"
                "    runner.build_engines()\n"
                "    runner.load_engines()"
            )
        if llm_engines_on_disk and not decode_ar_present and not rebuilt_this_session:
            raise RuntimeError(
                "Cached TRT LLM engines exist on disk but llm_decode_ar.engine is "
                "MISSING. The decode-engine ONNX graph has the SDLM block-mask "
                "attention branch constant-folded; AR steps and KV rebuilds need a "
                "separate AR-branch-baked engine (llm_decode_ar.engine). Pre-dual-"
                "engine artifacts cannot serve correct TRT inference. Re-export:\n"
                "    runner._wipe_stale_artifacts('add decode_ar engine')\n"
                "    runner.export_engines()\n"
                "    runner.build_engines()\n"
                "    runner.load_engines()"
            )
        if (TRT_DIR / "vision.engine").exists():
            self.vit_engine = TRTEngine(TRT_DIR / "vision.engine", device_id=device_map.get("vision"))
        if (TRT_DIR / "projector.engine").exists():
            self.proj_engine = TRTEngine(TRT_DIR / "projector.engine", device_id=device_map.get("projector"))
        if (TRT_DIR / "llm_prefill.engine").exists():
            self.prefill_engine = TRTEngine(TRT_DIR / "llm_prefill.engine", device_id=device_map.get("prefill"))
        if (TRT_DIR / "llm_decode_ar.engine").exists():
            self.decode_engine_ar = TRTEngine(TRT_DIR / "llm_decode_ar.engine", device_id=device_map.get("decode_ar"))
        if (TRT_DIR / "llm_decode.engine").exists():
            self.decode_engine = TRTEngine(TRT_DIR / "llm_decode.engine", device_id=device_map.get("decode_mtp"))
        # Backfill engine resolution attrs from the vision engine's baked
        # pixel_values shape when load_engines() is called without a prior
        # export_engines() (e.g. loading pre-built engines from a different
        # session for verification or audit runs). The shape is
        # (L_pre, 3, 14, 14) where L_pre = grid_h * grid_w. We need the
        # individual grid_h and grid_w, not just L_pre — derive from the
        # baked aspect ratio. The merge kernel size is 2 (from L_post=L_pre/4),
        # so eng_img_w/h = grid_w*14 / grid_h*14.
        if self.eng_img_w is None and self.vit_engine is not None:
            try:
                px_shape = self.vit_engine.engine.get_tensor_profile_shape("pixel_values", 0)[1]
                L_pre = int(px_shape[0])
                # The projector engine's input shape gives us L_post = L_pre//4
                # and confirms the merge factor; the aspect ratio comes from the
                # baked grid which we read off the LLM prefill engine's vision
                # token count if available, else from a config default.
                # Simpler: look up grid (h,w) from runner config which knows
                # the baked resolution from export. Fallback: assume 36x46.
                grid_h, grid_w = None, None
                # Try the lock_processor_resolution-cached value, then config.
                cached = getattr(self, "_baked_grid_hws", None)
                if cached is not None:
                    grid_h, grid_w = cached
                else:
                    # Heuristic: prefer the (36, 46) bake used across the project
                    # for ~644x504 ISP frames. Verify L_pre matches grid_h*grid_w.
                    for cand_h, cand_w in ((36, 46), (32, 32), (24, 32), (28, 36)):
                        if cand_h * cand_w == L_pre:
                            grid_h, grid_w = cand_h, cand_w
                            break
                if grid_h is None:
                    raise RuntimeError(
                        f"load_engines(): vision engine L_pre={L_pre} does not match any "
                        f"known baked grid. Engine was built with a non-canonical resolution; "
                        f"set runner._baked_grid_hws=(h,w) explicitly before load_engines()."
                    )
                self.grid_h, self.grid_w = grid_h, grid_w
                self.eng_img_w = grid_w * 14
                self.eng_img_h = grid_h * 14
                print(f"[runner] backfilled grid_hws=({grid_h},{grid_w}) "
                      f"eng_img=({self.eng_img_w}x{self.eng_img_h}) from vision engine bake")
            except Exception as e:
                print(f"[runner] WARNING: could not backfill grid_h/grid_w from vision engine: {e}")
        # Lock the processor's resize to match the baked engine resolution.
        if self.eng_img_w is not None:
            lock_processor_resolution(self.processor, self.eng_img_w, self.eng_img_h)
        print(f"[runner] engines loaded: "
              f"vit={'yes' if self.vit_engine else 'no'}, "
              f"proj={'yes' if self.proj_engine else 'no'}, "
              f"prefill={'yes' if self.prefill_engine else 'no'}, "
              f"decode_mtp={'yes' if self.decode_engine else 'no'}, "
              f"decode_ar={'yes' if self.decode_engine_ar else 'no'}")

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
        # .float() before .numpy() to handle bfloat16 (no numpy dtype). Cast to
        # np.float16 at the boundary so it matches the fp16 TRT engine input.
        px_np = enc["pixel_values"].detach().float().cpu().numpy().astype(np.float16)
        ids_np = enc["input_ids"].detach().cpu().numpy().astype(np.int64)
        _, toks = self._gen.generate(
            px_np, ids_np,
            max_new_tokens=max_new_tokens, generation_mode=generation_mode,
            # Canonical kwargs per nvidia/LocateAnything-3B model card.
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
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
            raise RuntimeError(
                "_detect_via_pt called but self.model is None. The PyTorch model is "
                "required for this path. Reload via LocateAnythingRunner.from_pretrained()."
            )
        from .patches import restore_vision_patches, apply_vision_patches
        new_snap = None
        if unpatched and self.patches_snapshot is not None:
            restore_vision_patches(self.model, self.patches_snapshot)
        try:
            enc = self._processor_call(image, prompt)
            with torch.inference_mode():
                # Canonical generate kwargs per nvidia/LocateAnything-3B model-card
                # LocateAnythingWorker.predict(): do_sample=True, temperature=0.7,
                # top_p=0.9, repetition_penalty=1.1. Greedy (do_sample=False)
                # causes mode-collapse in MTP/hybrid generation.
                out = self.model.generate(
                    pixel_values=enc["pixel_values"], input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], image_grid_hws=enc["image_grid_hws"],
                    tokenizer=self.tokenizer, max_new_tokens=max_new_tokens, use_cache=True,
                    generation_mode=generation_mode,
                    do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
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
               path: str = "auto", diagnostic: bool = True, verbose: bool = False,
               ) -> Tuple[List, str]:
        """Single-image inference. Returns (boxes, raw_decoded_text).

        boxes are (x1, y1, x2, y2) in original image pixel space.

        Parameters
        ----------
        path        : "auto" (TRT if engines loaded else PT), "trt", or "pt".
                      Explicit choice — NO silent retry on failure.
        diagnostic  : write WORK/'last_inference.txt' with the rendered prompt,
                      token ids, grid_hws, and raw model output. NOT a fallback —
                      always runs the same single path. Used to inspect what the
                      model actually saw.
        verbose     : print raw text + token count.

        Raises
        ------
        RuntimeError: prompt couldn't be canonicalized to the training phrasing,
                      requested path is unavailable, or the model produced no
                      <box> tags. NO silent return-zero — if you want to inspect
                      a failure, call .diagnose() explicitly.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        orig_w, orig_h = image.size

        # Canonicalize the prompt. The model is trained on a specific phrasing;
        # any other phrasing produces mode-collapse. If canonicalize_prompt can't
        # rewrite the prompt to canonical form, REFUSE to run — no silent garbage.
        canonical_prompt, rewritten = canonicalize_prompt(prompt)
        if rewritten and (diagnostic or verbose):
            print(f"[detect] prompt auto-canonicalized:\n          input:     {prompt!r}\n          rewritten: {canonical_prompt!r}")
        if not rewritten and "matches the following description:" not in canonical_prompt.lower():
            raise RuntimeError(
                f"Prompt is not in canonical form and could not be auto-rewritten: {prompt!r}\n"
                f"The model only accepts: "
                f"'Locate all the instances that matches the following description: <X>.'\n"
                f"Pass a canonical prompt, or extend canonicalize_prompt() to recognize "
                f"the phrasing you intend to use."
            )
        prompt = canonical_prompt

        # Pick the path explicitly. No silent retry on failure.
        if path == "auto":
            if self.prefill_engine is not None and self.decode_engine is not None and self.vit_engine is not None:
                path = "trt"
            elif self.model is not None:
                path = "pt"
            else:
                raise RuntimeError(
                    "No inference path available: TRT engines not loaded AND PyTorch "
                    "model is None. Call .from_pretrained(auto_export=True) or load "
                    "engines manually."
                )
        if path == "trt":
            boxes, text = self._detect_via_trt(image, prompt, max_new_tokens, generation_mode)
        elif path == "pt":
            boxes, text = self._detect_via_pt(image, prompt, max_new_tokens, generation_mode)
        else:
            raise ValueError(f"detect(path=...) must be 'auto', 'trt', or 'pt'; got {path!r}")

        # When in diagnostic mode AND the primary path returned 0 boxes, ALSO run
        # the other path so the dump shows both outputs side-by-side AND we can
        # capture layer-by-layer numerical diffs (vision/projector/prefill logits).
        # This is NOT a silent fallback — we still raise on failure; this just
        # populates the diagnostic file with comparison data.
        comparison_out = None
        layer_diag_text = ""
        if diagnostic and len(boxes) == 0:
            other = "pt" if path == "trt" else "trt"
            try:
                if other == "pt" and self.model is not None:
                    cmp_boxes, cmp_text = self._detect_via_pt(image, prompt, max_new_tokens, generation_mode)
                    comparison_out = (other, cmp_boxes, cmp_text, None)
                elif other == "trt" and self.vit_engine and self.proj_engine and self.prefill_engine and self.decode_engine:
                    cmp_boxes, cmp_text = self._detect_via_trt(image, prompt, max_new_tokens, generation_mode)
                    comparison_out = (other, cmp_boxes, cmp_text, None)
            except Exception as _e:
                comparison_out = (other, [], "", repr(_e))
            try:
                layer_diag_text = self._layer_diagnostic(image, prompt)
            except Exception as _e:
                layer_diag_text = f"<layer diagnostic failed: {_e!r}>"

        if diagnostic or verbose:
            self._write_diagnostic(image, prompt, path, boxes, text, max_new_tokens, generation_mode,
                                     orig_w, orig_h, comparison=comparison_out, layer_diag=layer_diag_text)

        if verbose:
            preview = text[:600].replace("\n", "  ")
            print(f"[detect] path={path} boxes={len(boxes)} chars={len(text)}")
            print(f"  raw: {preview}{'...' if len(text)>600 else ''}")

        if len(boxes) == 0:
            cmp_msg = ""
            if comparison_out is not None:
                _op, _ob, _ot, _oe = comparison_out
                if _oe is None:
                    cmp_msg = f" Compared {_op} path: {len(_ob)} boxes; first 200: {_ot[:200]!r}."
                else:
                    cmp_msg = f" Compared {_op} path raised: {_oe}."
            raise RuntimeError(
                f"detect() returned 0 boxes on path={path}. The model produced "
                f"{len(text)} chars; first 200: {text[:200]!r}.{cmp_msg} "
                f"Diagnostic dump written to {WORK / 'last_inference.txt'}."
            )
        return boxes, text

    def _layer_diagnostic(self, image, prompt: str) -> str:
        """Run vision encoder, projector, and prefill through both TRT and PT, and
        report shape/mean/std/min/max/NaN + max-abs-diff + cosine-similarity between
        the two implementations at each layer. Pinpoints which stage diverges.

        Returns a string ready to embed in last_inference.txt.
        """
        if self.model is None:
            return "# layer diagnostic SKIPPED (PT model not loaded)\n"
        if not (self.vit_engine and self.proj_engine and self.prefill_engine):
            return "# layer diagnostic SKIPPED (TRT engines not loaded)\n"

        lines = ["", "=" * 60, "LAYER DIAGNOSTIC (TRT vs PT, numerical)", "=" * 60]

        def _stats(name: str, t: torch.Tensor) -> str:
            t = t.detach().float().cpu()
            has_nan = bool(torch.isnan(t).any())
            has_inf = bool(torch.isinf(t).any())
            return (f"{name:18s}  shape={tuple(t.shape)}  "
                    f"mean={t.mean().item():+.5f}  std={t.std().item():.5f}  "
                    f"min={t.min().item():+.4f}  max={t.max().item():+.4f}  "
                    f"NaN={has_nan} Inf={has_inf}")

        def _compare(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
            a = a.detach().float().cpu().flatten()
            b = b.detach().float().cpu().flatten()
            n = min(a.numel(), b.numel())
            a, b = a[:n], b[:n]
            diff = (a - b).abs()
            cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            return (f"{name:18s}  max|TRT-PT|={diff.max().item():.5f}  "
                    f"mean|diff|={diff.mean().item():.5f}  "
                    f"cos_sim={cos:.6f}")

        # Set up inputs identical for both paths via letterbox->processor
        img_lb, _, _, _ = _letterbox(image, self.eng_img_w, self.eng_img_h)
        enc = self._processor_call(img_lb, prompt)
        pixel_values = enc["pixel_values"]
        grid_hws = enc["image_grid_hws"]
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # ---------------- VISION ----------------
        # Canonical (modeling_vit.MoonVitPretrainedModel.forward) ALREADY applies
        # the 2x2 patch merger internally, so vision_model returns the same
        # (L_post, D) tensor that our TRT [vit_engine + python_patch_merger]
        # produces. Compare those directly — no extra merger on the PT side.
        px_np = pixel_values.detach().float().cpu().numpy().astype(np.float16)
        with torch.inference_mode():
            pt_vit_out = self.model.vision_model(
                pixel_values=pixel_values.to(self.model.dtype), grid_hws=grid_hws,
            )
        # vision_model returns either a Tensor or a list-of-Tensors (per-image).
        # The canonical generate concatenates the list. With batch=1, it's a
        # single (L_post, D) tensor.
        if hasattr(pt_vit_out, "last_hidden_state"):
            pt_vit_post = pt_vit_out.last_hidden_state
        elif isinstance(pt_vit_out, (list, tuple)):
            pt_vit_post = torch.cat([t for t in pt_vit_out], dim=0)
        else:
            pt_vit_post = pt_vit_out
        trt_vit_np = self._gen._vision(px_np)  # (L_post, D), uses python_patch_merger
        trt_vit_t = torch.from_numpy(trt_vit_np.astype(np.float32))
        pt_vit_t = pt_vit_post.detach().float().cpu()  # already post-merger
        lines.append("")
        lines.append("[vision encoder output, post-2x2 merger]")
        lines.append(_stats("TRT vit+merger", trt_vit_t))
        lines.append(_stats("PT  vision_model", pt_vit_t))
        lines.append(_compare("vit diff", trt_vit_t, pt_vit_t))

        # ---------------- PROJECTOR ----------------
        trt_proj_np = self._gen._project(trt_vit_np)  # uses TRT proj_engine on TRT vit
        trt_proj_t = torch.from_numpy(trt_proj_np.astype(np.float32))
        with torch.inference_mode():
            pt_proj_t = self.model.mlp1(pt_vit_post.to(self.model.dtype).to(self.model.device))
        lines.append("")
        lines.append("[projector output (visual features fed to LM)]")
        lines.append(_stats("TRT proj",  trt_proj_t))
        lines.append(_stats("PT  mlp1",  pt_proj_t))
        lines.append(_compare("proj diff", trt_proj_t, pt_proj_t))

        # ---------------- PREFILL (logits at last position) ----------------
        n_img_in_ids = int((input_ids == int(self.config.image_token_index)).sum())
        ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
        pos_np = np.arange(ids_np.shape[1], dtype=np.int64)[None, :]
        att_np = np.ones_like(ids_np, dtype=np.int64)
        trt_logits, _trt_past = self._gen._prefill_trt(ids_np, trt_proj_np[:n_img_in_ids], pos_np, att_np)
        trt_logits_t = torch.from_numpy(trt_logits.astype(np.float32))  # (1, S, V)
        trt_last_logits = trt_logits_t[0, -1]  # (V,)

        # PT prefill via the same scatter pathway. Use the canonical extract_feature
        # bypass — we have visual features already as `pt_proj_t`.
        with torch.inference_mode():
            o = self.model.language_model.model(
                input_ids=input_ids.to(self.model.device),
                visual_features=pt_proj_t.to(self.model.dtype).to(self.model.device),
                image_token_index=int(self.config.image_token_index),
                position_ids=torch.from_numpy(pos_np).to(self.model.device),
                attention_mask=attn_mask.to(self.model.device),
                use_cache=True, return_dict=True,
            )
            pt_logits = self.model.language_model.lm_head(o.last_hidden_state).float()  # (1, S, V)
        pt_last_logits = pt_logits[0, -1].detach().cpu()

        lines.append("")
        lines.append("[prefill logits at LAST input position (predicting first output token)]")
        lines.append(_stats("TRT last_logits", trt_last_logits))
        lines.append(_stats("PT  last_logits", pt_last_logits))
        lines.append(_compare("logits diff",   trt_last_logits, pt_last_logits))

        # Top-5 from each
        def _top5(t: torch.Tensor, label: str) -> str:
            probs = torch.softmax(t.float(), dim=-1)
            top_p, top_i = probs.topk(5)
            decoded = [self.tokenizer.decode([int(i)], skip_special_tokens=False) for i in top_i.tolist()]
            pairs = ", ".join(f"{tid}={prob:.3f}({tok!r})" for tid, prob, tok in zip(top_i.tolist(), top_p.tolist(), decoded))
            return f"{label}: {pairs}"
        lines.append(_top5(trt_last_logits, "TRT top5"))
        lines.append(_top5(pt_last_logits,  "PT  top5"))

        # ============================================================
        # MTP DECODE FORWARD (TRT decode_engine_mtp vs PT, per-position)
        # ============================================================
        # WHY: prefill cos_sim only tells us the last-real-token state. The
        # failure mode (garbage at MTP positions 1..5, position 0 correct) lives
        # in the MTP DECODE engine, NOT prefill. This block feeds the canonical
        # MTP input — [last_real_token, mask, mask, mask, mask, mask] — through
        # BOTH the TRT decode_engine_mtp AND PT model.language_model.model with
        # the SAME prefill KV, then compares logits at every one of the BLOCK
        # output positions.
        #
        # KV CACHE DTYPE FLOW (this is half of what we are verifying):
        #   _prefill_trt returns `_trt_past` as raw engine outputs. Each entry's
        #   numpy dtype is whatever the prefill engine's binding declared:
        #     * BF16 binding -> np.uint16 (byte storage of bf16)
        #     * FP16 binding -> np.float16
        #   _decode_trt passes those buffers through verbatim. _coerce_input in
        #   trt/engine.py enforces uint16 for BF16 bindings, so a dtype mismatch
        #   between prefill output and decode input WOULD raise TypeError; if it
        #   does NOT raise, the bytes line up.
        #   For the PT path we route the SAME `_trt_past` through _kv_np_to_torch,
        #   which detects uint16 (-> torch.bfloat16 zero-copy view -> REF_DTYPE)
        #   vs fp16 numpy (-> astype float32 -> REF_DTYPE). This is the exact
        #   conversion the orchestrator uses in _decode_pt today, so we are
        #   exercising the canonical bridge.
        #
        # INTERPRETATION GUIDE — read signature as (mtp[0], mtp[1..5], ar[0]):
        #
        #   | ar[0] cos | mtp[0] cos | mtp[1..5] cos | Conclusion |
        #   |---|---|---|---|
        #   | >=0.99 | >=0.99 | <0.5      | mtp_attention_pattern_wrong (highest yield) |
        #   | ~0.95  | ~0.95  | ~0.95     | trt_silent_fp16_downcast |
        #   | broken/NaN | broken/NaN | broken/NaN | sample_tokens_dtype_path |
        #   | <0.9   | <0.9   | <0.3      | KV bridge dtype/layout mismatch (kv_uint16_view_corruption) |
        #   | >=0.99 | >=0.99 | >=0.99    | bug is in MTP loop bookkeeping, not engines |
        #   | >=0.99 | <0.9   | >=0.99    | position-id decrement off-by-one on dup-last (new hypothesis) |
        #
        # A. mtp[0] high (>=0.99), mtp[1..5] LOW (<0.5), ar[0] high (>=0.99)
        #    Confirms: mtp_attention_pattern_wrong — SDLM block-mask patch in
        #    decode_engine_mtp is structurally wrong (update_causal_mask_for_one_gen_window_2d
        #    patch constant-folded with wrong block_size, or modeling_qwen2.py:1279
        #    AR arm baked under tracing when MTP arm was intended).
        #    Refutes: trt_silent_fp16_downcast (would degrade ar[0] equally),
        #    kv_uint16_view_corruption (would degrade mtp[0] too — same KV),
        #    visual_feature_scatter_position (prefill-side), sample_tokens_dtype_path.
        #    Action: dump engine layer precisions; re-export with explicit
        #    text_mask_token_id at trace position; verify LLMDecode.k_trace ==
        #    text_config.block_size + 1.
        #
        # B. mtp[0..5] uniformly degraded (~0.94-0.97 everywhere), ar[0] similarly degraded
        #    Confirms: trt_silent_fp16_downcast — uniform precision loss across all
        #    positions and both engines. Matches prefill 0.945 envelope (~0.2%/layer × 28).
        #    Refutes: mtp_attention_pattern_wrong (would NOT touch ar[0]),
        #    kv_uint16_view_corruption (would worsen with depth — mtp[5] > mtp[1]),
        #    sample_tokens_dtype_path (would corrupt to integer-code-shaped values).
        #    Action: polygraphy inspect model decode_engine_mtp.engine --show-layer-precisions;
        #    rebuild with STRONGLY_TYPED or OBEY_PRECISION_CONSTRAINTS.
        #
        # C. mtp[0..5] in [-1, 0] range or NaN/Inf in TRT mtp_logits, ar[0] also broken
        #    Confirms: sample_tokens_dtype_path — _logits_to_torch_float is
        #    bf16-viewing what is actually fp32 bytes.
        #    Discriminator: _stats min/max sane in PT (~±20) and garbage in TRT
        #    (~±1e-30 or ~±1e10). try/except will catch view shape error.
        #    Action: print decode_engine.is_bf16['logits'] and dtype['logits'];
        #    verify LLMDecode.forward .float() cast survived export.
        #
        # D. mtp[0] low (<0.9), mtp[1..5] very low (<0.3), ar[0] low (<0.9)
        #    Confirms: kv_uint16_view_corruption OR prefill KV dtype mismatch at
        #    prefill->decode boundary. Both decode engines query SAME _trt_past.
        #    Cross-check: print prefill_engine.is_bf16['present_k_0'] vs
        #    decode_engine.is_bf16['past_k_0'] vs decode_engine_ar.is_bf16['past_k_0'].
        #    Adversarial: if _decode_trt did NOT raise, _coerce_input accepted
        #    dtype, so bindings AGREE — likely layout/stride (K/V time dim,
        #    head dim mis-permuted). Add _stats on _trt_past[0] vs pt_past[0][0].
        #
        # E. mtp[0..5] high (>=0.99) everywhere, ar[0] high (>=0.99), but generation garbage
        #    Confirms: bug downstream — MTP token-sampling loop (orchestrator.py:329-352),
        #    KV truncation/append between MTP rounds (orchestrator.py:381), or
        #    sample_tokens itself.
        #    Action: add a SECOND MTP iteration — feed truncated/appended past
        #    from round 1, compare round-2 logits to PT canonical MTP loop.
        #
        # F. mtp[0] LOW (<0.9), mtp[1..5] high (>=0.99)
        #    Confirms: position-id off-by-one on duplicate-last slot. pos[:, -BLOCK:] -= 1
        #    (orchestrator.py:324) acts on ALL BLOCK positions including position 0;
        #    if engine traced expecting decrement only on positions 1..BLOCK-1, position 0
        #    queries wrong RoPE. Promotes a new hypothesis.
        #
        # G. mtp[1..5] PT top-1 are coord/box tokens (<203>, <box>) and TRT top-1
        #    are unrelated structural tokens (<im_end>, punctuation)
        #    Strengthens: mtp_attention_pattern_wrong — mask positions attending
        #    to WRONG KV slice; likely [-block_size:, -block_size-1] = -inf patch
        #    missing in engine (masks attend to duplicate-last = previous round's
        #    already-committed token). "Right magnitudes, wrong attention pattern".
        if self.decode_engine is None:
            lines.append("")
            lines.append("# MTP DECODE diagnostic SKIPPED (decode_engine not loaded)")
        else:
            BLOCK = int(self.config.text_config.block_size)
            MASK_ID = int(self.config.text_config.text_mask_token_id)
            P = int(ids_np.shape[1])  # prompt length == prefill KV time dim

            # Pick the "last real token". Use the prefill TRT top-1 (matches
            # what the generate loop would actually commit). Falls back to the
            # last prompt token if argmax is junk.
            last_tok = int(torch.argmax(trt_last_logits).item())

            # Canonical MTP input — mirrors orchestrator.py:320-325 exactly.
            mtp_ids = np.concatenate([
                np.array([[last_tok]], dtype=np.int64),
                np.full((1, BLOCK - 1), MASK_ID, dtype=np.int64),
            ], axis=1)                                        # (1, BLOCK)
            mtp_pos = np.arange(P, P + BLOCK, dtype=np.int64)[None, :]
            mtp_pos[:, -BLOCK:] -= 1                          # position trick
            mtp_att = np.ones((1, P + BLOCK), dtype=np.int64)

            # ---- TRT decode_engine_mtp ----
            # `_trt_past` is whatever dtype the prefill engine emitted. The
            # decode_engine_mtp's _coerce_input WILL raise if the dtypes do not
            # match its bindings, so reaching the output line means the
            # KV bridge between the two engines is at least byte-consistent.
            try:
                trt_mtp_logits_o, _trt_mtp_next = self._gen._decode_trt(
                    mtp_ids, mtp_pos, mtp_att, _trt_past, branch="mtp",
                )
                trt_mtp_logits = self._gen._logits_to_torch_float(trt_mtp_logits_o)  # (1, BLOCK, V)
            except Exception as e:
                lines.append("")
                lines.append(f"# MTP DECODE TRT FAILED: {type(e).__name__}: {e}")
                trt_mtp_logits = None

            # ---- PT model.language_model.model with prefill KV ----
            # Convert the SAME `_trt_past` (uint16/fp16 numpy) to torch via the
            # orchestrator's _kv_np_to_torch — this is the canonical bridge.
            with torch.inference_mode():
                pt_past = tuple(
                    (self._gen._kv_np_to_torch(_trt_past[2 * j]),
                     self._gen._kv_np_to_torch(_trt_past[2 * j + 1]))
                    for j in range(self.n_layers)
                )
                pt_mtp_out = self.model.language_model.model(
                    input_ids=torch.from_numpy(mtp_ids).to(self.model.device),
                    position_ids=torch.from_numpy(mtp_pos).to(self.model.device),
                    attention_mask=torch.from_numpy(mtp_att).to(self.model.device),
                    past_key_values=pt_past,
                    use_cache=True, return_dict=True,
                )
                pt_mtp_logits = self.model.language_model.lm_head(
                    pt_mtp_out.last_hidden_state
                ).float().cpu()                              # (1, BLOCK, V)

            lines.append("")
            lines.append(f"[MTP DECODE forward — input = [tok={last_tok}, mask×{BLOCK-1}]  "
                         f"P={P}  positions={mtp_pos.tolist()[0]}]")
            if trt_mtp_logits is not None:
                lines.append(_stats("TRT mtp_logits", trt_mtp_logits))
            lines.append(_stats("PT  mtp_logits", pt_mtp_logits))

            # Per-position comparison (this is the discriminator)
            if trt_mtp_logits is not None:
                for k in range(BLOCK):
                    a = trt_mtp_logits[0, k]
                    b = pt_mtp_logits[0, k]
                    role = "dup_last" if k == 0 else f"mask_{k}"
                    lines.append(_compare(f"mtp[{k}] {role}", a, b))
                    lines.append("  " + _top5(a, f"  TRT mtp[{k}] top5"))
                    lines.append("  " + _top5(b, f"  PT  mtp[{k}] top5"))

            # ============================================================
            # AR DECODE FORWARD (TRT decode_engine_ar vs PT, single position)
            # ============================================================
            # Same prefill KV, but input_ids = [last_tok] only (no mask). This
            # exercises the engine traced through the modeling_qwen2.py:1279
            # AR early-return branch. If decode_ar disagrees with PT we know
            # the AR engine is broken too; if it agrees we have isolated the
            # bug to the SDLM block-mask path in decode_engine_mtp.
            if self.decode_engine_ar is not None:
                ar_ids = np.array([[last_tok]], dtype=np.int64)
                ar_pos = np.array([[P]], dtype=np.int64)
                ar_att = np.ones((1, P + 1), dtype=np.int64)
                try:
                    trt_ar_logits_o, _ = self._gen._decode_trt(
                        ar_ids, ar_pos, ar_att, _trt_past, branch="ar",
                    )
                    trt_ar_logits = self._gen._logits_to_torch_float(trt_ar_logits_o)
                except Exception as e:
                    lines.append("")
                    lines.append(f"# AR DECODE TRT FAILED: {type(e).__name__}: {e}")
                    trt_ar_logits = None

                with torch.inference_mode():
                    pt_past2 = tuple(
                        (self._gen._kv_np_to_torch(_trt_past[2 * j]),
                         self._gen._kv_np_to_torch(_trt_past[2 * j + 1]))
                        for j in range(self.n_layers)
                    )
                    pt_ar_out = self.model.language_model.model(
                        input_ids=torch.from_numpy(ar_ids).to(self.model.device),
                        position_ids=torch.from_numpy(ar_pos).to(self.model.device),
                        attention_mask=torch.from_numpy(ar_att).to(self.model.device),
                        past_key_values=pt_past2,
                        use_cache=True, return_dict=True,
                    )
                    pt_ar_logits = self.model.language_model.lm_head(
                        pt_ar_out.last_hidden_state
                    ).float().cpu()

                lines.append("")
                lines.append(f"[AR DECODE forward — input = [tok={last_tok}]  P={P}]")
                if trt_ar_logits is not None:
                    lines.append(_stats("TRT ar_logits", trt_ar_logits))
                    lines.append(_compare("ar[0]", trt_ar_logits[0, 0], pt_ar_logits[0, 0]))
                    lines.append("  " + _top5(trt_ar_logits[0, 0], "  TRT ar[0] top5"))
                lines.append(_stats("PT  ar_logits", pt_ar_logits))
                lines.append("  " + _top5(pt_ar_logits[0, 0], "  PT  ar[0] top5"))
            else:
                lines.append("")
                lines.append("# AR DECODE diagnostic SKIPPED (decode_engine_ar not loaded)")

        return "\n".join(lines) + "\n"

    def diagnose(self, image, prompt: str = "Locate all the instances that matches the following description: cats.",
                 max_new_tokens: int = 128, generation_mode: str = "hybrid",
                 ) -> dict:
        """Explicit three-tier diagnostic: run TRT, PT-patched, and PT-canonical-unpatched
        and dump all outputs side-by-side. Returns a dict with each tier's boxes/text.

        This is INTENT-EXPLICIT — call it when you want to compare runtimes, not
        as a silent fallback when detect() returns empty.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        orig_w, orig_h = image.size
        prompt, _ = canonicalize_prompt(prompt)

        results = {}
        # TRT
        if self.prefill_engine and self.decode_engine and self.vit_engine:
            boxes, text = self._detect_via_trt(image, prompt, max_new_tokens, generation_mode)
            results["trt"] = {"boxes": boxes, "text": text}
        # PT patched
        if self.model is not None:
            boxes, text = self._detect_via_pt(image, prompt, max_new_tokens, generation_mode)
            results["pt_patched"] = {"boxes": boxes, "text": text}
            # PT canonical unpatched
            if self.patches_snapshot is not None:
                boxes, text = self._detect_via_pt(image, prompt, max_new_tokens, generation_mode, unpatched=True)
                results["pt_canonical"] = {"boxes": boxes, "text": text}

        if not results:
            raise RuntimeError("diagnose(): no inference paths available (no TRT engines, no PT model)")

        # Compose a diagnostic dump
        segments = []
        for name, r in results.items():
            segments.append(f"### {name} (boxes={len(r['boxes'])}) ###\n{r['text']}")
        combined_text = "\n\n".join(segments) + "\n"
        # Use the path that produced the most boxes as the path_used label
        best = max(results.items(), key=lambda kv: len(kv[1]["boxes"]))
        self._write_diagnostic(image, prompt, f"diagnose({best[0]})", best[1]["boxes"], combined_text,
                               max_new_tokens, generation_mode, orig_w, orig_h)
        return results

    def _write_diagnostic(self, image, prompt: str, path_used: str, boxes, text: str,
                          max_new_tokens: int, generation_mode: str,
                          orig_w: int, orig_h: int,
                          comparison: Optional[tuple] = None,
                          layer_diag: str = "") -> None:
        """Write WORK/last_inference.txt with the rendered prompt, token ids, grids,
        and raw output. Exceptions propagate — chat-template failure is a real bug.

        comparison: optional (other_path_name, other_boxes, other_text, error_str_or_None)
                    captured by detect() when primary path failed.
        layer_diag: optional string with per-layer TRT-vs-PT numerical comparison
                    (from _layer_diagnostic). Empty when not applicable.
        """
        msg = [{"role":"user","content":[{"type":"image","image":image},
                                            {"type":"text","text":prompt}]}]
        rendered_text = self.processor.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs_d, vids_d = self.processor.process_vision_info(msg)
        enc_d = self.processor(text=[rendered_text], images=imgs_d, videos=vids_d, return_tensors="pt")
        token_ids = enc_d["input_ids"][0].tolist()
        n_img_tokens_actual = int((enc_d["input_ids"] == int(self.config.image_token_index)).sum())
        gh_d = enc_d.get("image_grid_hws")
        diag_grid_h, diag_grid_w = 0, 0
        if gh_d is not None:
            gh_arr = gh_d.tolist() if hasattr(gh_d, "tolist") else gh_d
            diag_grid_h, diag_grid_w = int(gh_arr[0][0]), int(gh_arr[0][1])

        # Environment context — important for cross-machine debugging
        import transformers, sys
        try:
            import torch as _torch
            torch_v = _torch.__version__
            cuda_avail = _torch.cuda.is_available()
            cuda_dev = _torch.cuda.get_device_name(0) if cuda_avail else "n/a"
            cuda_v = _torch.version.cuda if cuda_avail else "n/a"
        except Exception:
            torch_v = cuda_dev = cuda_v = "?"
            cuda_avail = False
        try:
            _import = __import__
            _trt_v = _import("tensorrt").__version__
        except Exception:
            _trt_v = "n/a"
        # PT weight sanity (tied? real values? — re-stat at write time)
        try:
            e = self.model.language_model.model.embed_tokens.weight
            h = self.model.language_model.lm_head.weight
            embed_std = e.std().item(); head_std = h.std().item()
            embed_mean = e.mean().item(); head_mean = h.mean().item()
            tied = (e.data_ptr() == h.data_ptr())
        except Exception:
            embed_std = head_std = embed_mean = head_mean = float("nan")
            tied = False
        # Engine fingerprint stamp
        try:
            fp_path = TRT_DIR / ".model_fingerprint"
            stamped_fp = fp_path.read_text().strip() if fp_path.exists() else "<no stamp>"
        except Exception:
            stamped_fp = "<read failed>"
        try:
            current_fp = self._model_fingerprint()
        except Exception:
            current_fp = "<compute failed>"

        # Package version + commit (so we can verify the user is running the
        # latest pushed code on Colab, not a stale pip cache).
        try:
            import lrai_locate_anything as _lra
            pkg_version = getattr(_lra, "__version__", "?")
        except Exception:
            pkg_version = "?"
        try:
            # If installed from a git URL, pip records the commit in METADATA;
            # this is the most reliable way to tell what's actually loaded.
            import subprocess
            _pi = subprocess.run(["pip", "show", "lrai_locate_anything"],
                                   capture_output=True, text=True, timeout=5).stdout
            pkg_location = next((L.split(":", 1)[1].strip()
                                 for L in _pi.split("\n")
                                 if L.startswith("Location")), "?")
        except Exception:
            pkg_location = "?"

        # Engine load status — without this we can't tell whether the
        # dual-engine fix (b8fe1dd) is actually active. If decode_ar is None
        # while decode is loaded, the runtime is silently running the old
        # single-engine path with the MTP branch baked everywhere.
        eng_status = {
            "vit_engine": "loaded" if self.vit_engine else "MISSING",
            "proj_engine": "loaded" if self.proj_engine else "MISSING",
            "prefill_engine": "loaded" if self.prefill_engine else "MISSING",
            "decode_engine_mtp": "loaded" if self.decode_engine else "MISSING",
            "decode_engine_ar": "loaded" if self.decode_engine_ar else "MISSING",
        }
        # Engine files on disk (separate from "loaded in memory")
        try:
            engine_files = sorted([p.name for p in TRT_DIR.iterdir() if p.suffix == ".engine"])
        except Exception:
            engine_files = []

        sections = [
            f"# prompt:           {prompt}",
            f"# path:             {path_used}",
            f"# boxes:            {len(boxes)}",
            f"# generation_mode:  {generation_mode}",
            f"# orig_img:         {orig_w}x{orig_h}",
            f"# eng_img (TRT):    {self.eng_img_w}x{self.eng_img_h}",
            f"# diag_grid_hws:    ({diag_grid_h},{diag_grid_w})  [from processor on NATIVE image]",
            f"# eng_grid_hws:     ({self.grid_h},{self.grid_w})  [baked into TRT engine]",
            f"# image_token_idx:  {int(self.config.image_token_index)}",
            f"# image_tokens_in_input_ids: {n_img_tokens_actual}",
            f"# input_ids_len:    {len(token_ids)}",
            f"# input_ids_first_20:  {token_ids[:20]}",
            f"# input_ids_last_20:   {token_ids[-20:]}",
            f"",
            f"# === environment ===",
            f"# python:           {sys.version.split()[0]}",
            f"# torch:            {torch_v}  cuda_avail={cuda_avail}  cuda_v={cuda_v}  device={cuda_dev}",
            f"# transformers:     {transformers.__version__}",
            f"# tensorrt:         {_trt_v}",
            f"# REF_DTYPE:        {REF_DTYPE}",
            f"# model.dtype:      {next(self.model.parameters()).dtype if self.model is not None else 'n/a'}",
            f"# lrai_locate_anything: version={pkg_version}  location={pkg_location}",
            f"",
            f"# === TRT engine load status ===",
            *[f"# {k}: {v}" for k, v in eng_status.items()],
            f"# engine files on disk in {TRT_DIR}: {engine_files}",
            f"# decode_ar required for the dual-engine fix (b8fe1dd); MISSING means TRT is running with the wrong branch baked for AR/rebuild steps",
            f"",
            f"# === pt weights sanity (in-memory, AT DUMP TIME) ===",
            f"# embed: mean={embed_mean:+.5f} std={embed_std:.5f}   (trained ~0.024, random init ~0.020)",
            f"# head:  mean={head_mean:+.5f} std={head_std:.5f}",
            f"# tied (data_ptr equality): {tied}",
            f"",
            f"# === engine fingerprint stamp ===",
            f"# stamped on disk:  {stamped_fp}",
            f"# current model:    {current_fp}",
            f"# match? {stamped_fp == current_fp}",
        ]

        # Comparison output (the OTHER path) if detect() ran one on a 0-box failure
        if comparison is not None:
            other_path, other_boxes, other_text, other_err = comparison
            sections.append("")
            sections.append(f"# === comparison: ran {other_path} path on the same input ===")
            if other_err is not None:
                sections.append(f"# {other_path} raised: {other_err}")
            else:
                sections.append(f"# {other_path} boxes: {len(other_boxes)}")
                sections.append(f"# {other_path} text ({len(other_text)} chars; first 600):")
                # Embed the comparison text below for the user to inspect
                sections.append("")
                sections.append(f"=== {other_path} model output ===")
                sections.append(other_text)

        sections.append("")
        sections.append(layer_diag if layer_diag else "")
        sections.append(f"=== rendered prompt (chat template applied) ===")
        sections.append(rendered_text)
        sections.append("")
        sections.append(f"=== {path_used} model output ===")
        sections.append(text)
        sections.append("")

        (WORK / "last_inference.txt").write_text("\n".join(sections))

