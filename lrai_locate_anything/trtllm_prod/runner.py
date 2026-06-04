"""End-to-end single-image inference for LocateAnything-3B on TRT-LLM.

Replaces the orchestrator's three-engine prefill+decode loop with a
single ModelRunner.generate call that takes (text_tokens, vision_prompt_table)
and returns the full output sequence.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch
from PIL import Image

from .moonvit_adapter import MoonViTAdapter
from lrai_locate_anything.parse import parse_boxes
from lrai_locate_anything.orchestrator import canonicalize_prompt

if TYPE_CHECKING:
    pass


class LocateAnythingTRTLLMRunner:
    """Single-image LocateAnything detection on TRT-LLM.

    Inference flow:
        1. tokenizer(prompt)                  -> input_ids
        2. MoonViTAdapter.forward(image)      -> prompt_table (1, L_post, hidden)
        3. tensorrt_llm.runtime.ModelRunner.generate(
               batch_input_ids=[input_ids],
               prompt_table=prompt_table,
               prompt_tasks=[0],
               max_new_tokens=...,
               ...)                            -> output_ids
        4. tokenizer.decode(output_ids)        -> raw text
        5. parse.parse_boxes_with_labels(text) -> List[(bbox, label)]
    """

    def __init__(
        self,
        llm_engine_path: "Path | str",
        hf_dir: "Path | str",
        *,
        vision_mode: str = "pt",
        weights_dir: "Path | str | None" = None,
        vision_proj_engine_path: "Path | str | None" = None,
        grid_h: int = 36,
        grid_w: int = 46,
    ):
        """Wire up the engines + tokenizer.

        Args:
            llm_engine_path:         Path to the .engine file built by
                build.build_llm_engine, OR the directory containing it
                alongside config.json. ModelRunner.from_dir needs the dir.
            hf_dir:                  HF checkpoint dir (for tokenizer +
                                     generation_config + LM config metadata
                                     such as image_token_index, eos_token_id).
            vision_mode:             'pt' (default) uses the Phase D PT
                                     MoonViTVisionModel; 'trt' uses the fused
                                     vision_proj.engine path.
            weights_dir:             Required (or defaulted) for vision_mode='pt'.
                                     Defaults to hf_dir when None.
            vision_proj_engine_path: Required for vision_mode='trt'. The
                                     export_prod vision_proj.engine path.
            grid_h, grid_w:          Patch-grid dims for PT mode (passed through
                                     to MoonViTAdapter). Defaults: 36 x 46.

        Fails loud (no try/except) on:
            - missing engine files
            - missing tokenizer assets in hf_dir
            - TRT-LLM import failure (cu13 LD_LIBRARY_PATH not set, etc.)
            - missing weights_dir / vision_proj_engine_path for the chosen mode
        """
        if vision_mode not in ("pt", "trt"):
            raise RuntimeError(
                f"vision_mode must be 'pt' or 'trt', got {vision_mode!r}"
            )

        # Default weights_dir to hf_dir for PT mode -- the LocateAnything HF
        # checkpoint dir IS the safetensors source.
        if vision_mode == "pt" and weights_dir is None:
            weights_dir = hf_dir
        if vision_mode == "pt" and weights_dir is None:
            raise RuntimeError("vision_mode='pt' requires weights_dir")
        if vision_mode == "trt" and vision_proj_engine_path is None:
            raise RuntimeError(
                "vision_mode='trt' requires vision_proj_engine_path"
            )

        llm_engine_path = Path(llm_engine_path)
        hf_dir = Path(hf_dir)
        if vision_proj_engine_path is not None:
            vision_proj_engine_path = Path(vision_proj_engine_path)
        if weights_dir is not None:
            weights_dir = Path(weights_dir)

        self.vision_mode = vision_mode

        # 1. Resolve engine_dir (TRT-LLM ModelRunner consumes a directory that
        # holds rank0.engine + config.json, not a single file path).
        if llm_engine_path.is_dir():
            engine_dir = llm_engine_path
        else:
            if not llm_engine_path.exists():
                raise FileNotFoundError(
                    f"LLM TRT engine file not found at {llm_engine_path}"
                )
            engine_dir = llm_engine_path.parent
        engine_config = engine_dir / "config.json"
        if not engine_config.exists():
            raise FileNotFoundError(
                f"TRT-LLM engine config.json not found at {engine_config}. "
                f"Expected alongside the .engine shard."
            )
        self.engine_dir = engine_dir

        # 2. Load the engine-side config to recover prompt_vocab_size and dtype.
        # ModelRunner ignores these but we need them to construct the virtual
        # prompt-token ids that map to prompt_table rows.
        with open(engine_config, "r") as f:
            engine_cfg = json.load(f)
        build_cfg = engine_cfg.get("build_config", {})
        self.max_prompt_embedding_table_size = int(
            build_cfg.get("max_prompt_embedding_table_size", 0)
        )
        if self.max_prompt_embedding_table_size == 0:
            # Check whether a sibling .new engine exists that was rebuilt with
            # the correct flag (rebuild_llm_bf16.sh writes to <dir>.new first).
            new_engine_dir = engine_dir.parent / (engine_dir.name + ".new")
            if new_engine_dir.is_dir():
                new_cfg_path = new_engine_dir / "config.json"
                if new_cfg_path.exists():
                    import json as _json
                    _nc = _json.loads(new_cfg_path.read_text())
                    _new_tbl = int(
                        _nc.get("build_config", {})
                        .get("max_prompt_embedding_table_size", 0)
                    )
                    if _new_tbl > 0:
                        raise RuntimeError(
                            f"Engine at {engine_dir} was built without "
                            f"max_prompt_embedding_table_size (got 0). "
                            f"A correctly-rebuilt engine is available at "
                            f"{new_engine_dir} (max_prompt_embedding_table_size="
                            f"{_new_tbl}). Promote it with:\n"
                            f"  mv {engine_dir} {engine_dir}.old && "
                            f"mv {new_engine_dir} {engine_dir}\n"
                            f"then re-run this command."
                        )
            raise RuntimeError(
                f"Engine at {engine_dir} was built without "
                f"max_prompt_embedding_table_size. Vision-prompt injection "
                f"requires this to be > 0. Rebuild with:\n"
                f"  trtllm-build --checkpoint_dir <ckpt> --output_dir {engine_dir} "
                f"--max_prompt_embedding_table_size <L_post> ...\n"
                f"where L_post = (grid_h // 2) * (grid_w // 2) = 414 for the "
                f"default 36x46 grid (see build.build_llm_engine)."
            )
        # vocab_size = real text vocab; virtual prompt slots start at vocab_size.
        pretrained_cfg = engine_cfg.get("pretrained_config", {})
        self.text_vocab_size = int(pretrained_cfg.get("vocab_size", 0))
        if self.text_vocab_size == 0:
            raise RuntimeError(
                f"Engine config at {engine_config} is missing pretrained_config.vocab_size."
            )
        # Engine dtype -- must agree with MoonViTAdapter's llm_dtype.
        engine_dtype_str = pretrained_cfg.get("dtype", "bfloat16")
        self.llm_dtype = _torch_dtype_from_str(engine_dtype_str)

        # 3. Load the LLM TRT runtime. This is the no-fallback fail-loud path:
        # if cu13 / cudnn / nccl LD_LIBRARY_PATH isn't set we let ImportError
        # propagate. See trtllm_prod/convert.py docstring.
        from tensorrt_llm.runtime import ModelRunner
        self.llm_runner = ModelRunner.from_dir(
            engine_dir=str(engine_dir),
            rank=0,
            debug_mode=False,
        )

        # 4. Load tokenizer + HF config (for image_token_index, eos_token_id,
        # special-token ids consumed by parse_boxes_with_labels).
        from transformers import AutoTokenizer, AutoProcessor
        if not (hf_dir / "tokenizer_config.json").exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {hf_dir}. "
                f"Pass the HF checkpoint dir, not the TRT-LLM checkpoint dir."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(hf_dir), trust_remote_code=True
        )
        # PT-mode vision adapter needs the HF AutoProcessor (handles
        # letterbox + normalization + grid_hws emission internally). TRT-mode
        # does letterbox inside MoonViTAdapter._letterbox; processor is unused
        # there but cheap to construct, so we always load it.
        self.processor = AutoProcessor.from_pretrained(
            str(hf_dir), trust_remote_code=True
        )
        hf_cfg_path = hf_dir / "config.json"
        if not hf_cfg_path.exists():
            raise FileNotFoundError(f"HF config.json not found at {hf_cfg_path}")
        with open(hf_cfg_path, "r") as f:
            hf_cfg = json.load(f)
        # LocateAnything-3B carries image_token_index on the top-level config
        # (vendor config_locateanything.py). FAIL LOUD if absent -- without it
        # we cannot find the slots in input_ids to swap for prompt-table rows.
        if "image_token_index" not in hf_cfg:
            raise RuntimeError(
                f"HF config at {hf_cfg_path} has no 'image_token_index'. "
                f"This is the placeholder token id that input_ids carries at "
                f"the image positions; the runner needs it to splice in the "
                f"virtual prompt-table ids."
            )
        self.image_token_id = int(hf_cfg["image_token_index"])
        # eos / pad: prefer the engine_cfg's end_id if present (build pipeline
        # may override Qwen2's default), else fall back to tokenizer.eos_token_id.
        self.eos_token_id = int(
            pretrained_cfg.get("eos_token_id", self.tokenizer.eos_token_id)
        )
        self.pad_token_id = int(
            pretrained_cfg.get("pad_token_id",
                               self.tokenizer.pad_token_id
                               if self.tokenizer.pad_token_id is not None
                               else self.eos_token_id)
        )

        # 5. Vision adapter. MoonViTAdapter handles both PT (Phase D
        # MoonViTVisionModel) and TRT (fused vision_proj.engine) modes; we
        # thread the runner-level vision_mode through and keep the engine path
        # preserved for TRT-mode callers. llm_dtype is passed as a *string*
        # because the adapter's PT path indexes a string->torch.dtype map.
        # Use the engine's pretrained_config dtype string (read in step 2).
        self.vision = MoonViTAdapter(
            vision_mode=vision_mode,
            weights_dir=str(weights_dir) if weights_dir is not None else None,
            grid_h=grid_h,
            grid_w=grid_w,
            vision_proj_engine_path=(
                str(vision_proj_engine_path)
                if vision_proj_engine_path is not None
                else None
            ),
            llm_dtype=engine_dtype_str,
        )
        # Back-compat alias matching the spec's attribute name.
        self.vision_adapter = self.vision

        # 6. Cache the chat-template assets so we can build prompts that match
        # the canonical format. The processor's chat template wraps the user
        # message in Qwen2's <|im_start|>user / <|im_end|> bracketing and
        # inserts the image-placeholder run at the right slot. We replay that
        # here by hand because TRT-LLM has no processor; we operate on
        # tokenizer + image_token_id directly.
        # Token id sequence for the <|im_start|>user header / image / text / footer
        # is the standard Qwen2.5-VL convention:
        #   <|im_start|>user\n<|vision_start|><image><image>...<|vision_end|>{TEXT}<|im_end|>\n
        #   <|im_start|>assistant\n
        # We cache the static segments as token ids so per-frame work is just
        # the dynamic-text encode + the image-id-run injection.
        self._vision_start_id = _safe_tok_to_id(self.tokenizer, "<|vision_start|>")
        self._vision_end_id = _safe_tok_to_id(self.tokenizer, "<|vision_end|>")
        # The number of image tokens to splice in is L_post = (grid_h * grid_w) / 4
        # (kh*kw=2*2 merger downsample). Pull from the vision adapter so the
        # two stay in lock-step.
        kh = kw = 2
        self.L_post = (self.vision.grid_h // kh) * (self.vision.grid_w // kw)

    def detect(
        self,
        image: "Image.Image",
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[List[Tuple[tuple, str]], str]:
        """Run a single grounding query against the image.

        Args:
            image:          RGB PIL image (or path-like that PIL can open).
            prompt:         User instruction (e.g. "locate the red car").
                            Auto-canonicalized via canonicalize_prompt() --
                            the model only accepts a specific phrasing; we
                            FAIL LOUD if the rewriter cannot canonicalize.
            max_new_tokens: Decode-side cap. Must be <= the engine's max_output_len.
            temperature:    0.0 = greedy. Anything >0 enables sampling.
            top_p:          Nucleus sampling cap. Ignored when temperature == 0.

        Returns:
            (detections, raw_text) where:
              detections is a list of ((x1, y1, x2, y2), label) tuples in
              the original image's pixel coordinates, and
              raw_text is the un-parsed decoded string (for debug / logging).
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        if image.mode != "RGB":
            image = image.convert("RGB")
        orig_w, orig_h = image.size

        # 1. Canonicalize prompt (FAIL LOUD if we can't).
        canon_prompt, _rewritten = canonicalize_prompt(prompt)
        if "matches the following description:" not in canon_prompt.lower():
            raise RuntimeError(
                f"Prompt is not in canonical form and could not be auto-rewritten: "
                f"{prompt!r}. The model only accepts: "
                f"'Locate all the instances that matches the following description: <X>.'"
            )

        # 2. Vision: PIL -> prompt_table. Adapter return depends on vision_mode:
        #   - PT mode: HF AutoProcessor handles preprocessing internally; the
        #     adapter returns a bare (1, L_post, hidden) tensor. LocateAnything
        #     emits bbox coords in the original PIL.size coordinate space, so
        #     un-letterboxing is the identity.
        #   - TRT mode: adapter returns (prompt_table, scale, pad_x, pad_y)
        #     because it letterboxes internally; box coords come back in the
        #     letterboxed engine resolution and must be un-warped.
        if self.vision.vision_mode == "pt":
            prompt_table = self.vision.forward(image, processor=self.processor)
            # PIL.size space: parse with the original image dims, no un-letterbox.
            parse_w, parse_h = image.size
            scale, pad_x, pad_y = 1.0, 0, 0
        else:
            prompt_table, scale, pad_x, pad_y = self.vision.forward(image)
            parse_w, parse_h = self.vision.img_w, self.vision.img_h
        L_post_actual = int(prompt_table.shape[1])
        if L_post_actual != self.L_post:
            raise RuntimeError(
                f"vision adapter emitted L_post={L_post_actual} but runner expected "
                f"{self.L_post} (grid={self.vision.grid_h}x{self.vision.grid_w}). "
                f"Re-build vision_proj engine for the configured grid."
            )
        if L_post_actual > self.max_prompt_embedding_table_size:
            raise RuntimeError(
                f"L_post={L_post_actual} exceeds engine's "
                f"max_prompt_embedding_table_size={self.max_prompt_embedding_table_size}. "
                f"Rebuild the LLM engine with a larger virtual-vocab cap."
            )

        # 3. Tokenize prompt + splice virtual image-token ids.
        input_ids = self._build_input_ids(canon_prompt, L_post_actual)  # (S,) long tensor on CUDA

        # 4. TRT-LLM generate. The prompt_tuning_config (prompt_table + tasks)
        # plumbs the per-batch virtual-token embeddings to the kernel that
        # scatters them into the embed_tokens output at the slots whose ids
        # are >= vocab_size. prompt_tasks = "0" (string of comma-separated
        # per-batch indices) selects row 0 of prompt_table for batch item 0.
        do_sample = temperature > 0.0
        outputs = self.llm_runner.generate(
            batch_input_ids=[input_ids],
            max_new_tokens=int(max_new_tokens),
            end_id=self.eos_token_id,
            pad_id=self.pad_token_id,
            temperature=float(temperature) if do_sample else 1.0,
            top_p=float(top_p) if do_sample else 0.0,
            top_k=0 if do_sample else 1,
            prompt_table=prompt_table,
            prompt_tasks="0",
            prompt_vocab_size=L_post_actual,
            output_sequence_lengths=True,
            return_dict=True,
        )

        # 5. Decode the new tokens only (strip the prompt prefix).
        # outputs["output_ids"] shape: (batch, beam, seq). We took batch=1, beam=1.
        output_ids = outputs["output_ids"]
        seq_lens = outputs.get("sequence_lengths", None)
        prompt_len = int(input_ids.shape[0])
        if seq_lens is not None:
            total_len = int(seq_lens[0][0])
            new_ids = output_ids[0][0][prompt_len:total_len]
        else:
            new_ids = output_ids[0][0][prompt_len:]
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

        # 6. Parse boxes and (TRT mode only) un-letterbox into ORIGINAL image
        # pixel coords. parse_boxes returns coordinates scaled to the (W, H)
        # we pass; for PT mode that is PIL.size (already final); for TRT mode
        # we pass the letterboxed engine resolution and undo the transform.
        boxes_lb = parse_boxes(raw_text, parse_w, parse_h)
        detections: List[Tuple[tuple, str]] = []
        need_unwarp = self.vision.vision_mode == "trt" and (
            scale != 1.0 or pad_x != 0 or pad_y != 0
        )
        for ((x1, y1, x2, y2), label) in boxes_lb:
            if need_unwarp:
                ox1 = (x1 - pad_x) / scale
                oy1 = (y1 - pad_y) / scale
                ox2 = (x2 - pad_x) / scale
                oy2 = (y2 - pad_y) / scale
            else:
                ox1, oy1, ox2, oy2 = x1, y1, x2, y2
            # Clip to the original image bounds (letterbox padding can produce
            # tiny negative or out-of-range coords for boxes that hugged the
            # padded edge; PT mode shouldn't but cheap to guard).
            ox1 = max(0.0, min(float(orig_w), ox1))
            oy1 = max(0.0, min(float(orig_h), oy1))
            ox2 = max(0.0, min(float(orig_w), ox2))
            oy2 = max(0.0, min(float(orig_h), oy2))
            detections.append(((ox1, oy1, ox2, oy2), label))
        return detections, raw_text

    # ------------------------------------------------------------------
    # Input-id construction
    # ------------------------------------------------------------------
    def _build_input_ids(self, canonical_prompt: str, n_img_tokens: int) -> "torch.Tensor":
        """Construct the input_ids tensor for the canonical chat-formatted prompt.

        Layout (Qwen2.5 chat template + image block):
            <|im_start|>user\n
            <|vision_start|>
            <VIRTUAL_IMG_ID_0> ... <VIRTUAL_IMG_ID_{n_img_tokens-1}>
            <|vision_end|>
            {canonical_prompt}
            <|im_end|>\n
            <|im_start|>assistant\n

        The virtual image-token ids start at self.text_vocab_size (TRT-LLM's
        prompt-table convention: ids >= vocab_size index into prompt_table).
        prompt_tasks="0" picks row 0 of prompt_table for batch item 0, so each
        virtual id (text_vocab_size + i) maps to prompt_table[0, i, :].

        Returns:
            torch.Tensor[long] of shape (S,), on CUDA.
        """
        # Encode the static header (everything up to and including the
        # vision_start sentinel).
        # We rely on the tokenizer's chat_template if available; otherwise
        # build the Qwen2 default by hand.
        if self.tokenizer.chat_template is not None:
            # Build the chat text WITHOUT an image placeholder, then surgically
            # insert the vision_start + virtual-img-ids + vision_end run at the
            # canonical position (right after the user role header). This avoids
            # relying on processor.process_vision_info on the TRT path.
            header_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": canonical_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Encode the full string. We will splice the image-id run in by
            # locating the user-text start (the first occurrence of the
            # canonical prompt body in the tokenized stream). Simpler & robust:
            # split the rendered text at the user prompt body, encode the two
            # halves separately, and concatenate with the image block in between.
            split_anchor = canonical_prompt
            if split_anchor not in header_text:
                raise RuntimeError(
                    f"chat_template rendered text does not contain the canonical "
                    f"prompt body verbatim; cannot splice image block. "
                    f"Rendered: {header_text!r}"
                )
            prefix_str, suffix_str = header_text.split(split_anchor, 1)
            prefix_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            body_ids = self.tokenizer.encode(split_anchor, add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False)
        else:
            # No chat template: fall back to the Qwen2 canonical bracketing,
            # built by hand from the special-token strings.
            prefix_str = "<|im_start|>user\n"
            suffix_str = "<|im_end|>\n<|im_start|>assistant\n"
            prefix_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            body_ids = self.tokenizer.encode(canonical_prompt, add_special_tokens=False)
            suffix_ids = self.tokenizer.encode(suffix_str, add_special_tokens=False)

        # Image block: <|vision_start|> + (text_vocab_size + i for i in range(n_img_tokens)) + <|vision_end|>
        img_block: List[int] = []
        if self._vision_start_id is not None:
            img_block.append(self._vision_start_id)
        for i in range(n_img_tokens):
            img_block.append(self.text_vocab_size + i)
        if self._vision_end_id is not None:
            img_block.append(self._vision_end_id)

        # Final sequence: prefix + img_block + body + suffix.
        all_ids = list(prefix_ids) + img_block + list(body_ids) + list(suffix_ids)
        # FAIL LOUD if no virtual-image ids made it in -- means we wired up wrong.
        n_virtual = sum(1 for tid in all_ids if tid >= self.text_vocab_size)
        if n_virtual != n_img_tokens:
            raise RuntimeError(
                f"input_ids built with {n_virtual} virtual image ids but "
                f"expected {n_img_tokens}. Image-block splicing is broken."
            )
        return torch.tensor(all_ids, dtype=torch.int32, device="cuda")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _torch_dtype_from_str(s: str) -> "torch.dtype":
    """Map TRT-LLM dtype-string -> torch.dtype. FAIL LOUD on unknown."""
    m = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if s not in m:
        raise RuntimeError(f"unknown engine dtype string: {s!r}")
    return m[s]


def _safe_tok_to_id(tokenizer, token: str):
    """Look up a special-token id; return None if the tokenizer doesn't know it.

    Some LocateAnything-3B tokenizer variants ship the Qwen2.5 special-token
    set; older variants do not. Caller decides whether the absence is fatal --
    for vision_start/vision_end we accept absence and rely on the surrounding
    chat-template text to carry the structure.
    """
    tid = tokenizer.convert_tokens_to_ids(token)
    unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
    if tid is None or tid == unk:
        return None
    return int(tid)
