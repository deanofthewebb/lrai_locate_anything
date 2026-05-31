"""Optional INT4 AWQ quantisation via NVIDIA Model Optimizer (modelopt).

CAVEAT: modelopt's INT4_AWQ_CFG + torch.onnx.export produces ONNX that stores INT4
weights as packed-INT8 with block-wise DequantizeLinear scales. TensorRT 10.7's parser
rejects that layout — its `DequantizeLinear` with `kBLOCKED` strictly requires the
input tensor's ONNX dtype to be `INT4`. The resulting engine will fail to build with
"Block quantization is supported only for DataType::kINT4".

For TRT-compatible INT4, use one of:
  - TRT-LLM: `trtllm-build --use_weight_only` after `quantize_static`
  - modelopt.onnx.quantization.quantize_static on the FP16 ONNX directly
  - Skip INT4 and use the FP16 LLM TRT engine (sufficient for most use cases)

This module is kept for completeness and produces ONNX that ORT can run; it just
doesn't currently build under plain TRT 10.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, List

import torch


def _default_calib_texts() -> List[str]:
    return [
        "Detect all cats. Return bounding boxes.",
        "Find every person in the image and bound each one.",
        "Where are the red cars? Provide bounding boxes for each.",
        "List bounding boxes for every animal visible.",
        "Locate the chair nearest the window.",
        "Identify the traffic signs and bound them.",
        "Find any text in the image; bound each text region.",
        "Point to the largest object in the scene.",
        "Detect every pedestrian, cyclist, and vehicle.",
        "Bound every face you can see.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
        "It was the best of times, it was the worst of times, it was the age of wisdom.",
        "<box><x1><y1><x2><y2></box> represents a bounding-box coordinate quadruple.",
        "<ref>object name</ref> tokens precede each <box>...</box> block in the output.",
    ]


def quantize_int4_awq(
    model,
    tokenizer,
    calib_texts: List[str] | None = None,
    verbose: bool = True,
) -> None:
    """Run modelopt INT4 AWQ on model.language_model in-place.

    Text-only calibration: AWQ-lite calibrates LM Linear inputs (post-LayerNorm features)
    whose distribution is dominated by the LN scale, not by whether the upstream
    embedding came from a text or scattered visual token. Text-only calibration with a
    varied corpus is the canonical AWQ-for-LM approach.

    Uses torch.no_grad() (not inference_mode) so modelopt's forward hooks can collect amax.
    """
    import modelopt.torch.quantization as mtq

    texts = calib_texts or _default_calib_texts()
    if verbose:
        print(f"[int4] calibrating with {len(texts)} prompts ...")

    def calibration_loop(m=None):
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                model.language_model.model(
                    input_ids=ids,
                    position_ids=torch.arange(ids.shape[1], device=ids.device)[None, :],
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    return_dict=True,
                )

    mtq.quantize(model.language_model, mtq.INT4_AWQ_CFG, forward_loop=calibration_loop)
    if verbose:
        print("[int4] quantisation complete")


def export_llm_int4(
    model,
    config,
    onnx_dir: Path,
    n_layers: int,
    hidden_size: int,
    n_kv_heads: int,
    head_dim: int,
    n_img_tokens: int,
    image_token_index: int,
    text_mask_token_id: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[Path, Path]:
    """Re-export prefill+decode after modelopt.quantize. Same wrappers, same args.

    Returns (prefill_int4_path, decode_int4_path).
    """
    from .llm import LLMPrefill, LLMDecode, export_with_external_data

    onnx_dir = Path(onnx_dir)
    pre_path = onnx_dir / "llm_prefill_int4.onnx"
    dec_path = onnx_dir / "llm_decode_int4.onnx"

    lm_main = model.language_model.model
    lm_head = model.language_model.lm_head
    TEXT_PAD = 64
    S_trace = n_img_tokens + TEXT_PAD * 2

    # Prefill dummy
    ids_d = torch.zeros(1, S_trace, dtype=torch.long, device=device)
    ids_d[0, TEXT_PAD : TEXT_PAD + n_img_tokens] = int(image_token_index)
    vf_d = torch.zeros(n_img_tokens, hidden_size, dtype=dtype, device=device)
    pos_d = torch.arange(S_trace, dtype=torch.long, device=device)[None]
    msk_d = torch.ones(1, S_trace, dtype=torch.long, device=device)
    kv_out_names = sum([[f"present_k_{i}", f"present_v_{i}"] for i in range(n_layers)], [])
    dyn_pre = {
        "input_ids": {1: "S"},
        "visual_features": {0: "L_post"},
        "position_ids": {1: "S"},
        "attention_mask": {1: "S"},
        "logits": {1: "S"},
        **{n: {2: "S"} for n in kv_out_names},
    }
    print("[int4] exporting prefill ...")
    torch.onnx.export(
        LLMPrefill(lm_main, lm_head, int(image_token_index)).eval(),
        (ids_d, vf_d, pos_d, msk_d),
        str(pre_path),
        input_names=["input_ids", "visual_features", "position_ids", "attention_mask"],
        output_names=["logits", *kv_out_names],
        dynamic_axes=dyn_pre,
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )

    # Decode dummy
    K_trace, P_trace = 6, 32
    ids_dec = torch.full((1, K_trace), int(text_mask_token_id), dtype=torch.long, device=device)
    ids_dec[0, 0] = 0
    pd_ = torch.arange(P_trace, P_trace + K_trace, dtype=torch.long, device=device)[None]
    md = torch.ones(1, P_trace + K_trace, dtype=torch.long, device=device)
    past_args = []
    for _ in range(n_layers):
        past_args.append(torch.zeros(1, n_kv_heads, P_trace, head_dim, dtype=dtype, device=device))
        past_args.append(torch.zeros(1, n_kv_heads, P_trace, head_dim, dtype=dtype, device=device))
    in_kv_names = sum([[f"past_k_{i}", f"past_v_{i}"] for i in range(n_layers)], [])
    out_kv_names = sum([[f"present_k_{i}", f"present_v_{i}"] for i in range(n_layers)], [])
    dyn_dec = {
        "input_ids": {1: "K"},
        "position_ids": {1: "K"},
        "attention_mask": {1: "P_plus_K"},
        "logits": {1: "K"},
        **{n: {2: "P"} for n in in_kv_names},
        **{n: {2: "P_plus_K"} for n in out_kv_names},
    }
    print("[int4] exporting decode ...")
    torch.onnx.export(
        LLMDecode(lm_main, lm_head, n_layers).eval(),
        (ids_dec, pd_, md, *past_args),
        str(dec_path),
        input_names=["input_ids", "position_ids", "attention_mask", *in_kv_names],
        output_names=["logits", *out_kv_names],
        dynamic_axes=dyn_dec,
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    return pre_path, dec_path
