"""Qwen2 LLM export: prefill + dynamic-K decode with externalised KV cache.

Canonical contract (from vendored modeling_qwen2.py:1184-1227):
  - Pass exactly one of (input_ids, inputs_embeds), never both (line 1200 raises).
  - Pass `visual_features` + `image_token_index` for multimodal — Qwen2Model's
    image_processing() embeds input_ids and scatters visual_features internally.
  - The SDLM block-mask path at line 1253 reads input_ids[0][-1] to decide MTP vs AR.

So the wrappers pass input_ids (+ visual_features for prefill). NEVER inputs_embeds.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Tuple

import torch
# onnx is imported lazily inside export_with_external_data so the package can be
# imported (and the wrapper class signatures inspected) without a heavy dep chain.


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class LLMPrefill(torch.nn.Module):
    """Wraps Qwen2Model.forward for prefill. Canonical contract: input_ids + visual_features."""

    def __init__(self, m: torch.nn.Module, h: torch.nn.Module, image_token_index: int):
        super().__init__()
        self.m = m
        self.h = h
        self.image_token_index = int(image_token_index)

    def forward(self, input_ids, visual_features, position_ids, attention_mask):
        o = self.m(
            input_ids=input_ids,
            visual_features=visual_features,
            image_token_index=self.image_token_index,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        # Match canonical Qwen2ForCausalLM.forward: fp32 logits.
        logits = self.h(o.last_hidden_state).float()
        flat = []
        for (k, v) in o.past_key_values:
            flat.append(k)
            flat.append(v)
        return (logits, *flat)


class LLMDecode(torch.nn.Module):
    """Wraps Qwen2Model.forward for decode (AR K=1 or MTP K=6). No visual_features."""

    def __init__(self, m: torch.nn.Module, h: torch.nn.Module, n_layers: int):
        super().__init__()
        self.m = m
        self.h = h
        self.n = n_layers

    def forward(self, input_ids, position_ids, attention_mask, *past_kv):
        pkv = [(past_kv[2 * i], past_kv[2 * i + 1]) for i in range(self.n)]
        o = self.m(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=tuple(pkv),
            use_cache=True,
            return_dict=True,
        )
        logits = self.h(o.last_hidden_state).float()
        flat = []
        for (k, v) in o.past_key_values:
            flat.append(k)
            flat.append(v)
        return (logits, *flat)


# ---------------------------------------------------------------------------
# Helper: large-model export with external data
# ---------------------------------------------------------------------------
def export_with_external_data(wrap, args, onnx_path: Path, **kw) -> None:
    """torch.onnx.export -> onnx.save_model(save_as_external_data=True).

    Forces dynamo=False; the modern dynamo path raises GuardOnDataDependentSymNode on
    `.item()` / `int(tensor_scalar)` calls. The legacy exporter bakes constants which
    is exactly what we want for fixed-resolution engines.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export] cached: {onnx_path}")
        return

    import onnx  # lazy: only needed inside this function
    tmp = onnx_path.parent / (onnx_path.stem + "_tmp")
    tmp.mkdir(exist_ok=True)
    tmp_file = tmp / "model.onnx"
    kw.setdefault("dynamo", False)
    torch.onnx.export(wrap, args, str(tmp_file), **kw)

    m = onnx.load(str(tmp_file), load_external_data=True)
    onnx.save_model(
        m, str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_path.stem + ".bin",
    )
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[export] saved {onnx_path}  (+ side-car .bin)")


# ---------------------------------------------------------------------------
# Driver functions
# ---------------------------------------------------------------------------
def export_llm_prefill(
    lm_main: torch.nn.Module,
    lm_head: torch.nn.Module,
    image_token_index: int,
    n_layers: int,
    hidden_size: int,
    n_img_tokens: int,
    onnx_path: Path,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    s_trace_text_pad: int = 64,
) -> Path:
    """Export the LLM prefill graph.

    n_img_tokens: number of post-merger image tokens (= L_pre / 4). The dummy input_ids
    has image_token_index at exactly that many positions; visual_features has that many rows.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export] cached: {onnx_path}")
        return onnx_path

    S_trace = n_img_tokens + s_trace_text_pad * 2
    ids_d = torch.zeros(1, S_trace, dtype=torch.long, device=device)
    # IMPORTANT: last position must NOT be text_mask_token_id so the block-mask path
    # takes its early-return branch (matches chat-template behaviour at runtime).
    ids_d[0, s_trace_text_pad : s_trace_text_pad + n_img_tokens] = int(image_token_index)
    assert int((ids_d == int(image_token_index)).sum().item()) == n_img_tokens

    vf_d = torch.zeros(n_img_tokens, hidden_size, dtype=dtype, device=device)
    pos_d = torch.arange(S_trace, dtype=torch.long, device=device)[None]
    msk_d = torch.ones(1, S_trace, dtype=torch.long, device=device)

    kv_out_names = sum([[f"present_k_{i}", f"present_v_{i}"] for i in range(n_layers)], [])
    dyn = {
        "input_ids": {1: "S"},
        "visual_features": {0: "L_post"},
        "position_ids": {1: "S"},
        "attention_mask": {1: "S"},
        "logits": {1: "S"},
        **{n: {2: "S"} for n in kv_out_names},
    }
    export_with_external_data(
        LLMPrefill(lm_main, lm_head, image_token_index).eval(),
        (ids_d, vf_d, pos_d, msk_d),
        onnx_path,
        input_names=["input_ids", "visual_features", "position_ids", "attention_mask"],
        output_names=["logits", *kv_out_names],
        dynamic_axes=dyn,
        opset_version=17,
        do_constant_folding=False,  # save peak RAM during export
    )
    return onnx_path


def export_llm_decode(
    lm_main: torch.nn.Module,
    lm_head: torch.nn.Module,
    n_layers: int,
    hidden_size: int,
    n_kv_heads: int,
    head_dim: int,
    text_mask_token_id: int,
    onnx_path: Path,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    k_trace: int = 6,
    p_trace: int = 32,
) -> Path:
    """Export the LLM decode graph (dynamic K ∈ {1, 6}, dynamic past_seq).

    The dummy input_ids puts text_mask_token_id at the LAST position so the SDLM
    block-mask branch is captured for K=6 (MTP). For K=1 at runtime the canonical
    short-circuits on seq_length==1 regardless of last-token value.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        print(f"[export] cached: {onnx_path}")
        return onnx_path

    ids_dec = torch.full((1, k_trace), int(text_mask_token_id), dtype=torch.long, device=device)
    ids_dec[0, 0] = 0  # non-mask first position (last committed real token)
    pd_ = torch.arange(p_trace, p_trace + k_trace, dtype=torch.long, device=device)[None]
    md = torch.ones(1, p_trace + k_trace, dtype=torch.long, device=device)
    past_args = []
    for _ in range(n_layers):
        past_args.append(torch.zeros(1, n_kv_heads, p_trace, head_dim, dtype=dtype, device=device))
        past_args.append(torch.zeros(1, n_kv_heads, p_trace, head_dim, dtype=dtype, device=device))

    in_kv_names = sum([[f"past_k_{i}", f"past_v_{i}"] for i in range(n_layers)], [])
    out_kv_names = sum([[f"present_k_{i}", f"present_v_{i}"] for i in range(n_layers)], [])
    dyn = {
        "input_ids": {1: "K"},
        "position_ids": {1: "K"},
        "attention_mask": {1: "P_plus_K"},
        "logits": {1: "K"},
        **{n: {2: "P"} for n in in_kv_names},
        **{n: {2: "P_plus_K"} for n in out_kv_names},
    }
    export_with_external_data(
        LLMDecode(lm_main, lm_head, n_layers).eval(),
        (ids_dec, pd_, md, *past_args),
        onnx_path,
        input_names=["input_ids", "position_ids", "attention_mask", *in_kv_names],
        output_names=["logits", *out_kv_names],
        dynamic_axes=dyn,
        opset_version=17,
        do_constant_folding=False,
    )
    return onnx_path
