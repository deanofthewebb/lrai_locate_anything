"""Merge llm_prefill.onnx + llm_decode_ar.onnx into a single llm_unified.onnx.

Follows Optimum's decoder_model_merged.onnx pattern (optimum/onnx/
graph_transformations.py::merge_decoders). Both source graphs share the same
Qwen2 body weights — exported from the same torch.nn.Module — so initializers
are deduplicated (one shared set on the OUTER graph) and the two graphs become
the then/else subgraphs of a single If node. SDLM/MTP is intentionally dropped:
only the AR decode branch is merged. Predicate is a top-level BOOL[1] input
``use_cache_branch`` — False -> prefill (no past), True -> decode (with past).
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto, helper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _hash_initializer(init: onnx.TensorProto) -> str:
    """SHA-256 of an initializer's raw bytes (external-data already resolved)."""
    if init.raw_data:
        payload = init.raw_data
    else:
        # Small constants stored in typed fields — serialise the whole proto.
        payload = init.SerializeToString()
    h = hashlib.sha256()
    h.update(payload)
    h.update(str(tuple(init.dims)).encode())
    h.update(str(int(init.data_type)).encode())
    return h.hexdigest()


def _collect_initializers(model: onnx.ModelProto) -> Dict[str, Tuple[onnx.TensorProto, str]]:
    return {init.name: (init, _hash_initializer(init)) for init in model.graph.initializer}


def _strip_initializers_from_graph(graph: onnx.GraphProto) -> List[onnx.TensorProto]:
    """Remove initializers from a graph in-place and return the removed list.

    The subgraph still references these tensors by name; resolution happens
    via outer-graph scope capture (ONNX If semantics).
    """
    removed = list(graph.initializer)
    del graph.initializer[:]
    return removed


def _io_name_set(model: onnx.ModelProto, attr: str) -> List[str]:
    return [v.name for v in getattr(model.graph, attr)]


def _build_subgraph(name: str, body: onnx.ModelProto, output_order: List[str]) -> onnx.GraphProto:
    """Construct an If-branch subgraph: nodes + (reordered) outputs only.

    Inputs are empty — captured by name from the outer graph. Initializers
    have already been hoisted to the outer scope (caller's responsibility).
    """
    out_by_name = {v.name: v for v in body.graph.output}
    ordered_outputs = [out_by_name[n] for n in output_order]
    return helper.make_graph(
        nodes=list(body.graph.node),
        name=name,
        inputs=[],
        outputs=ordered_outputs,
        initializer=[],  # outer scope owns the weights
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def merge_llm_to_unified(
    prefill_onnx: Path,
    decode_onnx: Path,
    out_path: Path,
    *,
    dedup_initializers: bool = True,
) -> Path:
    """Merge prefill+decode-AR ONNX into a single If-gated llm_unified.onnx.

    The merged graph has one If node; ``use_cache_branch`` (BOOL[1]) selects:
      * False -> prefill subgraph (no past KV; consumes visual_features)
      * True  -> decode-AR subgraph (consumes past_k_i/past_v_i)

    Weights are deduped at the OUTER initializer table (verified by SHA-256);
    the two branches share one set of Qwen2 body weights.
    """
    prefill_onnx = Path(prefill_onnx)
    decode_onnx = Path(decode_onnx)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre = onnx.load(str(prefill_onnx), load_external_data=True)
    dec = onnx.load(str(decode_onnx), load_external_data=True)

    # ---- 1. I/O contract validation --------------------------------------
    pre_in, dec_in = _io_name_set(pre, "input"), _io_name_set(dec, "input")
    pre_out, dec_out = _io_name_set(pre, "output"), _io_name_set(dec, "output")

    # Logits + per-layer present_k/v must exist in both, and outputs must align.
    assert pre_out[0] == "logits" and dec_out[0] == "logits", (
        f"Expected first output 'logits'; got pre={pre_out[0]!r} dec={dec_out[0]!r}"
    )
    if pre_out != dec_out:
        raise ValueError(
            f"Output schema mismatch — prefill={pre_out} vs decode={dec_out}. "
            "Both graphs must declare logits + present_k/v_{i} in the same order."
        )
    unified_output_order = list(pre_out)

    # Layer count is implied by present_k_{i} naming; sanity-check decode past_kv.
    n_layers = sum(1 for n in pre_out if n.startswith("present_k_"))
    for i in range(n_layers):
        for tag in ("past_k_", "past_v_"):
            if f"{tag}{i}" not in dec_in:
                raise ValueError(f"decode ONNX missing input {tag}{i}")

    # ---- 2. Initializer dedup (SHA-256 by name) --------------------------
    pre_inits = _collect_initializers(pre)
    dec_inits = _collect_initializers(dec)

    shared_names = set(pre_inits) & set(dec_inits)
    if dedup_initializers:
        mismatched = [
            n for n in shared_names if pre_inits[n][1] != dec_inits[n][1]
        ]
        if mismatched:
            raise ValueError(
                f"{len(mismatched)} initializers share a name but differ by SHA-256 — "
                f"first few: {mismatched[:5]}. Re-export both ONNX from a single "
                "torch.onnx.export pass or fall back to dedup_initializers=False."
            )

    # Union: one entry per name. Prefer prefill's tensor for shared keys (bytes
    # are identical when dedup verification passed).
    merged_inits: Dict[str, onnx.TensorProto] = {}
    for name, (init, _h) in pre_inits.items():
        merged_inits[name] = init
    for name, (init, _h) in dec_inits.items():
        merged_inits.setdefault(name, init)

    # Strip initializers from each subgraph — they live at outer scope only.
    _strip_initializers_from_graph(pre.graph)
    _strip_initializers_from_graph(dec.graph)

    # ---- 3. gs round-trip for cleanup + namespace isolation --------------
    # Re-run gs.cleanup() over each body so any orphaned nodes are pruned
    # before they become subgraph contents that could collide on intermediate
    # tensor names with the other branch.
    for g in (pre, dec):
        gs_g = gs.import_onnx(g)
        gs_g.cleanup().toposort()
        # Export back: we only need node ordering / dead-node removal here.
        rebuilt = gs.export_onnx(gs_g)
        del g.graph.node[:]
        g.graph.node.extend(rebuilt.graph.node)

    # ---- 4. Build branch subgraphs ---------------------------------------
    else_branch = _build_subgraph("prefill_no_past", pre, unified_output_order)
    then_branch = _build_subgraph("decode_with_past", dec, unified_output_order)

    # ---- 5. Outer graph inputs ------------------------------------------
    # Union of (prefill ∪ decode) inputs, by name, preserving prefill order
    # first then decode-only inputs. Then append use_cache_branch.
    seen: set = set()
    outer_inputs: List[onnx.ValueInfoProto] = []
    for v in list(pre.graph.input) + list(dec.graph.input):
        if v.name in seen:
            continue
        seen.add(v.name)
        outer_inputs.append(v)

    use_cache_vi = helper.make_tensor_value_info(
        "use_cache_branch", TensorProto.BOOL, [1]
    )
    outer_inputs.append(use_cache_vi)

    # Predicate squeeze: If wants a scalar BOOL. Insert a Squeeze node.
    pred_node = helper.make_node(
        "Squeeze",
        inputs=["use_cache_branch", "use_cache_axes"],
        outputs=["use_cache_branch_scalar"],
        name="locany::squeeze_use_cache_branch",
    )
    axes_init = helper.make_tensor(
        "use_cache_axes", TensorProto.INT64, [1], [0]
    )

    # ---- 6. If node ------------------------------------------------------
    if_node = helper.make_node(
        "If",
        inputs=["use_cache_branch_scalar"],
        outputs=unified_output_order,
        name="locany::llm_if",
        then_branch=then_branch,
        else_branch=else_branch,
    )

    # ---- 7. Outer graph outputs (reuse prefill's typed ValueInfo) --------
    outer_outputs = list(pre.graph.output)

    # ---- 8. Assemble merged graph + model --------------------------------
    merged_initializer_list = list(merged_inits.values()) + [axes_init]
    merged_graph = helper.make_graph(
        nodes=[pred_node, if_node],
        name="llm_merged",
        inputs=outer_inputs,
        outputs=outer_outputs,
        initializer=merged_initializer_list,
    )

    # Union of opset imports; keep highest version per domain.
    opset_by_domain: Dict[str, int] = {}
    for op in list(pre.opset_import) + list(dec.opset_import):
        opset_by_domain[op.domain] = max(op.version, opset_by_domain.get(op.domain, 0))
    opset_imports = [helper.make_opsetid(d, v) for d, v in opset_by_domain.items()]

    merged_model = helper.make_model(
        merged_graph,
        producer_name="lrai-locany",
        opset_imports=opset_imports,
    )
    merged_model.ir_version = max(pre.ir_version, dec.ir_version)

    # ---- 9. Save with external data --------------------------------------
    onnx.save_model(
        merged_model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_path.stem + ".bin",
        size_threshold=1024,
    )
    return out_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_unified_onnx(unified_path: Path) -> dict:
    """Inspect the merged file and run onnx.checker on disk (handles >2 GB)."""
    unified_path = Path(unified_path)
    # Load WITHOUT external data so we don't need ~12 GB of RAM just to verify
    # graph topology; initializers are inspected by count alone.
    m = onnx.load(str(unified_path), load_external_data=False)

    if_nodes = [n for n in m.graph.node if n.op_type == "If"]
    has_if = bool(if_nodes)

    prefill_outs: List[str] = []
    decode_outs: List[str] = []
    if has_if:
        for attr in if_nodes[0].attribute:
            sub = attr.g
            names = [v.name for v in sub.output]
            if attr.name == "else_branch":
                prefill_outs = names
            elif attr.name == "then_branch":
                decode_outs = names

    # Path-based check (only safe form for >2 GB models).
    try:
        onnx.checker.check_model(str(unified_path))
        shape_check: object = True
    except Exception as e:  # pragma: no cover — surface the message verbatim
        shape_check = f"{type(e).__name__}: {e}"

    return {
        "n_initializers": len(m.graph.initializer),
        "has_if_node": has_if,
        "prefill_branch_outputs": prefill_outs,
        "decode_branch_outputs": decode_outs,
        "shape_check": shape_check,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Merge prefill+decode_ar -> llm_unified.onnx")
    p.add_argument("--prefill", required=True, type=Path)
    p.add_argument("--decode", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip SHA-256 dedup verification (debug only).")
    args = p.parse_args()

    out = merge_llm_to_unified(
        args.prefill, args.decode, args.out,
        dedup_initializers=not args.no_dedup,
    )
    print(f"[llm_unified] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB + .bin)")
    report = verify_unified_onnx(out)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    _main()
