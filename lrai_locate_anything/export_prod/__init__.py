"""Production single-file export route.

The R&D path under `lrai_locate_anything/export/` produces 5 engines
(vision + projector + prefill + decode_mtp + decode_ar) joined by numpy
boundaries. This package fuses vision + patch_merger + projector into a
single ONNX graph for one-engine encoder inference. LLM unification is
Phase 3; this package owns Phase 1 only.
"""
