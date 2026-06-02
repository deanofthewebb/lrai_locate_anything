"""Production single-engine TRT build route.

Mirror of `lrai_locate_anything/trt/` but for the fused graphs produced
by `lrai_locate_anything/export_prod/`. Phase 1 ships one engine
(vision_proj); Phase 3 adds the unified LLM engine.
"""
