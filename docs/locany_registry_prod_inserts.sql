-- LocateAnything-3B vision_proj PROD registry insert
-- Generated 2026-06-05. Validated end-to-end on the full A5_F1 audit clip
-- (2706 frames @ 5 fps target, 8.5 min source video) against three other
-- LLM/vision configurations. See 4-way comparison in commit message and
-- /Users/deanwebb/Desktop/locany_trtllm_trt_vision_audit_20260605_145212/.
--
-- Pipeline:
--   locany.vision_proj.prod.onnx + .bin sidecar
--     -> [trtexec --fp16 build per-target-GPU] -> vision_proj.engine
--     -> MoonViTAdapter(vision_mode='trt')
--     -> TRT-LLM ModelRunner(llm.int8 W8A16) -> detections
--
-- Recommended companion LLM engine: W8A16 (int8 weight-only, bf16 acts).
-- That config delivers ai_fps=2.31, 24 min wall, 100 OUT crossings on A5_F1.
--
-- Full-audit parity vs PT-vision baselines:
--   PT+bf16     -> 4.03 dets/frame, ai_fps=1.92, 97 OUT
--   PT+W8A16    -> 3.80 dets/frame, ai_fps=2.34, 102 OUT
--   TRT+bf16    -> 3.79 dets/frame, ai_fps=1.69, 92 OUT
--   TRT+W8A16   -> 3.79 dets/frame, ai_fps=2.31, 100 OUT  <-- THIS ROW
--
-- Detection density delta TRT+W8A16 vs PT+W8A16: -0.26% (3.79 vs 3.80).
-- People OUT crossings identical (63/63). Customer KPI preserved.

INSERT INTO ml_models (
    model_id, location, config, model_type, created, updated,
    description, model_alias, default_parameters, training_id,
    engine_path, int8_cache, mapping, onnx, huggingface_repo
) VALUES (
    'locany.vision_proj.prod.onnx',
    's3://ml.livereachmedia.com/onnx/locany/locany.vision_proj.prod.onnx',
    's3://ml.livereachmedia.com/onnx/locany/locany.vision_proj.prod.onnx',
    'onnx',
    '2026-06-05 18:00:00',
    '2026-06-05 22:00:00',
    'LocateAnything-3B fused vision encoder + static patch_merger + mlp1 projector (one ONNX graph, fp16, opset 17). Bakes pos_emb at grid_hws=(36,46), patch_size=14, merge_kernel=2 over 644x504 letterboxed input. Output: visual_features fp16 (414, 2048) — drops directly into the TRT-LLM prompt_table for LocateAnythingTRTLLMRunner. Consumer: download .onnx (0.6 MB graph) + sibling .bin (855.6 MB external_data sidecar) and rename the .bin to locany.vision_proj.prod.bin (graph internal reference). Build per-arch via `trtexec --onnx=... --saveEngine=vision_proj.engine --fp16 --memPoolSize=workspace:4096` (typical engine size ~820 MB). FULL-CLIP A5_F1 parity vs PT-vision W8A16 baseline: 3.79/frame vs 3.80/frame (-0.26%); 63 people OUT crossings identical. Ship with W8A16 LLM engine for ai_fps=2.31 (matches PT+W8A16 throughput).',
    'locany.vision_proj.prod',
    '{"detector_options": {"kind": "locateanything_vision_proj", "llm_engine_quant_recommended": "W8A16", "bake_grid_hws": [36, 46], "patch_size": 14, "merge_kernel": 2, "input_shape": [414, 3, 14, 14], "output_shape": [414, 2048], "output_dtype": "float16", "letterbox_to_hw": [504, 644], "letterbox_fill": [128, 128, 128], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "external_data_filename": "locany.vision_proj.prod.bin", "engine_build_cmd": "trtexec --onnx={location} --saveEngine=vision_proj.engine --fp16 --memPoolSize=workspace:4096", "runner_class": "lrai_locate_anything.trtllm_prod.runner.LocateAnythingTRTLLMRunner", "runner_kwargs": {"vision_mode": "trt"}}}',
    0,
    '',
    '',
    '{"kind": "detect"}',
    NULL,
    NULL
);
