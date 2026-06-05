-- LocateAnything-3B vision_proj PROD registry insert
-- Generated 2026-06-05 after validating ONNX-deployable path matches
-- the PT-vision W8A16 baseline.
--
-- Pipeline: locany.vision_proj.prod.onnx -> [trtexec build per-arch]
--           -> vision_proj.engine -> MoonViTAdapter(vision_mode='trt')
--           -> TRT-LLM ModelRunner(llm.int8 W8A16) -> detections
--
-- A5_F1 30-sec parity vs PT-vision baseline (4.03 dets/frame):
-- TRT-vision dets/frame=3.9139 (delta -2.88%)

INSERT INTO ml_models (model_id, location, config, model_type, created, updated, description, model_alias, default_parameters, training_id, engine_path, int8_cache, mapping, onnx, huggingface_repo) VALUES
('locany.vision_proj.prod.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.vision_proj.prod.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.vision_proj.prod.onnx',
 'onnx', '2026-06-05 18:00:00', '2026-06-05 18:00:00',
 'LocateAnything-3B fused vision encoder + static patch_merger + mlp1 projector (one ONNX graph, fp16). Bakes pos_emb at grid_hws=(36,46), patch_size=14, merge_kernel=2 over 644x504 letterboxed input. Output: visual_features fp16 (414, 2048) — drops directly into the TRT-LLM prompt_table for LocateAnythingTRTLLMRunner. Consumer: download .onnx + sibling .data sidecar and rename the data file to locany.vision_proj.prod.bin (graph internal reference). A5_F1 parity vs PT-vision baseline: 3.9139/frame (delta -2.88%).',
 'locany.vision_proj.prod',
 '{"detector_options": {"kind": "locateanything_vision_proj", "llm_model_id": "locany.llm.w8a16.poc.engine", "bake_grid_hws": [36, 46], "patch_size": 14, "merge_kernel": 2, "letterbox_to": [504, 644], "engine_build": "trtexec --onnx={location} --saveEngine=vision_proj.engine --fp16"}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);
