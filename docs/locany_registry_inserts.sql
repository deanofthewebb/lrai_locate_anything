-- LocateAnything-3B ml_models registry inserts (POC)
--
-- All 5 ONNX models uploaded 2026-06-02 to s3://ml.livereachmedia.com/onnx/locany/.
-- Schema matches the parakeet-tdt example: 15 columns, .onnx.data sidecar in
-- `location`, .onnx graph in `config`. Three LLM rows have external-data
-- sidecars; their descriptions include a consumer rename instruction so
-- onnxruntime can resolve the graph's internal `<engine>.bin` reference.
--
-- Chain (via detector_options links):
--   vision -> projector -> prefill -> { decode_ar (terminal) | decode_mtp (terminal, hybrid mode) }
--
-- To rotate to production: replace `.poc.` with `.prod.` in model_id, location,
-- config, model_alias, default_parameters references, and re-upload under
-- s3://ml.livereachmedia.com/onnx/locany/locany.<engine>.prod.onnx*.

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.vision.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.vision.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.vision.poc.onnx',
 'onnx', '2026-06-02 18:00:00', '2026-06-02 18:00:00',
 'LocateAnything-3B vision encoder (MoonViT) — bakes pos_emb at grid_hws=(36,46) with patch_size=14, merge_kernel=2 over 644x504 letterboxed input. Output: vit_feats float16 (1656, 1152).',
 'locany.vision.poc',
 '{"detector_options": {"kind": "locateanything_vision", "projector_model_id": "locany.projector.poc.onnx", "bake_grid_hws": [36, 46], "patch_size": 14, "merge_kernel": 2}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.projector.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.projector.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.projector.poc.onnx',
 'onnx', '2026-06-02 18:00:00', '2026-06-02 18:00:00',
 'LocateAnything-3B vision-to-LLM projector (MLP: LayerNorm + Linear + GELU + Linear). Input: vit_feats_4x float16 (414, 4608); output: proj_feats float16 (414, 2048).',
 'locany.projector.poc',
 '{"detector_options": {"kind": "locateanything_projector", "prefill_model_id": "locany.llm_prefill.poc.onnx"}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.llm_prefill.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_prefill.poc.onnx.data',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_prefill.poc.onnx',
 'onnx', '2026-06-02 18:00:00', '2026-06-02 18:00:00',
 'LocateAnything-3B LLM prefill engine (Qwen2.5-3B body, bf16, STRONGLY_TYPED). Consumes input_ids + visual_features, emits first logit + KV cache (36 layers, head_dim 128). Consumer: after downloading the companion .onnx.data sidecar, rename it locally to llm_prefill.bin (the ONNX graph''s hard-coded external_data reference) so onnxruntime can load the external weights.',
 'locany.llm_prefill.poc',
 '{"detector_options": {"kind": "locateanything_llm_prefill", "decode_model_id": "locany.llm_decode_ar.poc.onnx", "decode_mtp_model_id": "locany.llm_decode.poc.onnx", "image_token_id": 151665, "box_start_token": 151666, "box_end_token": 151668, "eos_token": 151645, "n_layers": 36, "hidden_size": 2048}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.llm_decode_ar.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_decode_ar.poc.onnx.data',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_decode_ar.poc.onnx',
 'onnx', '2026-06-02 18:00:00', '2026-06-02 18:00:00',
 'LocateAnything-3B LLM AR-decode engine (Qwen2.5-3B body, bf16, AR attention baked). Consumes last_token + past_kv, emits next_token logits + updated KV. Terminal in the AR path. Consumer: rename the downloaded .onnx.data to llm_decode_ar.bin to satisfy the graph''s external_data reference.',
 'locany.llm_decode_ar.poc',
 '{"detector_options": {"kind": "locateanything_llm_decode_ar", "max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.llm_decode.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_decode.poc.onnx.data',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_decode.poc.onnx',
 'onnx', '2026-06-02 18:00:00', '2026-06-02 18:00:00',
 'LocateAnything-3B LLM MTP-decode engine (Qwen2.5-3B body, bf16, SDLM block-mask attention baked; block_size=6 for multi-token prediction). Alternate decoder for hybrid generation mode. Consumer: rename the downloaded .onnx.data to llm_decode.bin to satisfy the graph''s external_data reference.',
 'locany.llm_decode.poc',
 '{"detector_options": {"kind": "locateanything_llm_decode_mtp", "max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, "block_size": 6}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);

INSERT INTO `ml_models` (`model_id`, `location`, `config`, `model_type`, `created`, `updated`, `description`, `model_alias`, `default_parameters`, `training_id`, `engine_path`, `int8_cache`, `mapping`, `onnx`, `huggingface_repo`)
VALUES
('locany.llm_unified.poc.onnx',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_unified.poc.onnx.data',
 's3://ml.livereachmedia.com/onnx/locany/locany.llm_unified.poc.onnx',
 'onnx', '2026-06-03 10:00:00', '2026-06-03 10:00:00',
 'LocateAnything-3B unified LLM (prefill + AR decode merged via onnx-graphsurgeon If-node on use_cache_branch BOOL input; Qwen2.5-3B body, bf16). Single shared weight set (5.9 GB vs 11.6 GB for the separate prefill+decode_ar pair). MTP/SDLM branch dropped. Smoke-tested in onnxruntime CUDA EP. NOT compatible with TRT 10 IIfConditional build (data-dependent shapes in subgraphs); ship via onnxruntime-genai or as a backup artifact while TRT-LLM adoption proceeds. Consumer: rename the downloaded .onnx.data to llm_unified.bin (the ONNX graph''s hard-coded external_data reference) so onnxruntime can load the external weights.',
 'locany.llm_unified.poc',
 '{"detector_options": {"kind": "locateanything_llm_unified", "vision_model_id": "locany.vision.poc.onnx", "projector_model_id": "locany.projector.poc.onnx", "use_cache_branch_input": "use_cache_branch", "n_layers": 36, "hidden_size": 2048, "n_kv_heads": 2, "head_dim": 128, "image_token_id": 151665, "box_start_token": 151666, "box_end_token": 151668, "eos_token": 151645, "runtime": "onnxruntime-genai", "max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1}}',
 0, '', '', '{"kind": "detect"}', NULL, NULL);
