"""Backward-compatibility shims for transformers >= 4.55.

The vendored modeling code in `nvidia/LocateAnything-3B` was written against an older
transformers API. Each shim here is a documented monkey-patch that surfaced as a real
crash during notebook iteration; the inline comments are the post-mortem.

`install_transformers_shims()` is idempotent and safe to call multiple times.
"""
from __future__ import annotations
import sys


def install_transformers_shims(verbose: bool = True) -> None:
    """Install all transformers / modeling shims required for nvidia/LocateAnything-3B
    to load and run under transformers >= 4.55. Order matters — DynamicCache must be
    patched before any model construction (`Qwen2Model.forward` references
    `to_legacy_cache()` mid-graph).
    """
    import transformers
    # transformers >=5.0 silently fails to fill embed_tokens.weight and lm_head.weight
    # from the safetensors checkpoint for the vendored LocateAnything modeling code.
    # The progress reports `Materializing param=...` for all 770 keys, missing_keys=[],
    # unexpected_keys=[], no errors — but the actual values stay at random init
    # (std == initializer_range == 0.02). All generation paths then mode-collapse
    # on <ref>$$$$$ / <ref>"""""". 4.55–4.57 work correctly. Reproduced locally
    # against 5.0.0 with the bare AutoModel.from_pretrained call.
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise RuntimeError(
            f"transformers=={transformers.__version__} is incompatible with "
            f"nvidia/LocateAnything-3B's vendored modeling code. "
            f"Transformers >=5.0 silently fails to load the safetensors checkpoint "
            f"(weights stay at random init, producing universal mode-collapse). "
            f"Install transformers<5.0 (e.g. 4.57.6): "
            f"`pip install \"transformers>=4.55,<5.0\"` and restart the runtime."
        )

    from transformers import modeling_utils as _mu
    from transformers.cache_utils import DynamicCache as _DC

    # ---------------------------------------------------------------------------
    # 1) _tied_weights_keys: legacy `List[str]` -> modern `Dict[str, str]`.
    # transformers >=4.55 calls .keys() on the value inside `post_init`. The
    # vendored Qwen2ForCausalLM declares it as a list. Older transformers don't
    # have the method at all — no shim needed there.
    # ---------------------------------------------------------------------------
    if hasattr(_mu.PreTrainedModel, "get_expanded_tied_weights_keys") and \
       not getattr(_mu.PreTrainedModel.get_expanded_tied_weights_keys, "_locany_patched", False):
        _orig_tied = _mu.PreTrainedModel.get_expanded_tied_weights_keys

        def _patched_tied(self, all_submodels: bool = False):
            tk = getattr(self, "_tied_weights_keys", None)
            if isinstance(tk, (list, tuple, set)):
                self._tied_weights_keys = {k: k for k in tk}
            elif tk is None:
                self._tied_weights_keys = {}
            return _orig_tied(self, all_submodels)

        _patched_tied._locany_patched = True
        _mu.PreTrainedModel.get_expanded_tied_weights_keys = _patched_tied
        if verbose:
            print("[shims] _tied_weights_keys list→dict upcast installed")

    # ---------------------------------------------------------------------------
    # 2) mark_tied_weights_as_initialized: gracefully no-op when post_init wasn't called.
    # The outer LocateAnythingForConditionalGeneration.__init__ skips post_init(); without
    # the shim, transformers' _finalize_load_state_dict hits an AttributeError on
    # `self.all_tied_weights_keys`. Only present in transformers >=4.55.
    # ---------------------------------------------------------------------------
    if hasattr(_mu.PreTrainedModel, "mark_tied_weights_as_initialized") and \
       not getattr(_mu.PreTrainedModel.mark_tied_weights_as_initialized, "_locany_patched", False):
        _orig_mark = _mu.PreTrainedModel.mark_tied_weights_as_initialized

        def _patched_mark(self):
            tk = getattr(self, "all_tied_weights_keys", None)
            if not tk:
                self.all_tied_weights_keys = {}
                return
            return _orig_mark(self)

        _patched_mark._locany_patched = True
        _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark
        if verbose:
            print("[shims] mark_tied_weights_as_initialized graceful no-op installed")

    # ---------------------------------------------------------------------------
    # 3) DynamicCache.to_legacy_cache: removed in transformers >=4.56.
    # The vendored modeling_qwen2.py at line 1385 calls
    # `next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache`.
    # ---------------------------------------------------------------------------
    if not hasattr(_DC, "to_legacy_cache") or not getattr(_DC.to_legacy_cache, "_locany_shim", False):
        def _to_legacy(self):
            legacy = ()
            if hasattr(self, "layers") and self.layers:
                for layer in self.layers:
                    # Explicit `is None` checks (NOT `a or b`) — `tensor or other_tensor`
                    # triggers a truthiness check, which raises "Boolean value of Tensor
                    # with more than one value is ambiguous" on filled cache layers.
                    k = getattr(layer, "keys", None)
                    if k is None:
                        k = getattr(layer, "key_cache", None)
                    v = getattr(layer, "values", None)
                    if v is None:
                        v = getattr(layer, "value_cache", None)
                    if k is not None and v is not None:
                        legacy += ((k, v),)
            elif hasattr(self, "key_cache") and hasattr(self, "value_cache"):
                for i in range(len(self.key_cache)):
                    legacy += ((self.key_cache[i], self.value_cache[i]),)
            return legacy

        _to_legacy._locany_shim = True
        _DC.to_legacy_cache = _to_legacy

        if not hasattr(_DC, "from_legacy_cache"):
            @classmethod
            def _from_legacy(cls, past):
                obj = cls()
                if past is None:
                    return obj
                for k, v in past:
                    idx = len(getattr(obj, "layers", []) or getattr(obj, "key_cache", []))
                    obj.update(k, v, layer_idx=idx)
                return obj

            _DC.from_legacy_cache = _from_legacy
        if verbose:
            print("[shims] DynamicCache.to_legacy_cache / from_legacy_cache restored")


def rehydrate_config(config, raw_json: dict) -> None:
    """Copy every text_config and vision_config field from the original config.json onto
    the loaded config object. transformers >=4.51 moved Qwen2's RoPE params into a
    `rope_parameters` dict, but the vendored modeling_qwen2 still reads
    `config.rope_theta` (and many other dotted attrs) directly.
    """
    for k, v in raw_json.get("text_config", {}).items():
        setattr(config.text_config, k, v)
    for k, v in raw_json.get("vision_config", {}).items():
        setattr(config.vision_config, k, v)
    assert hasattr(config.text_config, "rope_theta"), "rope_theta rehydration failed"
