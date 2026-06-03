#!/usr/bin/env bash
# trtllm_env.sh — source this BEFORE running anything that imports tensorrt_llm.
#
# Purpose: TRT-LLM v1.2.1 ships TensorRT cu13 (libcublasLt.so.13, libcudnn.so.9)
# but the actual cu13 libraries from NVIDIA's pypi index land in non-standard
# paths under site-packages/nvidia/cu13/lib/ + nvidia/cudnn/lib/. Python's
# dynamic linker won't find them without LD_LIBRARY_PATH explicitly pointing
# at those dirs.
#
# Usage:
#   source scripts/trtllm_env.sh
#   python -m lrai_locate_anything.trtllm_prod.convert ...
#
# Or for one-off:
#   bash -c "source scripts/trtllm_env.sh && python ..."
#
# Expects: the trtllm venv at /mnt/ssd0/locany_test_lvl2/venvs/trtllm/ on
# learn02. Override via TRTLLM_VENV env var for other hosts.

TRTLLM_VENV="${TRTLLM_VENV:-/mnt/ssd0/locany_test_lvl2/venvs/trtllm}"
SITE_PACKAGES="${TRTLLM_VENV}/lib/python3.10/site-packages"

# cu13 + cudnn lib dirs (non-standard pip layout)
CU13_LIB="${SITE_PACKAGES}/nvidia/cu13/lib"
CUDNN_LIB="${SITE_PACKAGES}/nvidia/cudnn/lib"

# Pre-flight sanity
if [ ! -f "${CU13_LIB}/libcublasLt.so.13" ]; then
    echo "[trtllm_env] WARNING: libcublasLt.so.13 not found at ${CU13_LIB}" >&2
    echo "[trtllm_env] Run: pip install --extra-index-url https://pypi.nvidia.com nvidia-cublas nvidia-cudnn" >&2
fi

export LD_LIBRARY_PATH="${CU13_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
export PATH="${TRTLLM_VENV}/bin:${PATH}"

# Optional: activate venv so 'python', 'pip', 'trtllm-build' all resolve to the venv
if [ -f "${TRTLLM_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${TRTLLM_VENV}/bin/activate"
fi

# Pre-flight assertion: tensorrt must be 10.14.x (NOT 10.16 system) inside this env
PY_TRT_VER=$("${TRTLLM_VENV}/bin/python" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
if [[ "${PY_TRT_VER}" != 10.14.* ]] && [[ "${PY_TRT_VER}" != 10.9.* ]]; then
    echo "[trtllm_env] WARNING: tensorrt version ${PY_TRT_VER} is not what TRT-LLM v1.2.1 expects (10.14 or 10.9)" >&2
fi

echo "[trtllm_env] venv=${TRTLLM_VENV}"
echo "[trtllm_env] trt=${PY_TRT_VER}"
