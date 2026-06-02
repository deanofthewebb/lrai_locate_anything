#!/usr/bin/env bash
# learn02 bootstrap + dual-GPU ISP audit launcher.
# Idempotent: re-runs skip already-done steps.
#
# Run this on learn02 ONCE your lewm_test training is done (both GPUs free).
# It:
#   1. Verifies GPU availability (warns if anything else is using > 1 GB VRAM)
#   2. rsyncs weights, line configs, video clips from learn03 to /mnt/ssd0/
#      (using --partial --append-verify so resumes are cheap)
#   3. Clones / updates the lrai_locate_anything repo
#   4. Sets up a Python 3.10 uv venv with deps pinned (transformers<5.0)
#   5. Launches dual-GPU audit via scripts/lrai_isp_audit_dualgpu.sh
#
# Workspace is /mnt/ssd0/locany_test_lvl2 (learn02 has only 38 GB on /,
# but 359 GB on /mnt/ssd0; model is 7.3 GB + videos ~22 GB + engines ~21 GB).

set -u -o pipefail

WORKSPACE="/mnt/ssd0/locany_test_lvl2"
# learn02 cannot resolve "learn03" by name (no DNS / no tailscale magicdns).
# Default to learn03's tailscale IP, but allow override (set LEARN03_HOST=skip
# when assets are being pushed in externally, e.g. via a Mac-mediated tar relay).
LEARN03_HOST="${LEARN03_HOST:-lrm@100.69.48.85}"
LEARN03_WEIGHTS="/home/lrm/locany_test/weights"
LEARN03_LINES="/home/lrm/locany_test/lines_temp"
LEARN03_VIDEOS="/mnt/ssd1/people_count/clips/ISP"
LEARN03_REPO="/home/lrm/locany_test/repo"        # only as a fallback for clone

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

echo "[bootstrap] === Step 1/5: GPU availability check ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -rn | head -1)
if [ "${USED:-0}" -gt 1000 ]; then
  echo "[bootstrap] WARNING: a GPU is using >${USED} MiB. Another process is active."
  echo "[bootstrap] Continuing anyway, but expect OOM if VRAM is squeezed."
fi

echo ""
echo "[bootstrap] === Step 2/5: rsync weights + lines + videos from $LEARN03_HOST ==="

if [ "$LEARN03_HOST" = "skip" ]; then
  echo "[bootstrap] LEARN03_HOST=skip — assuming assets are being delivered externally."
else
  # 2a. Model weights (~7.3 GB)
  if [ ! -f "$WORKSPACE/weights/config.json" ]; then
    echo "[bootstrap] rsync weights from $LEARN03_HOST ..."
    mkdir -p "$WORKSPACE/weights"
    rsync -av --partial --append-verify \
      "$LEARN03_HOST:$LEARN03_WEIGHTS/" "$WORKSPACE/weights/" \
      | tail -20 || echo "[bootstrap] weights rsync failed (will validate below)"
  else
    echo "[bootstrap] weights/ already present"
  fi

  # 2b. Counting-line configs (~12 KB)
  mkdir -p "$WORKSPACE/lines_temp"
  rsync -av "$LEARN03_HOST:$LEARN03_LINES/" "$WORKSPACE/lines_temp/" | tail -5 \
    || echo "[bootstrap] lines rsync failed (will validate below)"

  # 2c. Video clips (~22 GB)
  mkdir -p "$WORKSPACE/clips"
  echo "[bootstrap] rsync 12 ISP video clips from $LEARN03_HOST (idempotent --partial) ..."
  rsync -av --partial --append-verify \
    --include='Gate A*.mp4' --include='*/' --exclude='*' \
    "$LEARN03_HOST:$LEARN03_VIDEOS/" "$WORKSPACE/clips/" \
    | tail -20 || echo "[bootstrap] clips rsync failed (will validate below)"
fi

# 2d. Validate that assets actually landed (regardless of how they got here).
echo "[bootstrap] validating workspace assets ..."
MISSING=0
if [ ! -f "$WORKSPACE/weights/config.json" ]; then
  echo "[bootstrap] MISSING: $WORKSPACE/weights/config.json"
  MISSING=1
fi
N_CLIPS=$(find "$WORKSPACE/clips" -maxdepth 1 -name 'Gate A*.mp4' 2>/dev/null | wc -l)
if [ "$N_CLIPS" -lt 12 ]; then
  echo "[bootstrap] only $N_CLIPS / 12 video clips present in $WORKSPACE/clips"
  MISSING=1
fi
if [ "$MISSING" -ne 0 ]; then
  echo "[bootstrap] ABORT: required assets missing. Refusing to launch audit on empty workspace."
  echo "[bootstrap] Push assets from another host then re-run this script (it's idempotent)."
  exit 2
fi
echo "[bootstrap] OK: weights present, $N_CLIPS/12 clips present"

echo ""
echo "[bootstrap] === Step 3/5: clone / update repo ==="
if [ ! -d "$WORKSPACE/repo/.git" ]; then
  git clone --depth=1 https://github.com/deanofthewebb/lrai_locate_anything.git \
    "$WORKSPACE/repo"
else
  ( cd "$WORKSPACE/repo" && git fetch --depth=1 origin main && git reset --hard origin/main )
fi
( cd "$WORKSPACE/repo" && git log -1 --oneline )

echo ""
echo "[bootstrap] === Step 4/5: venv + deps ==="
# Make sure uv is on PATH (installer drops it in ~/.local/bin).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$WORKSPACE/venv/bin/python" ]; then
  uv venv --python 3.10 "$WORKSPACE/venv"
fi

# `uv venv` does NOT install a pip binary into the venv. Use `uv pip install
# --python <venv-python>` to install into that venv without needing pip.
UV_PIP=( uv pip install --python "$WORKSPACE/venv/bin/python" )

# Check whether deps are already installed; skip the heavy step if so.
if ! "$WORKSPACE/venv/bin/python" -c "import torch, transformers, numpy" >/dev/null 2>&1; then
  echo "[bootstrap] installing torch+deps into venv via uv pip ..."
  "${UV_PIP[@]}" "torch~=2.5.0" --index-url https://download.pytorch.org/whl/cu121
  "${UV_PIP[@]}" \
    "transformers>=4.55,<5.0" "tensorrt-cu12~=10.16.0" "cuda-python<12.9" \
    "accelerate" "peft" "huggingface_hub" "opencv-python-headless" \
    "Pillow" "numpy<2.0" "einops" "onnx" "onnxruntime-gpu"
  # Stub decord + lmdb (modeling_locateanything requires them for video paths we don't use)
  SP=$("$WORKSPACE/venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
  for name in decord lmdb; do
    [ -f "$SP/$name.py" ] || echo "# stub for inference-only setup" > "$SP/$name.py"
  done
else
  echo "[bootstrap] venv already has torch+transformers+numpy; skipping heavy install"
fi
# Always (re-)install the package in editable mode so a refreshed repo is picked up.
"${UV_PIP[@]}" -e "$WORKSPACE/repo"
"$WORKSPACE/venv/bin/python" -c "
import torch, transformers, lrai_locate_anything as L
print(f'  torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__}')
print(f'  lrai_locate_anything={L.__version__}')
print(f'  GPUs visible: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    [{i}] {torch.cuda.get_device_name(i)}  '
          f'{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB  '
          f'cc={torch.cuda.get_device_capability(i)}')
"

echo ""
echo "[bootstrap] === Step 5/5: launch dual-GPU audit ==="
mkdir -p "$WORKSPACE/audit_results"

# Launch in the foreground if interactive, in nohup if not
LAUNCHER="$WORKSPACE/repo/scripts/lrai_isp_audit_dualgpu.sh"
if [ ! -x "$LAUNCHER" ]; then
  chmod +x "$LAUNCHER"
fi

if [ -t 0 ]; then
  exec "$LAUNCHER" \
    --videos "$WORKSPACE/clips" \
    --lines  "$WORKSPACE/lines_temp" \
    --out    "$WORKSPACE/audit_results" \
    --weights "$WORKSPACE/weights" \
    --venv   "$WORKSPACE/venv" \
    --repo   "$WORKSPACE/repo"
else
  # Non-interactive: nohup so the user can disconnect
  nohup "$LAUNCHER" \
    --videos "$WORKSPACE/clips" \
    --lines  "$WORKSPACE/lines_temp" \
    --out    "$WORKSPACE/audit_results" \
    --weights "$WORKSPACE/weights" \
    --venv   "$WORKSPACE/venv" \
    --repo   "$WORKSPACE/repo" \
    > "$WORKSPACE/full_audit.out" 2>&1 &
  echo "[bootstrap] launched in background (pid $!); tail $WORKSPACE/full_audit.out"
fi
