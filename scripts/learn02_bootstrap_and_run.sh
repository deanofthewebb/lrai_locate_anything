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
LEARN03_HOST="${LEARN03_HOST:-learn03}"          # SSH alias for learn03
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

# 2a. Model weights (~7.3 GB)
if [ ! -f "$WORKSPACE/weights/config.json" ]; then
  echo "[bootstrap] rsync weights from $LEARN03_HOST ..."
  mkdir -p "$WORKSPACE/weights"
  rsync -av --partial --append-verify \
    "$LEARN03_HOST:$LEARN03_WEIGHTS/" "$WORKSPACE/weights/" \
    | tail -20
else
  echo "[bootstrap] weights/ already present"
fi

# 2b. Counting-line configs (~12 KB)
mkdir -p "$WORKSPACE/lines_temp"
rsync -av "$LEARN03_HOST:$LEARN03_LINES/" "$WORKSPACE/lines_temp/" | tail -5

# 2c. Video clips (~22 GB)
mkdir -p "$WORKSPACE/clips"
echo "[bootstrap] rsync 12 ISP video clips from $LEARN03_HOST (idempotent --partial) ..."
rsync -av --partial --append-verify \
  --include='Gate A*.mp4' --include='*/' --exclude='*' \
  "$LEARN03_HOST:$LEARN03_VIDEOS/" "$WORKSPACE/clips/" \
  | tail -20

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
if [ ! -x "$WORKSPACE/venv/bin/python" ]; then
  # Use uv if available (much faster), fall back to system python
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.10 "$WORKSPACE/venv"
  else
    # Install uv inline (CI-safe path)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.10 "$WORKSPACE/venv"
  fi
  "$WORKSPACE/venv/bin/pip" install --quiet --upgrade pip
  "$WORKSPACE/venv/bin/pip" install --quiet --upgrade-strategy=eager \
    "torch~=2.5.0" --index-url https://download.pytorch.org/whl/cu121
  "$WORKSPACE/venv/bin/pip" install --quiet --upgrade-strategy=eager \
    "transformers>=4.55,<5.0" "tensorrt-cu12~=10.16.0" "cuda-python<12.9" \
    "accelerate" "peft" "huggingface_hub" "opencv-python-headless" \
    "Pillow" "numpy<2.0" "einops" "onnx" "onnxruntime-gpu"
  # Stub decord + lmdb (modeling_locateanything requires them for video paths we don't use)
  SP=$("$WORKSPACE/venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
  for name in decord lmdb; do
    [ -f "$SP/$name.py" ] || echo "# stub for inference-only setup" > "$SP/$name.py"
  done
  "$WORKSPACE/venv/bin/pip" install --quiet -e "$WORKSPACE/repo"
else
  echo "[bootstrap] venv present, updating package install ..."
  "$WORKSPACE/venv/bin/pip" install --quiet -e "$WORKSPACE/repo"
fi
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
