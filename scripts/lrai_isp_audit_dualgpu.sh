#!/usr/bin/env bash
# Dual-GPU ISP audit runner for learn02 (or any 2-GPU box).
# Two workers consume from a shared queue file, each pinned to one GPU.
# Resumable: clips with an existing non-empty CSV are skipped.
#
# Usage:
#   ./scripts/lrai_isp_audit_dualgpu.sh \
#       --videos  /home/lrm/Downloads/ISP_audits \
#       --lines   /home/lrm/Downloads/ISP_audits/lines_temp \
#       --out     /home/lrm/locany_test/audit_results \
#       --weights /home/lrm/locany_test/weights \
#       --venv    /home/lrm/locany_test/venv
#
# Defaults assume the learn02 layout the prior agent left behind. Override
# any path with the matching --flag.
#
# GPU pinning + per-GPU VRAM headroom:
#   GPU 0 -> RTX 3080 (10 GB)    --max-side 1280   (model is 7.74 GB bf16; 10 GB is tight)
#   GPU 1 -> RTX 3080 Ti (12 GB) --max-side 0      (full 4K resolution, fits)
# Swap per the actual numbering on your machine if learn02 reports differently.
#
# Output files (per-clip):
#   <out>/<key>_lrai_fps5.csv          counting CSV
#   <out>/<key>_lrai_fps5.log          stderr stream
#   <out>/<key>_overlay.mp4            QA overlay video (boxes + count lines)
#   <out>/queue.lock                   flock guard on the work queue
#   <out>/queue.txt                    remaining clips (rewritten by workers)

set -u -o pipefail

# ---- Args -----------------------------------------------------------------
VIDEOS_DIR="${VIDEOS_DIR:-/home/lrm/Downloads/ISP_audits}"
LINES_DIR="${LINES_DIR:-/home/lrm/Downloads/ISP_audits/lines_temp}"
OUT_DIR="${OUT_DIR:-/home/lrm/locany_test/audit_results}"
WEIGHTS="${WEIGHTS:-/home/lrm/locany_test/weights}"
VENV="${VENV:-/home/lrm/locany_test/venv}"
REPO="${REPO:-/home/lrm/locany_test/repo}"
TARGET_FPS="${TARGET_FPS:-5}"
PROMPT='Locate all the instances that matches the following description: roller bag</c>shoulder bag</c>carry-on</c>person.'

while [ $# -gt 0 ]; do
  case "$1" in
    --videos)  VIDEOS_DIR="$2"; shift 2 ;;
    --lines)   LINES_DIR="$2";  shift 2 ;;
    --out)     OUT_DIR="$2";    shift 2 ;;
    --weights) WEIGHTS="$2";    shift 2 ;;
    --venv)    VENV="$2";       shift 2 ;;
    --repo)    REPO="$2";       shift 2 ;;
    --fps)     TARGET_FPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"
QUEUE="$OUT_DIR/queue.txt"
LOCK="$OUT_DIR/queue.lock"

# ---- Build the work queue (size-ascending so smaller clips drain first) ---
# (key|filename) pairs
declare -A CLIPS=(
  [A5_F1]="Gate A5 (new)_Flight1_0028-0037.mp4"
  [A4_F1]="Gate A4 (new)_Flight1_1343-1357.mp4"
  [A2_F1]="Gate A2 (new)_Flight1_1102-1138.mp4"
  [A2_F2]="Gate A2 (new)_Flight2_1328-1400.mp4"
  [A5_F2]="Gate A5 (new)_Flight2_0800-0815.mp4"
  [A7_F1]="Gate A7 (new)_Flight1_1234-1306.mp4"
  [A7_F2]="Gate A7 (new)_Flight2_1635-1707.mp4"
  [A8_F1]="Gate A8 (new)_Flight1_1144-1212.mp4"
  [A3_F2]="Gate A3 (new)_Flight2_1733-1809.mp4"
  [A4_F2]="Gate A4 (new)_Flight2_1931-2010.mp4"
  [A3_F1]="Gate A3 (new)_Flight1_0955-1041.mp4"
  [A8_F2]="Gate A8 (new)_Flight2_1540-1628.mp4"
)
ORDER=(A5_F1 A4_F1 A2_F1 A2_F2 A5_F2 A7_F1 A7_F2 A8_F1 A3_F2 A4_F2 A3_F1 A8_F2)

# Skip clips already done; queue only what's pending
> "$QUEUE"
for key in "${ORDER[@]}"; do
  out_csv="$OUT_DIR/${key}_lrai_fps${TARGET_FPS}.csv"
  if [ -s "$out_csv" ]; then
    echo "[setup] SKIP $key — csv already present ($(stat -c%s "$out_csv") bytes)"
  else
    echo "$key" >> "$QUEUE"
  fi
done
n_pending=$(wc -l < "$QUEUE")
echo "[setup] queued $n_pending clips for processing"
if [ "$n_pending" -eq 0 ]; then
  echo "[setup] nothing to do."; exit 0
fi

# ---- Worker ---------------------------------------------------------------
worker() {
  local gpu=$1
  local max_side=$2
  local tag="GPU${gpu}"
  while true; do
    # Atomically pop the first line from the queue
    local key
    key=$(flock "$LOCK" bash -c "
      if [ ! -s '$QUEUE' ]; then
        exit 1
      fi
      head -1 '$QUEUE'
      tail -n +2 '$QUEUE' > '$QUEUE.tmp' && mv '$QUEUE.tmp' '$QUEUE'
    ") || break
    [ -z "$key" ] && break

    local fname="${CLIPS[$key]}"
    local out_csv="$OUT_DIR/${key}_lrai_fps${TARGET_FPS}.csv"
    local out_log="$OUT_DIR/${key}_lrai_fps${TARGET_FPS}.log"
    local out_video="$OUT_DIR/${key}_overlay.mp4"

    if [ -s "$out_csv" ]; then
      echo "[$tag] SKIP $key (csv already present)"
      continue
    fi

    echo "[$tag] === $key  $fname  (max_side=$max_side) ==="
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$VENV/bin/python" "$REPO/scripts/lrai_isp_audit.py" \
        --video "$VIDEOS_DIR/$fname" \
        --lines "$LINES_DIR/${key}.json" \
        --out-csv "$out_csv" \
        --out-video "$out_video" \
        --target-fps "$TARGET_FPS" \
        --weights "$WEIGHTS" \
        --prompt "$PROMPT" \
        --path pt \
        ${max_side:+--max-side "$max_side"} \
        2>&1 | tee "$out_log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
      echo "[$tag] $key DONE in ${elapsed}s (rc=$rc)"
    else
      echo "[$tag] $key FAILED in ${elapsed}s (rc=$rc) — see $out_log"
    fi
  done
  echo "[$tag] worker idle (queue empty) — exiting"
}

# ---- Spawn workers --------------------------------------------------------
# GPU 0 = 3080 (10 GB) -> downscale to 1280 longest side. 2048 OOM'd on model
#                          load (7.74 GB bf16 + activations > 10 GB); 1280
#                          leaves ~2 GB headroom for activations + decode.
# GPU 1 = 3080 Ti (12 GB) -> no downscale; full 4K fits
worker 0 1280 &
PID0=$!
worker 1 0 &
PID1=$!

echo "[main] launched worker GPU0 PID=$PID0  GPU1 PID=$PID1"
echo "[main] tail logs with: tail -F $OUT_DIR/*.log"

wait $PID0 $PID1
echo "[main] ===== ALL WORKERS DONE ====="

# Summary
echo "[main] per-clip status:"
for key in "${ORDER[@]}"; do
  csv="$OUT_DIR/${key}_lrai_fps${TARGET_FPS}.csv"
  if [ -s "$csv" ]; then
    rows=$(($(wc -l < "$csv") - 1))
    last=$(tail -1 "$csv" | awk -F',' '{print "in="$8" out="$9}')
    echo "  $key  rows=$rows  $last"
  else
    echo "  $key  MISSING"
  fi
done
