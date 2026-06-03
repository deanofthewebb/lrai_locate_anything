#!/usr/bin/env bash
# verify_trtllm_audit_e2e.sh — end-to-end TRT-LLM verification audit
#
# Run AFTER Phases B-G of the TRT-LLM adoption are complete. This script:
#   1. Sources the trtllm env (LD_LIBRARY_PATH for cu13 libs, venv activate)
#   2. Re-renders A5_F1 via the TRT-LLM pipeline using the audit script
#      with --path trtllm (a new flag Phase E must add to lrai_isp_audit.py)
#   3. Parses per-class IN/OUT counts from the resulting CSV
#   4. Asserts all 4 prompt classes show up (roller bags, carry-ons,
#      shoulder bags, people) — gate condition for production ready
#   5. scp the overlay mp4 + CSV to the Mac for visual QA
#   6. Prints a final pass/fail verdict
#
# Usage:
#   bash scripts/verify_trtllm_audit_e2e.sh [clip_key]
# Default clip_key = A5_F1
#
# Prerequisite (verified by pre-flight check):
#   - /mnt/ssd0/locany_test_lvl2/engines_trtllm/llm.engine exists (Phase C)
#   - /mnt/ssd0/locany_test_lvl2/engines_trtllm/vision_encoder.engine exists (Phase D)
#   - lrai_isp_audit.py supports --path trtllm (Phase E)

set -e -u -o pipefail

CLIP_KEY=${1:-A5_F1}
HOST=${HOST:-learn02}
TRTLLM_VENV=${TRTLLM_VENV:-/mnt/ssd0/locany_test_lvl2/venvs/trtllm}
ENGINE_DIR=${ENGINE_DIR:-/mnt/ssd0/locany_test_lvl2/engines_trtllm}
CLIPS_DIR=/mnt/ssd0/locany_test_lvl2/clips
LINES_DIR=/mnt/ssd0/locany_test_lvl2/lines_temp
WORK_DIR=/mnt/ssd0/locany_test_lvl2/audit_trtllm_e2e
MAC_OUT=/Users/deanwebb/Desktop/lrai_locate_anything/docs/audit_overlay_qa

declare -A CLIP_FILES=(
  [A5_F1]="Gate A5 (new)_Flight1_0028-0037.mp4"
)
CLIP_FILE=${CLIP_FILES[$CLIP_KEY]:-$CLIP_KEY.mp4}

echo "== Phase H — TRT-LLM end-to-end audit verification =="
echo "clip_key=$CLIP_KEY"
echo "host=$HOST"
echo ""

# 1. Pre-flight on the remote
echo "== Pre-flight =="
ssh "$HOST" "
set -e
[ -f \"$ENGINE_DIR/llm.engine\" ] || { echo \"MISSING: $ENGINE_DIR/llm.engine (Phase C not complete)\"; exit 1; }
[ -f \"$ENGINE_DIR/vision_encoder.engine\" ] || { echo \"MISSING: $ENGINE_DIR/vision_encoder.engine (Phase D not complete)\"; exit 1; }
source /mnt/ssd0/locany_test_lvl2/repo/scripts/trtllm_env.sh
python -c 'import tensorrt_llm; print(\"trtllm version:\", tensorrt_llm.__version__)'
"

# 2. Run the audit via TRT-LLM path
echo ""
echo "== Audit run =="
ssh "$HOST" "
set -e
mkdir -p $WORK_DIR
source /mnt/ssd0/locany_test_lvl2/repo/scripts/trtllm_env.sh
cd /mnt/ssd0/locany_test_lvl2/repo
CUDA_VISIBLE_DEVICES=1 python scripts/lrai_isp_audit.py \
  --video \"$CLIPS_DIR/$CLIP_FILE\" \
  --lines $LINES_DIR/$CLIP_KEY.json \
  --weights /mnt/ssd0/locany_test_lvl2/weights \
  --engine-dir $ENGINE_DIR \
  --out-csv $WORK_DIR/$CLIP_KEY.csv \
  --out-video $WORK_DIR/$CLIP_KEY.overlay.mp4 \
  --max-side 1280 --target-fps 5 \
  --path trtllm --tracker bytetrack 2>&1 | tail -40
"

# 3. scp the overlay + CSV to Mac
echo ""
echo "== scp to Mac =="
mkdir -p "$MAC_OUT"
scp "$HOST:$WORK_DIR/$CLIP_KEY.overlay.mp4" "$MAC_OUT/${CLIP_KEY}_trtllm.overlay.mp4"
scp "$HOST:$WORK_DIR/$CLIP_KEY.csv" "$MAC_OUT/${CLIP_KEY}_trtllm.csv"
ls -lh "$MAC_OUT/${CLIP_KEY}_trtllm.overlay.mp4" "$MAC_OUT/${CLIP_KEY}_trtllm.csv"

# 4. Parse per-class counts
echo ""
echo "== Per-class breakdown =="
python3 - <<'PYEOF'
import csv, os, sys

p = os.environ.get("CSV", "$MAC_OUT/${CLIP_KEY}_trtllm.csv")
pc_in, pc_out = {}, {}
total_rows = 0
with open(p) as f:
    r = csv.DictReader(f)
    print("CSV columns:", r.fieldnames)
    for row in r:
        total_rows += 1
        cls = (row.get("class") or "unknown").strip()
        d = row.get("direction", "").strip()
        if d == "IN": pc_in[cls] = pc_in.get(cls, 0) + 1
        elif d == "OUT": pc_out[cls] = pc_out.get(cls, 0) + 1

expected_classes = {"roller bags", "shoulder bags", "carry-ons", "people"}
observed = set(pc_in.keys()) | set(pc_out.keys())
missing = expected_classes - observed
unexpected = observed - expected_classes - {"unknown"}

print(f"total rows: {total_rows}")
classes = sorted(observed)
print(f"{'class':24s}  {'IN':>4s} {'OUT':>4s}")
print("-" * 40)
for c in classes:
    print(f"{c:24s}  {pc_in.get(c,0):4d} {pc_out.get(c,0):4d}")

print()
if missing:
    print(f"MISSING classes: {missing}")
    sys.exit(1)
if unexpected:
    print(f"UNEXPECTED classes: {unexpected}")
print("PASS: all 4 expected classes detected")
PYEOF

echo ""
echo "== DONE =="
echo "Overlay: $MAC_OUT/${CLIP_KEY}_trtllm.overlay.mp4"
echo "CSV:     $MAC_OUT/${CLIP_KEY}_trtllm.csv"
