# =============================================================================
# Colab driver: full 12-clip ISP counting-line audit using lrai_locate_anything
# =============================================================================
# Paste this WHOLE block into a single Colab cell. Runs in foreground; the cell
# blocks until all 12 clips finish.
#
# WHAT IT DOES
#   1. pip install lrai_locate_anything from main (latest commit)
#   2. Mount Google Drive
#   3. Loop through 12 clips × target_fps=5; for each:
#        a. Run scripts/lrai_isp_audit.py via subprocess
#        b. Stream stderr to a log file in Drive (also visible in real time)
#        c. Save per-clip CSV to Drive — that's your live-updating
#           "results as they come" path (Drive desktop client syncs to Mac)
#   4. Resumable: skips clips that already have a complete CSV in Drive
#
# PREREQUISITES
#   - 12 video clips uploaded to Drive at: /content/drive/MyDrive/ISP_audits/
#       (the same filenames as ~/Downloads/ISP_audits on the Mac:
#        "Gate A2 (new)_Flight1_1102-1138.mp4" etc.)
#   - 12 lines_temp/*.json files uploaded to Drive at:
#       /content/drive/MyDrive/ISP_audits/lines_temp/
#       (A2_F1.json, A2_F2.json, ..., A8_F2.json)
#
# Expected throughput on Colab A100 PT path: 5-10 ai_fps. Total wall-clock for
# all 12 clips at fps=5: ~4-8 hours. Resumable so a session timeout just means
# re-run this cell.
#
# Results land in:
#   /content/drive/MyDrive/ISP_audits/results_lrai/
#     <key>_lrai_fps5.csv       per-clip crossings CSV (tracker_bot schema)
#     <key>_lrai_fps5.log       per-clip stderr (ai_fps, proc_fps, progress)
#     _summary.csv              GT vs prediction table (regenerated after each clip)
# =============================================================================

import os, sys, json, time, subprocess, shutil, csv
from pathlib import Path

# ---- Config ----------------------------------------------------------------
DRIVE_ROOT      = Path("/content/drive/MyDrive/ISP_audits")
VIDEOS_DIR      = DRIVE_ROOT
LINES_DIR       = DRIVE_ROOT / "lines_temp"
RESULTS_DIR     = DRIVE_ROOT / "results_lrai"
WEIGHTS_LOCAL   = Path("/content/locany_weights")
REPO_LOCAL      = Path("/content/lrai_locate_anything")
TARGET_FPS      = 5
PROMPT          = "Locate all the instances that matches the following description: people."

# (key, filename) — ascending by clip duration so smaller clips arrive first
CLIPS = [
    ("A5_F1", "Gate A5 (new)_Flight1_0028-0037.mp4"),
    ("A4_F1", "Gate A4 (new)_Flight1_1343-1357.mp4"),
    ("A2_F1", "Gate A2 (new)_Flight1_1102-1138.mp4"),
    ("A2_F2", "Gate A2 (new)_Flight2_1328-1400.mp4"),
    ("A5_F2", "Gate A5 (new)_Flight2_0800-0815.mp4"),
    ("A7_F1", "Gate A7 (new)_Flight1_1234-1306.mp4"),
    ("A7_F2", "Gate A7 (new)_Flight2_1635-1707.mp4"),
    ("A8_F1", "Gate A8 (new)_Flight1_1144-1212.mp4"),
    ("A3_F2", "Gate A3 (new)_Flight2_1733-1809.mp4"),
    ("A4_F2", "Gate A4 (new)_Flight2_1931-2010.mp4"),
    ("A3_F1", "Gate A3 (new)_Flight1_0955-1041.mp4"),
    ("A8_F2", "Gate A8 (new)_Flight2_1540-1628.mp4"),
]

# Ground truth from head counts.xlsx (Audit Predictions sheet, manually transcribed)
GT = {
    "A2_F1": (59, 142), "A2_F2": (107, 118),
    "A3_F1": (189, 187), "A3_F2": (144, 216),
    "A4_F1": (18, 190),  "A4_F2": (148, 182),
    "A5_F1": (0, 80),    "A5_F2": (63, 1),
    "A7_F1": (149, 162), "A7_F2": (138, 199),
    "A8_F1": (126, 150), "A8_F2": (147, 205),
}

# ---- Mount Drive ----------------------------------------------------------
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Install lrai_locate_anything from main -------------------------------
print("[setup] cloning + installing lrai_locate_anything @ main ...", flush=True)
if not REPO_LOCAL.exists():
    subprocess.check_call(["git", "clone", "--depth=1",
                           "https://github.com/deanofthewebb/lrai_locate_anything.git",
                           str(REPO_LOCAL)])
else:
    subprocess.check_call(["git", "-C", str(REPO_LOCAL), "fetch", "--depth=1", "origin", "main"])
    subprocess.check_call(["git", "-C", str(REPO_LOCAL), "reset", "--hard", "origin/main"])
sha = subprocess.check_output(["git", "-C", str(REPO_LOCAL), "rev-parse", "--short", "HEAD"]).decode().strip()
print(f"[setup] repo @ {sha}", flush=True)

# Install with deps pinned (transformers<5.0 mandatory)
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                       "--upgrade-strategy=eager", "transformers>=4.55,<5.0"])
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                       "--no-cache-dir", "--force-reinstall", "--no-deps",
                       f"git+https://github.com/deanofthewebb/lrai_locate_anything.git@{sha}"])

# Stub decord + lmdb (modeling_locateanything requires them for unused video code)
import sys, types, site
sp = next(p for p in site.getsitepackages() if "dist-packages" in p or "site-packages" in p)
for name in ("decord", "lmdb"):
    f = Path(sp) / f"{name}.py"
    if not f.exists():
        f.write_text("# stub for inference-only setup\n")

# ---- Download model weights (cached across cell re-runs) ------------------
if not (WEIGHTS_LOCAL / "config.json").exists():
    print("[setup] downloading nvidia/LocateAnything-3B weights (~8 GB) ...", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/LocateAnything-3B", local_dir=str(WEIGHTS_LOCAL),
                      allow_patterns=["*.json", "*.py", "*.txt", "*.safetensors",
                                        "*.safetensors.index.json", "tokenizer*",
                                        "chat_template*", "generation_config*"])

# ---- Verify videos + lines present ----------------------------------------
missing_v = [f for _, f in CLIPS if not (VIDEOS_DIR / f).exists()]
missing_l = [f"{k}.json" for k, _ in CLIPS if not (LINES_DIR / f"{k}.json").exists()]
if missing_v or missing_l:
    print(f"[setup] ERROR: missing in Drive:")
    for v in missing_v: print(f"  {VIDEOS_DIR/v}")
    for l in missing_l: print(f"  {LINES_DIR/l}")
    print("\nUpload these to your Drive at the paths above, then re-run this cell.")
    raise SystemExit(2)

# ---- Audit loop ------------------------------------------------------------
def _summary_row(key: str, csv_path: Path) -> dict:
    """Parse the per-clip CSV's final in_count_after / out_count_after."""
    if not csv_path.exists():
        return {"key": key, "pred_in": "", "pred_out": "", "ai_fps": "", "proc_fps": ""}
    in_count = out_count = 0
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                in_count = max(in_count, int(row["in_count_after"]))
                out_count = max(out_count, int(row["out_count_after"]))
            except (ValueError, KeyError):
                pass
    # Pull ai_fps + proc_fps from log if present
    log_path = csv_path.with_suffix(".log")
    ai_fps = proc_fps = ""
    if log_path.exists():
        for line in log_path.read_text().splitlines()[-30:]:
            if "ai_fps " in line and "=" in line:
                ai_fps = line.split("ai_fps")[-1].split("=")[-1].strip().split()[0]
            if "proc_fps " in line and "=" in line:
                proc_fps = line.split("proc_fps")[-1].split("=")[-1].strip().split()[0]
    return {"key": key, "pred_in": in_count, "pred_out": out_count,
            "ai_fps": ai_fps, "proc_fps": proc_fps}

def _write_summary():
    summary_path = RESULTS_DIR / "_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "key", "gate", "flight", "gt_in", "gt_out",
            "pred_in", "pred_out", "abs_err_in", "abs_err_out",
            "pct_err_in", "pct_err_out", "ai_fps", "proc_fps", "status",
        ])
        w.writeheader()
        for key, _ in CLIPS:
            csv_path = RESULTS_DIR / f"{key}_lrai_fps{TARGET_FPS}.csv"
            r = _summary_row(key, csv_path)
            gt_in, gt_out = GT.get(key, ("", ""))
            try:
                ae_in  = abs(int(r["pred_in"])  - gt_in)
                ae_out = abs(int(r["pred_out"]) - gt_out)
                pe_in  = (ae_in  / gt_in  * 100) if gt_in  else (100 if r["pred_in"]  else 0)
                pe_out = (ae_out / gt_out * 100) if gt_out else (100 if r["pred_out"] else 0)
            except (ValueError, TypeError):
                ae_in = ae_out = pe_in = pe_out = ""
            gate, flight = key.split("_")
            status = "done" if csv_path.exists() else "pending"
            w.writerow({"key": key, "gate": gate, "flight": flight,
                          "gt_in": gt_in, "gt_out": gt_out,
                          "pred_in": r["pred_in"], "pred_out": r["pred_out"],
                          "abs_err_in": ae_in, "abs_err_out": ae_out,
                          "pct_err_in": pe_in, "pct_err_out": pe_out,
                          "ai_fps": r["ai_fps"], "proc_fps": r["proc_fps"],
                          "status": status})
    print(f"[summary] wrote {summary_path}", flush=True)

t_audit_start = time.time()
for key, fname in CLIPS:
    out_csv = RESULTS_DIR / f"{key}_lrai_fps{TARGET_FPS}.csv"
    out_log = RESULTS_DIR / f"{key}_lrai_fps{TARGET_FPS}.log"
    if out_csv.exists() and out_csv.stat().st_size > 100:
        print(f"[audit] SKIP {key} — csv already present ({out_csv.stat().st_size} bytes)", flush=True)
        _write_summary()
        continue
    print(f"\n========== {key}  {fname} ==========", flush=True)
    t0 = time.time()
    with out_log.open("w") as logf:
        proc = subprocess.Popen([
            sys.executable, str(REPO_LOCAL / "scripts" / "lrai_isp_audit.py"),
            "--video", str(VIDEOS_DIR / fname),
            "--lines", str(LINES_DIR / f"{key}.json"),
            "--out-csv", str(out_csv),
            "--target-fps", str(TARGET_FPS),
            "--weights", str(WEIGHTS_LOCAL),
            "--prompt", PROMPT,
            # PT path until TRT decode engines are fixed (probe found zero-logit
            # regression after bf16 export switch). PT works end-to-end.
            "--path", "pt",
        ], stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line); logf.flush()
        proc.wait()
    elapsed = time.time() - t0
    print(f"[audit] {key} done in {elapsed/60:.1f} min  (rc={proc.returncode})", flush=True)
    _write_summary()

total = time.time() - t_audit_start
print(f"\n========== AUDIT COMPLETE: total {total/3600:.2f} h ==========", flush=True)
_write_summary()
print(f"[final] results in {RESULTS_DIR}")
