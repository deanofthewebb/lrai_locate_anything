#!/usr/bin/env python3
"""Multi-GPU TRT verification — produces an overlay image proving TRT runs
end-to-end across two GPUs on commodity hardware (e.g. learn02: 3080 10GB +
3080 Ti 12GB, where no single card fits all 3 LLM engines + vision + proj).

Splits the pipeline:
    GPU 0:  vision.engine + projector.engine + llm_decode_ar.engine + llm_decode.engine
    GPU 1:  llm_prefill.engine
The runner round-trips engine I/O through host buffers, so cross-device
data movement is automatic.

This script does NOT build engines — it expects them already built (the
STRONGLY_TYPED bf16 build from commit 7fd1449). It loads the PT model only
because LocateAnythingRunner needs tokenizer + processor + config; PT
weights can stay on CPU (or even at fp32) since the inference path is
TRT-only.

Usage:
    LRAI_WORKDIR=/mnt/ssd0/locany_test_lvl2 \\
    python scripts/lrai_trt_verify_multigpu.py \\
        --weights /mnt/ssd0/locany_test_lvl2/weights \\
        --image   /mnt/ssd0/locany_test_lvl2/cats.jpg \\
        --prompt  cats \\
        --out     /mnt/ssd0/locany_test_lvl2/trt_verify_overlay.jpg

LRAI_WORKDIR must point at the workspace whose engines/ subdir contains the
built .engine files. Pass --gpu-small / --gpu-big to override the default 0/1 split.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", default="cats")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-small", type=int, default=0,
                    help="GPU id for vision+proj+decode (3080 10GB on learn02)")
    ap.add_argument("--gpu-big", type=int, default=1,
                    help="GPU id for prefill (3080 Ti 12GB on learn02)")
    ap.add_argument("--mode", default="slow", choices=("slow", "hybrid"),
                    help="slow=AR only (skips decode_mtp); hybrid=full pipeline")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    # Production: TRT-only inference. Force PT model to CPU so it doesn't
    # consume GPU VRAM that the TRT engines need. The PT model still loads
    # (the runner reads tokenizer/processor/config from it) but won't sit
    # in CUDA memory. Must be set BEFORE importing lrai_locate_anything.
    os.environ.setdefault("LRAI_PT_CPU_ONLY", "1")

    from cuda.bindings import runtime as cudart
    n_dev = cudart.cudaGetDeviceCount()[1]
    print(f"[verify] CUDA devices: {n_dev}", file=sys.stderr)
    for i in range(n_dev):
        cudart.cudaSetDevice(i)
        props = cudart.cudaGetDeviceProperties(i)[1]
        free, total = cudart.cudaMemGetInfo()[1:]
        name = props.name.decode() if isinstance(props.name, bytes) else props.name
        print(f"[verify]   GPU {i}: {name}  free={free/1e9:.1f}/{total/1e9:.1f} GB",
              file=sys.stderr)

    from lrai_locate_anything.shims import install_transformers_shims
    install_transformers_shims(verbose=False)
    from lrai_locate_anything.orchestrator import LocateAnythingRunner

    print(f"[verify] weights dir: {args.weights}", file=sys.stderr)

    device_map = {
        "vision":      args.gpu_small,
        "projector":   args.gpu_small,
        "prefill":     args.gpu_big,
        "decode_ar":   args.gpu_small,
        "decode_mtp":  args.gpu_small,
    }
    print(f"[verify] engine placement:", file=sys.stderr)
    for k, v in device_map.items():
        print(f"[verify]   {k:10s} -> GPU {v}", file=sys.stderr)

    t0 = time.time()
    runner = LocateAnythingRunner.from_pretrained(local_dir=args.weights)
    print(f"[verify] runner constructed in {time.time()-t0:.1f}s "
          f"(PT model loaded; CPU residency)", file=sys.stderr)
    t0 = time.time()
    runner.load_engines(device_map=device_map)
    print(f"[verify] engines placed across GPUs in {time.time()-t0:.1f}s", file=sys.stderr)

    for i in range(n_dev):
        cudart.cudaSetDevice(i)
        free, total = cudart.cudaMemGetInfo()[1:]
        used_gb = (total - free) / 1e9
        print(f"[verify] GPU {i} VRAM used after engine load: {used_gb:.2f} GB",
              file=sys.stderr)

    image = Image.open(args.image).convert("RGB")
    print(f"[verify] image: {args.image.name}  size={image.size}", file=sys.stderr)

    full_prompt = (
        f"Locate all the instances that matches the following description: {args.prompt}."
    )
    print(f"[verify] running TRT inference (mode={args.mode}) ...", file=sys.stderr)
    t0 = time.time()
    boxes, text = runner.detect(
        image, full_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_mode=args.mode,
        path="trt",
    )
    dt = time.time() - t0
    print(f"[verify] TRT detect produced {len(boxes)} boxes in {dt:.2f}s",
          file=sys.stderr)
    print(f"[verify] RAW TEXT (first 400 chars):", file=sys.stderr)
    print(text[:400], file=sys.stderr)
    print(f"[verify] boxes (orig image coords):", file=sys.stderr)
    for (x1, y1, x2, y2) in boxes:
        print(f"           ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})", file=sys.stderr)

    canvas = image.copy()
    d = ImageDraw.Draw(canvas)
    for (x1, y1, x2, y2) in boxes:
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
    label = f"learn02 dual-GPU TRT bf16  |  {len(boxes)} boxes  |  {dt:.2f}s"
    d.text((10, 10), label, fill=(0, 255, 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(args.out), quality=92)
    print(f"[verify] saved overlay -> {args.out}", file=sys.stderr)
    print(f"[verify] EXIT_BOXES={len(boxes)}", file=sys.stderr)
    return 0 if len(boxes) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
