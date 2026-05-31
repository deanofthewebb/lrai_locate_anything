"""Minimal single-image detection example.

Usage:
    python examples/run_image.py path/to/image.jpg "Detect all cats."
"""
import sys
from pathlib import Path
from lrai_locate_anything import LocateAnythingRunner, run_image

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img_path = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Detect all objects. Return bounding boxes."

    runner = LocateAnythingRunner.from_pretrained(
        auto_export=True,
        sample_image=img_path,
        sample_prompt=prompt,
    )
    boxes, annotated, raw = run_image(runner, img_path, prompt=prompt)
    print(f"Detected {len(boxes)} boxes:")
    for b in boxes:
        print(f"  ({b[0]:.0f},{b[1]:.0f}) - ({b[2]:.0f},{b[3]:.0f})")
    out = img_path.with_suffix(".boxed.jpg")
    annotated.save(out)
    print(f"Annotated image saved to {out}")
    print("\nRaw output:")
    print(raw[:400])

if __name__ == "__main__":
    main()
