"""Per-frame detection on a video.

Usage:
    python examples/run_video.py input.mp4 output.mp4 "Detect people and luggage."
"""
import sys
from pathlib import Path
from lrai_locate_anything import LocateAnythingRunner, run_video

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Detect all objects. Return bounding boxes."

    # Use the first frame of the video to determine the baked engine resolution.
    import cv2
    cap = cv2.VideoCapture(str(in_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"could not read {in_path}")
        sys.exit(1)
    from PIL import Image
    sample_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sample_path = in_path.with_suffix(".sample.jpg")
    sample_pil.save(sample_path)

    runner = LocateAnythingRunner.from_pretrained(
        auto_export=True, sample_image=sample_path, sample_prompt=prompt,
    )
    metrics = run_video(runner, in_path, out_path, prompt=prompt, max_frames=60)
    print(f"\nProcessed {metrics['frames']} frames in {metrics['total_s']:.1f}s "
          f"({metrics['fps']:.2f} fps, avg {metrics['avg_ms_per_frame']:.0f} ms/frame, "
          f"P99 {metrics['p99_ms']:.0f} ms)")
    print(f"Output: {out_path}")

if __name__ == "__main__":
    main()
