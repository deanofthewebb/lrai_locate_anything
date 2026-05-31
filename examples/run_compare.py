"""Side-by-side multi-runtime comparison on your own clip.

Usage:
    python examples/run_compare.py custom.mp4 compare.mp4 "Detect luggage and people."
"""
import sys
from pathlib import Path
import json
from lrai_locate_anything import LocateAnythingRunner
from lrai_locate_anything.pipelines import run_compare

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Detect all luggage and people."

    runner = LocateAnythingRunner.from_pretrained(auto_export=True)
    metrics = run_compare(runner, in_path, out_path, prompt=prompt, max_frames=30)
    print()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
