"""Colab deploy: install lrai_locate_anything from GitHub, run a sanity check.

Paste this whole file into a Colab cell, or:
    !curl -sL https://raw.githubusercontent.com/deanofthewebb/lrai_locate_anything/main/scripts/colab_deploy.py | python -

The cell reads GITHUB_PAT from Colab userdata for private-repo access; if your repo
is public, the PAT is optional.
"""
import os
import subprocess
import sys


def _sh(cmd: str) -> None:
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if r.returncode:
        raise SystemExit(r.returncode)


def main():
    # Read PAT from Colab secrets if available.
    pat = None
    try:
        from google.colab import userdata
        for key in ("GITHUB_PAT", "GITHUB_TOKEN"):
            try:
                pat = userdata.get(key)
                if pat:
                    print(f"Using {key} from Colab secrets.")
                    break
            except Exception:
                continue
    except ImportError:
        pat = os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN")

    user = "deanofthewebb"
    repo = "lrai_locate_anything"
    if pat:
        url = f"https://{user}:{pat}@github.com/{user}/{repo}.git"
    else:
        url = f"https://github.com/{user}/{repo}.git"

    _sh(f"pip -q install --force-reinstall 'git+{url}'")

    # Quick sanity check
    print("\nSanity check:")
    _sh(f"python -c 'import lrai_locate_anything; print(lrai_locate_anything.__version__)'")
    print()
    print("Quick start:")
    print("    from lrai_locate_anything import LocateAnythingRunner, run_image")
    print("    runner = LocateAnythingRunner.from_pretrained(auto_export=True)")
    print("    boxes, img, txt = run_image(runner, 'photo.jpg', 'Detect all cats.')")


if __name__ == "__main__":
    main()
