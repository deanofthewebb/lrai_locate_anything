"""Colab deploy: install lrai_locate_anything from GitHub using a PAT from Colab secrets.

Paste this whole file into a Colab cell, or:
    !curl -sL https://raw.githubusercontent.com/deanofthewebb/lrai_locate_anything/main/scripts/colab_deploy.py | python -

Reads GITHUB_PAT (or GITHUB_TOKEN) from Colab userdata. Falls back to anonymous
git+https if no PAT is set — works for the public repo but rate-limited and won't
authenticate if the repo becomes private.
"""
import os
import subprocess
import sys


def _sh(cmd: str) -> None:
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if r.returncode:
        raise SystemExit(r.returncode)


def main() -> None:
    pat = None
    pat_src = None
    try:
        from google.colab import userdata  # type: ignore
        for key in ("GITHUB_PAT", "GITHUB_TOKEN"):
            try:
                v = userdata.get(key)
                if v:
                    pat, pat_src = v, f"Colab secret {key}"
                    break
            except Exception:
                continue
    except ImportError:
        for key in ("GITHUB_PAT", "GITHUB_TOKEN"):
            v = os.environ.get(key)
            if v:
                pat, pat_src = v, f"env var {key}"
                break

    user, repo = "deanofthewebb", "lrai_locate_anything"
    if pat:
        print(f"Using {pat_src} for authenticated install.")
        # x-access-token works for both classic + fine-grained PATs.
        url = f"https://x-access-token:{pat}@github.com/{user}/{repo}.git"
    else:
        print("No GITHUB_PAT / GITHUB_TOKEN found — falling back to anonymous install.")
        url = f"https://github.com/{user}/{repo}.git"

    _sh(f"pip -q install --force-reinstall 'git+{url}'")

    print("\nSanity check:")
    _sh("python -c 'import lrai_locate_anything; print(lrai_locate_anything.__version__)'")
    print()
    print("Quick start:")
    print("    from lrai_locate_anything import LocateAnythingRunner, run_image")
    print("    runner = LocateAnythingRunner.from_pretrained(auto_export=True)")
    print("    boxes, img, txt = run_image(runner, 'photo.jpg', 'Detect all cats.')")


if __name__ == "__main__":
    main()
