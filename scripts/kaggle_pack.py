"""Pack the project code as a Kaggle dataset folder (run on the laptop, standard library only).

    python scripts/kaggle_pack.py --user <kaggle-username>

Writes ``dist/kaggle/ipe-code/`` (``ipe_code.zip`` with the code and the frozen split,
``kaggle_setup.py`` and ``dataset-metadata.json``) and prints the upload commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ("src", "notebooks", "pyproject.toml", "uv.lock", "README.md", "outputs/splits")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--user", required=True, help="Kaggle username")
    a = p.parse_args()

    out = ROOT / "dist" / "kaggle" / "ipe-code"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "ipe_code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for item in CONTENT:
            path = ROOT / item
            for f in [path] if path.is_file() else sorted(path.rglob("*")):
                if (
                    f.is_file()
                    and "__pycache__" not in f.parts
                    and ".ipynb_checkpoints" not in f.parts
                ):
                    zf.write(f, f.relative_to(ROOT).as_posix())
    shutil.copy2(ROOT / "scripts" / "kaggle_setup.py", out)
    meta = {
        "title": "IPE prediction code",
        "id": f"{a.user}/ipe-code",
        "licenses": [{"name": "unknown"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"packed {out.relative_to(ROOT)}; upload it (private by default) with:")
    print("  first time : kaggle datasets create -p dist/kaggle/ipe-code")
    print('  afterwards : kaggle datasets version -p dist/kaggle/ipe-code -m "update code"')


if __name__ == "__main__":
    main()
