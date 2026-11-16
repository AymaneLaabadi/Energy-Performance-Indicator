"""Prepare a Kaggle notebook session (standard library only). First cell of the notebook::

    import glob
    %run {glob.glob("/kaggle/input/**/kaggle_setup.py", recursive=True)[0]}

Inputs: the data dataset, the ``ipe-code`` dataset (``scripts/kaggle_pack.py``) and, to resume
a run, the output of a previous version of the notebook. The script

* unzips the code into ``/tmp/ipe_proj`` (the working directory) and links the data there;
* links ``outputs`` to ``/kaggle/working/outputs``, the only thing kept as the notebook output;
* copies the frozen split and restores runs found in the inputs (most epochs wins);
* sets ``sys.path`` and ``PYTHONPATH``, so ``!python -m ipe_prediction.training.train`` works.

With the "GPU T4 x2" accelerator, add ``--ddp`` to train on both GPUs (one process each).
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

INPUT, WORKING, PROJ = Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp/ipe_proj")


def _walk() -> list[tuple[Path, list[str]]]:
    """``(folder, file names)`` for every folder under /kaggle/input (following symlinks)."""
    return [(Path(root), files) for root, _, files in os.walk(INPUT, followlinks=True)]


def _epochs(run_dir: Path) -> int:
    hist = run_dir / "history.csv"
    return len(hist.read_text().splitlines()) - 1 if hist.exists() else 0


def _install_code(folders: list[tuple[Path, list[str]]]) -> Path:
    """Copy the code to PROJ, from the extracted folder (Kaggle unzips uploads) or the zip."""
    shutil.rmtree(PROJ, ignore_errors=True)
    for folder, files in folders:
        if "pyproject.toml" in files and (folder / "src" / "ipe_prediction").is_dir():
            shutil.copytree(folder, PROJ, ignore=shutil.ignore_patterns("__pycache__"))
            return folder
    for folder, files in folders:
        if "ipe_code.zip" in files:
            with zipfile.ZipFile(folder / "ipe_code.zip") as zf:
                zf.extractall(PROJ)
            return folder / "ipe_code.zip"
    raise FileNotFoundError("code not found: add the ipe-code dataset as an input")


def setup() -> None:
    folders = _walk()
    data = next((f for f, files in folders if "batch_1.pt" in files), None)
    if data is None:
        found = "\n".join(f"  {f}" for f, _ in folders[:40])
        raise FileNotFoundError(
            f"no batch_1.pt under {INPUT}; add the data dataset. Found:\n{found}"
        )
    code = _install_code(folders)
    (PROJ / "quentin_house").symlink_to(data, target_is_directory=True)
    print(f"data {data}\ncode {code}")

    outputs = WORKING / "outputs"
    (outputs / "runs").mkdir(parents=True, exist_ok=True)
    for f in (PROJ / "outputs" / "splits").glob("*"):  # never overwrite (e.g. the test marker)
        (outputs / "splits").mkdir(exist_ok=True)
        if not (outputs / "splits" / f.name).exists():
            shutil.copy2(f, outputs / "splits" / f.name)
    shutil.rmtree(PROJ / "outputs", ignore_errors=True)
    (PROJ / "outputs").symlink_to(outputs, target_is_directory=True)

    runs = [f for f, files in folders if "last.pt" in files and f.parent.name == "runs"]
    for src in runs:
        dst = outputs / "runs" / src.name
        if _epochs(src) > _epochs(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"restored run {src.name} ({_epochs(src)} epochs)")

    sys.path.insert(0, str(PROJ / "src"))
    os.environ["PYTHONPATH"] = str(PROJ / "src")
    os.chdir(PROJ)

    import torch

    n_gpu = torch.cuda.device_count()
    gpu = f"{n_gpu} x {torch.cuda.get_device_name(0)}" if n_gpu else "none (enable a GPU)"
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | GPU {gpu}")
    ddp = " --ddp" if n_gpu > 1 else ""
    print(f"next: !python -m ipe_prediction.training.train --name <run> --epochs <n>{ddp}")


if __name__ == "__main__":
    setup()
