"""Evaluate the best checkpoint of a run on one split.

The **test** split is evaluated **once**, at the very end, on the final model: the first test
evaluation writes ``outputs/splits/TEST_EVALUATED.json`` and later attempts are refused.

Usage::

    uv run ipe-evaluate --run outputs/runs/tcn --split val
    uv run ipe-evaluate --run outputs/runs/<final_model> --split test     # once, at the end
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch

from ipe_prediction.training.dataset import ThermalDataset, make_dataloader, prepare_data
from ipe_prediction.training.engine import amp_dtype, evaluate, load_model
from ipe_prediction.training.metrics import constraint_report, per_family_metrics

logger = logging.getLogger("ipe_prediction.evaluate")


def evaluate_run(run_dir: Path, split_name: str) -> None:
    """Score ``best.pt`` of ``run_dir`` on ``split_name``; write metrics, predictions, summary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prep, cfg = load_model(run_dir, device=str(device))
    marker = Path(cfg.data.split_dir) / "TEST_EVALUATED.json"
    if split_name == "test" and marker.exists():
        raise RuntimeError(f"the test split was already evaluated: {marker.read_text()}")

    tensors, split = prepare_data(cfg.data)
    loader = make_dataloader(ThermalDataset(tensors, split[split_name]), cfg.train.eval_batch_size)
    res = evaluate(model, prep, loader, device, amp_dtype(device, cfg.train.amp))

    res["metrics"].to_csv(run_dir / f"metrics_{split_name}.csv")
    torch.save(
        {k: res[k] for k in ("pred", "target", "index")}
        | {"excitation": tensors.excitation[res["index"]], "names": tensors.excitation_names},
        run_dir / f"predictions_{split_name}.pt",
    )
    summary = {
        "split": split_name,
        "loss": res["loss"],
        "family": per_family_metrics(res["metrics"]).round(4).to_dict(orient="index"),
        "constraints": constraint_report(res["pred"]),
    }
    (run_dir / f"summary_{split_name}.json").write_text(json.dumps(summary, indent=2))
    logger.info("%s metrics:\n%s", split_name, res["metrics"].round(3).to_string())
    if split_name == "test":
        stamp = {"run": str(run_dir), "at": datetime.now().isoformat(timespec="seconds")}
        marker.write_text(json.dumps(stamp))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point (``uv run ipe-evaluate``)."""
    p = argparse.ArgumentParser(description="Evaluate a trained run on one split.")
    p.add_argument("--run", required=True, type=Path, help="run directory with best.pt")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_run(a.run, a.split)


if __name__ == "__main__":
    main()
