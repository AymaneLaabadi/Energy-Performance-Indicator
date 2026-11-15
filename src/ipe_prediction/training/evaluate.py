"""Final evaluation of a trained run on one split.

The **test** split is meant to be evaluated **once**, at the very end of the project, on the
finally selected model. This script enforces it: the first test evaluation writes a marker file
(``outputs/splits/TEST_EVALUATED.json``) and any later attempt is refused.

Usage::

    uv run ipe-evaluate --run outputs/runs/baseline_cnn_gru --split val
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
from ipe_prediction.training.engine import evaluate, load_run
from ipe_prediction.training.losses import StandardizedMSELoss
from ipe_prediction.training.metrics import constraint_report, per_family_metrics

logger = logging.getLogger("ipe_prediction.evaluate")


def evaluate_run(run_dir: Path, split_name: str, device: str = "cuda") -> None:
    """Evaluate the best checkpoint of ``run_dir`` on ``split_name`` and save metrics/predictions."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(run_dir, str(dev))
    marker = Path(cfg.data.split_dir) / "TEST_EVALUATED.json"
    if split_name == "test" and marker.exists():
        raise RuntimeError(f"The test split was already evaluated: {marker.read_text()}")

    tensors, split = prepare_data(cfg.data)
    loader = make_dataloader(
        ThermalDataset(tensors, split[split_name]), cfg.train.eval_batch_size, shuffle=False
    )
    res = evaluate(model, loader, StandardizedMSELoss(model.target_scaler), dev, cfg.train.amp)
    res["metrics"].to_csv(run_dir / f"metrics_{split_name}.csv")
    idx = res["index"]
    torch.save(
        {
            "pred": res["pred"],
            "target": res["target"],
            "index": idx,
            "excitation": tensors.excitation[idx],
            "excitation_names": tensors.excitation_names,
        },
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
        marker.write_text(
            json.dumps({"run": str(run_dir), "at": datetime.now().isoformat(timespec="seconds")})
        )


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
