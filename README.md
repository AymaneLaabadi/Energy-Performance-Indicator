# IPE prediction — building thermal indicators from short transients

Predict six intrinsic thermal indicators of a building (`HLC`, `HLC_mit`, `HLC_ext`, `tau_b`,
`tau_n`, `tau_inf`) from 8 hours of indoor air temperature `Tint(t)` and heating power `U(t)`,
using 1,000,000 dynamic thermal simulations of `quentin_house` (coEnergy assessment).

**Start with [`notebooks/00_report.ipynb`](notebooks/00_report.ipynb)**: the executable report
(problem, data analysis, methodological choices, results, limits). The other notebooks hold the
detailed analyses it summarises.

## Main results (validation split, 100k simulations)

Final model: dilated residual TCN with attention pooling (186k parameters), trained on the 800k
training simulations (`final_tcn_mse`). Standardised MSE 0.734 (1.0 = predicting the training mean).

| Indicator | MAPE | P95 APE | R² |
|---|---|---|---|
| HLC | 8.3 % | 21.0 % | 0.42 |
| HLC_ext | 9.6 % | 24.2 % | 0.42 |
| HLC_mit | 15.9 % | 42.4 % | 0.27 |
| tau_n | 12.2 % | 29.6 % | 0.24 |
| tau_b | 14.9 % | 36.6 % | 0.11 |
| tau_inf | 17.9 % | 44.1 % | 0.09 |

The plateau is an **identifiability limit of 8 h tests**, not a modelling one: under heating the
curve is almost entirely determined by the building, but different buildings give near-identical
curves. The test measures very well an effective heat capacity `C_n = HLC x tau_n`
(R² 0.84) and much less how it splits into heat losses and time constants (notebooks 04, 06).

## Setup

```bash
uv sync
```

Download the `quentin_house/` folder (20 `batch_*.pt` files, ~3.7 GB) into the project root.
It is git-ignored. Outputs (`outputs/`: frozen split, runs, figures) are git-ignored too; the
split is recreated identically from its seed if missing.

## Reproduce

```bash
uv run ipe-train --name final_tcn_mse                         # final model (defaults of config.py)
uv run ipe-train --name final_tcn_gaussian_nll --loss gaussian_nll   # same model + variance head
uv run ipe-evaluate --run outputs/runs/final_tcn_mse --split val
uv run ipe-evaluate --run outputs/runs/final_tcn_mse --split test    # once, at the very end
```

About 1 min per epoch on an RTX 3070 (bf16), 30 epochs. Then run the notebooks.

| Notebook | Content |
|---|---|
| `00_report.ipynb` | **final report**: approach, choices and results |
| `01_eda.ipynb` | first EDA: data contract, targets, excitations, controllability |
| `02_architectures.ipynb` | train registered or notebook-defined models, compare runs |
| `03_target_structure.ipynb` | no repeated building, intrinsic dimension of the targets |
| `04_identifiability_uncertainty.ipynb` | final models: accuracy, error by protocol, calibrated intervals, canonical correlations |
| `05_informative_subset.ipynb` | training only on informative tests (no gain) |
| `06_advanced_eda.ipynb` | identical-input groups: building vs hidden state, information per protocol and duration, curve twins |
| `07_data_quality.ipynb` | integrity and quality checks (no cleaning needed) |

## Layout

```
src/ipe_prediction/
  util_assessement.py   loader provided with the assessment (kept verbatim)
  data.py               canonical target order, units, batch file listing
  training/             everything `ipe-train` / `ipe-evaluate` use
    config.py           DataConfig, TrainConfig, ExperimentConfig (defaults = final run)
    models/             one file per model, each with its own config dataclass
      __init__.py       registry: MODELS = {name: (Config, Model)}, build_model
      tcn_attn.py       dilated residual TCN + attention pooling (final model)
      cnn_gru.py        CNN + bidirectional GRU baseline
      model_a.py        residual conv blocks + average pooling -> bidirectional GRU
    splits.py           frozen, stratified, seeded 80/10/10 split over the 1M simulations
    dataset.py          compact in-memory tensors, input filter, Dataset and DataLoader
    preprocessing.py    model inputs and target scaling (fitted on the training split)
    losses.py           MSE or beta-NLL in the standardised target space
    metrics.py          per-target / per-family metrics, constraint checks
    engine.py           train / evaluate loops, lr schedule, mixed precision, load_model
    train.py            CLI + run(): training on train/val (never touches test)
    evaluate.py         CLI: evaluation of a run on one split; test allowed only once
  eda/                  analysis helpers used by the notebooks, never by training
    loading.py          raw-series loaders (load_dataset, load_targets_only)
    descriptors.py      descriptive statistics (excitation descriptors, PCA, downsampling)
    plotting.py         matplotlib style and figures
    identifiability.py  identical-input groups, ridge / kNN information measures, curve twins
    quality.py          raw-file scan, curve anomalies, group outliers, duplicates
scripts/
  kaggle_pack.py, kaggle_setup.py   train on a Kaggle GPU (see below)
notebooks/              see the table above
```

## Methodology in brief

* **Split**: 80/10/10 over the 1M simulations (seed 42), stratified on the excitation regime
  (never on / always on / PRBS x duty-cycle quintile x switching-speed tercile) and the HLC
  quartile; frozen in `outputs/splits/split_seed42.npz`. The test split is used once, at the end.
* **Inputs**: `ΔT(t) = Tint(t) - Tint(0)` (all simulations start from the same temperature), scaled
  by its training standard deviation, and the heater state (0/1, `U` is 0 or 7 kW), 480 steps.
* **Targets**: heat-loss coefficients standardised in W/K, time constants in log scale then
  standardised. The network predicts `HLC_mit`, `HLC_ext` and the three time constants; `HLC` is
  their sum, so the additivity constraint holds exactly.
* **Loss**: MSE in that standardised space, equal weights (or beta-NLL with a variance per target).
* **Training**: AdamW, linear warmup (1 epoch) then cosine decay 1e-3 -> 1e-5, batch 512, bf16
  autocast, gradient clipping at 1, best validation epoch kept.
* **Precise BatchNorm**: before each validation, the BatchNorm statistics are recomputed over 100
  training batches (weights frozen). The default moving-average statistics depend on the last
  batches seen (curves of very different amplitude) and made the validation loss jump from epoch
  to epoch; with exact statistics the score reflects the weights only.

Training options: `uv run ipe-train --help` (`--model`, `--epochs`, `--lr`, `--loss`, `--targets`,
`--max-train`, `--min-duty-cycle`, `--min-step-hours`, `--resume`, ...). To add a model, create
`models/<name>.py` with a config dataclass and an `nn.Module` built as
`Model(config, in_channels, n_outputs)`, and register it in `MODELS`.

## Branches

* `main`: the simple single-GPU pipeline.
* `model-a`: this work (Model A, targets selection, beta-NLL, input filter, analysis notebooks).
* `sysid-hybrid`: system-identification features + TCN hybrid (tested, no gain; notebook in report).
* `kaggle-ddp`: `main` + multi-GPU training (DistributedDataParallel, Kaggle T4 x2), learning-rate
  restarts and the overfitting run (below).

## Training on Kaggle (free GPU)

```bash
kaggle datasets init -p quentin_house && kaggle datasets create -p quentin_house   # data, once
python scripts/kaggle_pack.py --user <kaggle-username>
kaggle datasets create -p dist/kaggle/ipe-code      # then: kaggle datasets version -p ... -m "..."
```

In a Kaggle notebook with both datasets as inputs:

```python
import glob
%run {glob.glob("/kaggle/input/**/kaggle_setup.py", recursive=True)[0]}
!python -m ipe_prediction.training.train --name final_tcn_mse
```

### Both GPUs (T4 x2) and the overfitting run

`--ddp` starts one process per GPU (DistributedDataParallel). `--batch-size` stays the global
batch (256 per GPU with the default 512); every process scores the validation set, process 0 logs
and saves. A run can be resumed with or without `--ddp`.

Goal of this run: find how far the best model (`tcn_attn`) can **fit the 800k training
simulations**, generalisation being only measured, not sought. `final_tcn_mse` ends with a
training loss equal to its validation loss (0.739 vs 0.737): it underfits, so the run removes
every regulariser and trains for hundreds of epochs without ever annealing the learning rate:

```python
!python -m ipe_prediction.training.train --name overfit_tcn --ddp --epochs 300 \
    --lr-schedule cosine_restarts --lr 1e-3 --min-lr 1e-4 --restart-epochs 50 \
    --weight-decay 0 --val-subset 0 --train-eval-subset 100000 \
    --model-set dropout=0 mlp_hidden=256
# next Kaggle session (previous output as input): extend the same run
!python -m ipe_prediction.training.train --name overfit_tcn --ddp --resume --epochs 700
```

* `cosine_restarts` (SGDR): cosine cycles of 50 epochs from 1e-3 down to 1e-4, then back to 1e-3.
  The lr never goes below 10 % of its peak, and each restart kicks the weights out of a plateau;
  its shape does not depend on `--epochs`, so the run can be extended with `--resume --epochs N`
  (also true of `constant`). No early stopping, no reduce-on-plateau.
* `dropout=0`, `--weight-decay 0`: no regularisation; wider head (`mlp_hidden=256`). The width
  of the TCN (`channels=64`) can be raised with `--model-set channels=96`, at ~2x the time per epoch.
* `--val-subset 0`: the whole validation split (100k) each epoch; the test split (100k) stays
  untouched until `ipe-evaluate --split test`.
* `--train-eval-subset 100000`: 100k training simulations also scored each epoch in evaluation
  mode; the gap with the validation is the overfitting. Each epoch logs the loss, MAPE and **R2
  of every output** on the validation (and the training R2 on the next line); `history.csv`
  keeps them all, `summary.json` compares the last and the best epochs.
* A Kaggle session stops after 12 h: train in chunks (`--epochs` then `--resume --epochs`)
  sized from the seconds per epoch shown in the log.
