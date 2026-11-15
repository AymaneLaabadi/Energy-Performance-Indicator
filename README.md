# IPE prediction — building thermal indicators from short transients

Predict six intrinsic thermal indicators of a building (`HLC`, `HLC_mit`, `HLC_ext`, `tau_b`,
`tau_n`, `tau_inf`) from 8 hours of indoor air temperature `Tint(t)` and heating power `U(t)`,
using 1,000,000 dynamic thermal simulations of `quentin_house` (coEnergy assessment).

## Setup

```bash
uv sync
```

Download the `quentin_house/` folder (20 `batch_*.pt` files, ~3.7 GB) into the project root.
It is git-ignored.

## Layout

```
src/ipe_prediction/
  util_assessement.py   loader provided with the assessment (kept verbatim)     [shared]
  data.py               canonical target order, units, batch file listing        [shared]
  training/             everything `ipe-train` / `ipe-evaluate` use
    config.py           typed experiment configuration (data / model / train)
    splits.py           frozen, stratified, seeded 80/10/10 split over the 1M simulations
    dataset.py          compact in-memory tensors, prepare_data, Dataset and DataLoader
    preprocessing.py    input scale and target scaler (fitted on the training split)
    models.py           TCN + attention pooling (default), CNN-GRU baseline
    losses.py           MSE in the standardised target space
    metrics.py          per-target / per-family metrics, constraint checks
    engine.py           train_one_epoch / evaluate loops, checkpoint loading
    experiment.py       notebook API: train(name, Net, **settings), compare, plot, load_model
    train.py            CLI: training on train/val with early stopping (never touches test)
    evaluate.py         CLI: evaluation of a run on one split; test allowed only once
  eda/                  notebook-only helpers, never imported by training
    loading.py          raw-series loaders (load_dataset, load_targets_only)
    descriptors.py      descriptive statistics (excitation descriptors, PCA, downsampling)
    plotting.py         matplotlib style and figures
notebooks/
  01_eda.ipynb          exploratory data analysis
  02_architectures.ipynb  define a network in a cell, train it with the pipeline, compare runs
```

## Data split

All 1M simulations are split 80/10/10 (seed 42), stratified on the excitation regime
(never on / always on / PRBS x duty-cycle quintile x switching-speed tercile) and the HLC
quartile (68 strata). The indices are frozen in `outputs/splits/split_seed42.npz`, with a
representativeness report next to them.

## Train and evaluate

```bash
uv run ipe-train --name tcn                           # TCN + attention pooling (default --arch tcn_attn)
uv run ipe-train --name baseline --arch cnn_gru       # the CNN-GRU baseline
uv run ipe-train --name baseline --resume             # continue a stopped run from its last epoch
uv run ipe-train --name baseline --resume --epochs 800 # ... or extend it
uv run ipe-train --name quick --epochs 5 --max-train 50000 --no-amp
```

Training runs for `--epochs` epochs (no early stopping), with bf16 autocast (`--amp`, default on)
and batch size 512. Each epoch writes `last.pt` (model, optimizer, RNG state) and, when the
validation loss improves, `best.pt`. The shuffling order is derived from `(seed, epoch)`, so a
resumed run reproduces an uninterrupted one. With `--resume` only `--epochs` can be changed.

```bash
uv run ipe-evaluate --run outputs/runs/baseline --split val
```

The test split is evaluated **once**, at the end, on the final model
(`--split test`); a marker file then blocks any further test evaluation.
