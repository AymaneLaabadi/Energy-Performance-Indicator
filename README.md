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
    config.py           DataConfig, TrainConfig, ExperimentConfig
    models/             one file per model, each with its own config dataclass
      __init__.py       registry: MODELS = {name: (Config, Model)}, build_model
      cnn_gru.py        CNN + bidirectional GRU baseline
      tcn_attn.py       dilated residual TCN + attention pooling
    splits.py           frozen, stratified, seeded 80/10/10 split over the 1M simulations
    dataset.py          compact in-memory tensors, prepare_data, Dataset and DataLoader
    preprocessing.py    model inputs and target scaling (fitted on the training split)
    losses.py           MSE in the standardised target space
    metrics.py          per-target / per-family metrics, constraint checks
    engine.py           train / evaluate loops, lr schedule, mixed precision, load_model
    train.py            CLI + run(): training on train/val (never touches test)
    evaluate.py         CLI: evaluation of a run on one split; test allowed only once
  eda/                  notebook-only helpers, never imported by training
    loading.py          raw-series loaders (load_dataset, load_targets_only)
    descriptors.py      descriptive statistics (excitation descriptors, PCA, downsampling)
    plotting.py         matplotlib style and figures
scripts/
  kaggle_pack.py        laptop: pack code + split as a Kaggle dataset folder
  kaggle_setup.py       Kaggle: session setup (data link, outputs, run restore, import path)
notebooks/
  01_eda.ipynb          exploratory data analysis
  02_architectures.ipynb  train registered or notebook-defined models, compare runs
```

## Data split

All 1M simulations are split 80/10/10 (seed 42), stratified on the excitation regime
(never on / always on / PRBS x duty-cycle quintile x switching-speed tercile) and the HLC
quartile (68 strata). The indices are frozen in `outputs/splits/split_seed42.npz`, with a
representativeness report next to them.

## Train and evaluate

```bash
uv run ipe-train --name tcn                          # TCN + attention pooling (default model)
uv run ipe-train --name gru --model cnn_gru          # the CNN-GRU baseline
uv run ipe-train --name quick --epochs 3 --max-train 50000
uv run ipe-train --name tcn --resume                 # continue a stopped run
```

Options: `--epochs` (60), `--batch-size` (512), `--lr` (1e-3), `--max-train`, `--extra-features`
(5 input channels instead of 2), `--no-amp`. Model hyper-parameters are set in the model's config
dataclass (`models/<name>.py`), or passed as `model_config` from a notebook.

AdamW, linear warmup (1 epoch) then cosine decay to 1e-6 at the last epoch, so `--epochs` is the
real budget. Mixed precision: bf16 on Ampere+ GPUs (RTX 30xx), fp16 with gradient scaling on
T4 / P100 (Kaggle). Each epoch writes `history.csv`, `last.pt` (to resume) and, when the validation
loss improves, `best.pt`.

To add a model: create `models/<name>.py` with a config dataclass and an `nn.Module` built as
`Model(config, in_channels)` mapping `(B, C, L)` to `(B, 5)`, and register it in `MODELS`.

```bash
uv run ipe-evaluate --run outputs/runs/tcn --split val
```

The test split is evaluated **once**, at the end, on the final model (`--split test`); a marker
file then blocks any further test evaluation.

## Training on Kaggle (free GPU)

Upload the data once and the code after each change, as private datasets:

```bash
kaggle datasets init -p quentin_house     # then edit the id in quentin_house/dataset-metadata.json
kaggle datasets create -p quentin_house
python scripts/kaggle_pack.py --user <kaggle-username>
kaggle datasets create -p dist/kaggle/ipe-code                      # first time
kaggle datasets version -p dist/kaggle/ipe-code -m "update code"    # afterwards
```

In a Kaggle notebook (Accelerator: GPU; inputs: both datasets, plus the output of a previous
version of the notebook to resume a run):

```python
import glob
%run {glob.glob("/kaggle/input/**/kaggle_setup.py", recursive=True)[0]}
```
```python
!python -m ipe_prediction.training.train --name tcn_full --epochs 60
# next session: !python -m ipe_prediction.training.train --name tcn_full --resume
```

Training uses one GPU. Multi-GPU training (DDP, for the Kaggle "GPU T4 x2" accelerator) lives on
the `kaggle-ddp` branch.

Use "Save Version > Save & Run All" for long runs (up to 12 h, the browser can be closed); the
runs are in the version's Output (`outputs/runs/<name>/`).
