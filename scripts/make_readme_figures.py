"""Build the slide-like summary figures shown in the README (``docs/figures/*.png``).

Usage (from the project root, after the runs and notebooks)::

    uv run python scripts/make_readme_figures.py

Curves and learning curves are read from the data (a few batch files) and from
``outputs/runs/``; the other numbers are the results computed in the notebooks (each one cites its
source), copied here so that the figures can be rebuilt without re-running every analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

from ipe_prediction.data import batch_files, targets_from_raw
from ipe_prediction.eda import plotting as P
from ipe_prediction.util_assessement import load_single_batch

OUT = Path("docs/figures")
RUNS = Path("outputs/runs")
HOURS = np.arange(480) / 60
N_SLIDES = 9
W, H = 16.0, 9.0  # slide size (inches)

BLUE, ORANGE, AQUA = P.SERIES[0], P.SERIES[1], P.SERIES[2]
PAGE, CARD, BORDER = "#f4f3ef", "#ffffff", "#e2e0d8"
BLUE_SOFT, INK, INK_2, MUTED = "#e9f1fc", P.INK, P.INK_2, P.MUTED
TC = P.TARGET_COLORS
FONT = ["Segoe UI", "DejaVu Sans"]


def style() -> None:
    P.set_style()
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT[0],
            "mathtext.it": f"{FONT[0]}:italic",
            "mathtext.bf": f"{FONT[0]}:bold",
            "figure.facecolor": PAGE,
            "savefig.facecolor": PAGE,
            "axes.facecolor": "none",
            "axes.edgecolor": "#cfcdc4",
            "axes.axisbelow": True,
            "axes.labelsize": 11,
            "axes.labelcolor": INK_2,
            "grid.color": "#ecebe6",
            "grid.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


# --------------------------------------------------------------------------- layout primitives
def fx(inches: float) -> float:
    return inches / W


def fy(inches: float) -> float:
    return inches / H


def rounded(fig: plt.Figure, rect, color: str, edge: str | None = None, radius: float = 0.14):
    """Rounded rectangle in figure coordinates, with a radius in inches."""
    x, y, w, h = rect
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor=edge or "none",
            linewidth=1 if edge else 0,
            zorder=0,
            boxstyle=f"round,pad=0,rounding_size={fx(radius)}",
            mutation_aspect=W / H,
        )
    )


def dot(fig: plt.Figure, cx: float, cy: float, r: float, color: str) -> None:
    """Circle of radius ``r`` inches centred at figure coordinates ``(cx, cy)``."""
    fig.add_artist(Ellipse((cx, cy), fx(2 * r), fy(2 * r), transform=fig.transFigure,
                           color=color, zorder=1))  # fmt: skip


def wrap(text: str, size: float, width_in: float) -> list[str]:
    """Greedy word wrap using the real width of the text in the slide font."""
    prop = FontProperties(family=FONT)
    lines, line = [], ""
    text = text.replace(" %", " %")  # keep a number with its unit
    for word in text.split(" "):
        trial = f"{line} {word}".strip()
        if TextPath((0, 0), trial, size=size, prop=prop).get_extents().width / 72 <= width_in:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def slide(number: int, kicker: str, title: str, takeaway: str) -> plt.Figure:
    """16:9 slide: section kicker, title, one-line takeaway, footer with a progress bar."""
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.text(0.045, 0.94, f"{number:02d}   {kicker.upper()}", fontsize=12, weight="semibold",
             color=BLUE, va="center")  # fmt: skip
    fig.text(0.045, 0.885, title, fontsize=30, weight="semibold", color=INK, va="center")
    fig.text(0.045, 0.825, takeaway, fontsize=15, color=INK_2, va="center")
    for k in range(N_SLIDES):
        rounded(fig, (0.045 + k * 0.021, 0.034, 0.017, fy(0.07)),
                BLUE if k < number else "#dcdad2", radius=0.035)  # fmt: skip
    fig.text(0.955, 0.04, "IPE prediction  ·  coEnergy assessment", fontsize=10, color=MUTED,
             ha="right", va="center")  # fmt: skip
    return fig


def card(fig: plt.Figure, rect, title: str | None = None, subtitle: str | None = None) -> float:
    """White card; returns the height (inches) used by its title block."""
    rounded(fig, rect, CARD, BORDER)
    x, y, _w, h = rect
    top, used = y + h - fy(0.28), 0.28
    if title:
        fig.text(x + fx(0.3), top, title, fontsize=14, weight="semibold", color=INK, va="top")
        used += 0.32
    if subtitle:
        fig.text(x + fx(0.3), y + h - fy(used + 0.04), subtitle, fontsize=11, color=MUTED,
                 va="top", linespacing=1.35)  # fmt: skip
        used += 0.04 + 0.25 * (subtitle.count("\n") + 1)
    return used


def chart(fig, rect, title, subtitle=None, left=0.7, right=0.35, bottom=0.6, gap=0.3):
    """Card with a title and an axes inside it (paddings in inches)."""
    used = card(fig, rect, title, subtitle)
    x, y, w, h = rect
    inner = [x + fx(left), y + fy(bottom), w - fx(left + right), h - fy(used + gap + bottom)]
    ax = fig.add_axes(inner)
    ax.set_facecolor("none")
    return ax


def text_card(fig, rect, title, items, numbered=False, subtitle=None, skip=0.0) -> None:
    """Card with items ``(head, body)``: a bold head line and a wrapped explanation.

    ``skip`` (inches) leaves room under the title for a figure drawn by the caller.
    """
    used = card(fig, rect, title, subtitle) + skip
    x, y, w, h = rect
    w_in = w * W
    indent = 0.62
    bodies = [wrap(body, 11.5, w_in - indent - 0.35) for _, body in items]
    line_in = 11.5 * 1.4 / 72
    needed = sum(0.27 + len(b) * line_in for b in bodies)
    free = h * H - used - 0.2 - 0.3 - needed
    step_gap = min(0.45, max(0.2, free / max(1, len(items))))
    y_in = h * H - used - 0.2  # from the bottom of the card
    for k, ((head, _), lines) in enumerate(zip(items, bodies, strict=True)):
        cx, cy = x + fx(0.42), y + fy(y_in - 0.11)
        if numbered:
            dot(fig, cx, cy, 0.14, BLUE_SOFT)
            fig.text(cx, cy, str(k + 1), fontsize=10.5, weight="semibold", color=BLUE,
                     ha="center", va="center")  # fmt: skip
        else:
            rounded(fig, (cx - fx(0.05), cy - fy(0.05), fx(0.1), fy(0.1)), BLUE, radius=0.025)
        fig.text(x + fx(indent), y + fy(y_in), head, fontsize=12.5, weight="semibold", color=INK,
                 va="top")  # fmt: skip
        y_in -= 0.27
        fig.text(x + fx(indent), y + fy(y_in), "\n".join(lines), fontsize=11.5, color=INK_2,
                 va="top", linespacing=1.4)  # fmt: skip
        y_in -= len(lines) * line_in + step_gap
    if y_in + step_gap < 0.2:
        print(f"  warning: text card {title!r} overflows by {0.2 - y_in - step_gap:.2f} in")


def stat(fig, rect, value, label, color=BLUE, size=36) -> None:
    """Stat tile: a big number and its meaning."""
    card(fig, rect)
    x, y, _w, h = rect
    fig.text(x + fx(0.3), y + h - fy(0.25), value, fontsize=size, weight="semibold", color=color,
             va="top")  # fmt: skip
    fig.text(x + fx(0.3), y + h - fy(0.32 + size / 72 * 1.25), label, fontsize=11.5,
             color=INK_2, va="top", linespacing=1.4)  # fmt: skip


def hbars(ax, labels, values, colors, fmt="{:.2f}", xmax=None, height=0.6) -> None:
    """Horizontal bars, first label on top, value at the bar end, no axis clutter."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=height)
    ax.set_yticks(y, labels, fontsize=11.5, color=INK_2)
    ax.grid(axis="y", visible=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    xmax = xmax or max(values) * 1.15
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.012 * xmax, yi, fmt.format(v), va="center", fontsize=11, color=INK,
                weight="semibold")  # fmt: skip


def end_label(ax, x, y, text, color, dy=0.0) -> None:
    """Direct label at the end of a line."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center",
                fontsize=11, color=color, weight="semibold", annotation_clip=False)  # fmt: skip


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- data
def load_examples(n_files: int = 4) -> dict[str, torch.Tensor]:
    """Temperature rise, heater state and targets of the first batch files."""
    dts, us, ys = [], [], []
    for path in batch_files("quentin_house")[:n_files]:
        raw = load_single_batch(path)
        t = raw.indoor_temperature.round(decimals=2).float()
        dts.append(t - t[:, :1])
        us.append((raw.indoor_U > 3500).float())
        ys.append(targets_from_raw(raw).float())
        del raw
    return {"dT": torch.cat(dts), "U": torch.cat(us), "Y": torch.cat(ys)}


# --------------------------------------------------------------------------- slides
TOP = 0.765  # top of the content area (figure fraction)
BOT = 0.085  # bottom of the content area


def slide_problem(d: dict[str, torch.Tensor]) -> None:
    fig = slide(
        1,
        "The problem",
        "Predicting a building's thermal indicators from an 8-hour test",
        "Heating power in, indoor temperature out: six intrinsic indicators to recover.",
    )
    duty = d["U"].mean(1)
    switches = d["U"].diff(dim=1).abs().sum(1)
    picks = {
        "never heated": (int((duty == 0).nonzero()[0]), ORANGE),
        "on/off (PRBS)": (
            int(((switches >= 2) & (switches <= 4) & (duty > 0.5)).nonzero()[0]),
            AQUA,
        ),
        "always heated": (int((duty == 1).nonzero()[0]), BLUE),
    }
    rect = (0.045, BOT, 0.43, TOP - BOT)
    card(fig, rect, "One test = 8 hours, 480 steps", "three examples from the data")
    x, y, w, h = rect
    ax_u = fig.add_axes([x + fx(1.45), y + h - fy(2.15), w - fx(1.45 + 1.65), fy(0.85)])
    ax_t = fig.add_axes([x + fx(1.45), y + fy(0.65), w - fx(1.45 + 1.65), h - fy(2.95 + 0.65)])
    for k, (label, (i, c)) in enumerate(picks.items()):
        ax_u.step(HOURS, d["U"][i].numpy() * 0.7 + k * 1.1, where="post", color=c, lw=2.2)
        ax_t.plot(HOURS, d["dT"][i].numpy(), color=c, lw=2.4)
        end_label(ax_t, 8, float(d["dT"][i, -1]), label, c)
    ax_u.set_yticks([0.35, 1.45, 2.55], [n for n in picks], fontsize=10.5)
    ax_u.set_xticks([])
    ax_u.grid(False)
    for s in ("left", "bottom"):
        ax_u.spines[s].set_visible(False)
    ax_u.set_xlim(0, 8)
    ax_t.set_xlim(0, 8)
    fig.text(x + fx(0.3), y + h - fy(1.12), "INPUT  ·  heater 0 / 7 kW", fontsize=10.5,
             color=BLUE, va="top", weight="semibold")  # fmt: skip
    fig.text(x + fx(0.3), y + h - fy(2.5), "OUTPUT  ·  indoor temperature rise ΔT (°C)",
             fontsize=10.5, color=BLUE, va="top", weight="semibold")  # fmt: skip
    ax_t.set_xlabel("time (h)")
    ax_t.set_yticks([0, 5, 10, 15])
    ax_t.spines["left"].set_visible(False)

    rect = (0.49, 0.4, 0.465, TOP - 0.4)
    card(fig, rect, "Six indicators to predict (five independent)")
    x, y, w, h = rect
    rows = [
        ("HLC", "W/K", "global heat loss coefficient"),
        ("HLC_mit", "W/K", "losses to adjoining spaces"),
        ("HLC_ext", "W/K", "losses to the outside  (HLC = mit + ext)"),
        ("tau_b", "h", "structural discharge, ~50 h"),
        ("tau_n", "h", "persistence of night cooling (8 h)"),
        ("tau_inf", "h", "mean residence time, ~31 h"),
    ]
    for k, (name, unit, meaning) in enumerate(rows):
        yy = y + h - fy(1.0 + k * 0.36)
        dot(fig, x + fx(0.38), yy, 0.06, TC[name])
        fig.text(x + fx(0.58), yy, name, fontsize=13, weight="semibold", color=INK, va="center")
        fig.text(x + fx(1.65), yy, unit, fontsize=11.5, color=MUTED, va="center")
        fig.text(x + fx(2.25), yy, meaning, fontsize=12, color=INK_2, va="center")
        if k == 2:
            fig.add_artist(plt.Line2D([x + fx(0.3), x + w - fx(0.3)], [yy - fy(0.18)] * 2,
                                      color=BORDER, lw=1))  # fmt: skip

    rect = (0.49, BOT, 0.465, 0.3)
    card(fig, rect, "First-order physics")
    x, y, w, h = rect
    fig.text(x + fx(0.3), y + h - fy(0.95), r"$C\,\frac{dT}{dt} = U - H\,(T - T_{out})$",
             fontsize=19, color=INK, va="center")  # fmt: skip
    notes = [
        "Gain 1/H: needs heating to be seen",
        "Time constant C/H: tens of hours, glimpsed in 8 h",
        "Walls not at equilibrium at t = 0: a hidden state",
    ]
    for k, note in enumerate(notes):
        yy = y + h - fy(1.6 + k * 0.33)
        rounded(fig, (x + fx(0.32), yy - fy(0.045), fx(0.09), fy(0.09)), BLUE, radius=0.02)
        fig.text(x + fx(0.55), yy, note, fontsize=12, color=INK_2, va="center")
    save(fig, "01_problem.png")


def slide_data() -> None:
    fig = slide(
        2,
        "The data",
        "What the 1,000,000 simulations really contain",
        "Clean, noise-free simulator output, but far fewer distinct experiments than it seems.",
    )
    tiles = [
        ("1,000,000", "simulations of quentin_house\n480 steps of 1 min = 8 h", BLUE),
        ("2,798", "distinct heating sequences\none PRBS seed for all", BLUE),
        ("99.99 %", "of validation inputs also\nappear in training", ORANGE),
        ("0", "missing values, outliers or\nconstraint violations", AQUA),
    ]
    tw, gap = (0.91 - 3 * 0.015) / 4, 0.015
    for k, (value, label, c) in enumerate(tiles):
        stat(fig, (0.045 + k * (tw + gap), 0.56, tw, TOP - 0.56), value, label, c)

    ax = chart(fig, (0.045, BOT, 0.43, 0.455), "Heating regime", "share of the simulations",
               left=1.55, right=0.6)  # fmt: skip
    hbars(ax, ["on/off (PRBS)", "never heated", "always heated"], [90.7, 8.4, 0.9],
          [BLUE, ORANGE, BLUE], "{:.1f} %", xmax=105)  # fmt: skip
    ax.set_xticks([])
    text_card(
        fig,
        (0.49, BOT, 0.465, 0.455),
        "What it implies",
        [
            ("No cleaning needed", ("U is exactly 0 or 7 kW, every test starts at the same "
             "temperature, all physical constraints hold.")),
            ("Unique buildings, linked targets", ("the three taus move together, and against "
             "the losses (tau ≈ C / H).")),
            ("A built-in controlled experiment", ("identical inputs isolate the building from "
             "the hidden initial state (slide 6).")),
        ],
    )  # fmt: skip
    save(fig, "02_data.png")


def slide_approach() -> None:
    fig = slide(
        3,
        "Our approach",
        "One simple, physically consistent pipeline",
        "Raw curves in, the six indicators out; every choice was tested against alternatives.",
    )
    steps = [
        ("Inputs", "ΔT(t) = Tint − Tint(0)\nheater on / off\n480 steps"),
        ("Dilated TCN", "7 residual blocks\nreceptive field = 8 h\n186k parameters"),
        ("Attention pooling", "learns which part\nof the test matters"),
        ("5 outputs", "HLC_mit, HLC_ext\nlog tau_b, tau_n, tau_inf\nstandardised"),
        ("6 indicators", "HLC = HLC_mit + HLC_ext\nexact by construction"),
    ]
    gap = 0.03
    bw = (0.91 - 4 * gap) / 5
    y0, bh = 0.5, TOP - 0.5
    for k, (head, body) in enumerate(steps):
        x = 0.045 + k * (bw + gap)
        model = k in (1, 2)
        rounded(fig, (x, y0, bw, bh), BLUE_SOFT if model else CARD, None if model else BORDER)
        dot(fig, x + fx(0.42), y0 + bh - fy(0.45), 0.17, BLUE if model else BLUE_SOFT)
        fig.text(x + fx(0.42), y0 + bh - fy(0.45), str(k + 1), fontsize=12, weight="semibold",
                 color="white" if model else BLUE, ha="center", va="center")  # fmt: skip
        fig.text(x + fx(0.3), y0 + bh - fy(0.95), head, fontsize=15, weight="semibold",
                 color=INK, va="top")  # fmt: skip
        fig.text(x + fx(0.3), y0 + bh - fy(1.45), body, fontsize=12, color=INK_2, va="top",
                 linespacing=1.5)  # fmt: skip
        if k < len(steps) - 1:
            fig.add_artist(plt.Line2D([x + bw + 0.006, x + bw + gap - 0.006], [y0 + bh / 2] * 2,
                                      color=MUTED, lw=1.5, marker=">", markevery=[1], ms=6))  # fmt: skip
    text_card(
        fig,
        (0.045, BOT, 0.445, 0.395),
        "Data and evaluation",
        [
            ("Stratified, frozen split", ("80 / 10 / 10 on regime × duty cycle × switching speed "
             "× HLC quartile; the test split is scored once.")),
            ("Standardised targets", "losses in W/K, taus in log; MSE with equal weights."),
            ("Physical metrics", "MAPE, P95 error (the SSRN criterion) and R² per indicator."),
        ],
    )  # fmt: skip
    text_card(
        fig,
        (0.51, BOT, 0.445, 0.395),
        "Training",
        [
            ("Optimiser", ("AdamW, warmup then cosine 1e-3 → 1e-5, batch 512, bf16, 30 epochs "
             "on 800k simulations (~1 min / epoch, RTX 3070).")),
            ("Precise BatchNorm", "statistics recomputed exactly before every validation."),
            ("Uncertainty", "β-NLL variance head, then conformal 95 % intervals."),
        ],
    )  # fmt: skip
    save(fig, "03_approach.png")


def slide_results() -> None:
    fig = slide(
        4,
        "Results",
        "What the final model achieves",
        "Validation split, 100k simulations: HLC within 8 % on average; the slow time constants "
        "barely beat the mean.",
    )
    # notebook 00 (section 5): best checkpoint scored on the whole validation split
    m = pd.DataFrame(
        {
            "R2": [0.419, 0.416, 0.265, 0.244, 0.114, 0.085],
            "MAPE_%": [8.3, 9.6, 15.9, 12.2, 14.9, 17.9],
            "P95_APE_%": [21.0, 24.2, 42.4, 29.6, 36.6, 44.1],
        },
        index=["HLC", "HLC_ext", "HLC_mit", "tau_n", "tau_b", "tau_inf"],
    )
    order = list(m.index)
    colors = [TC[n] for n in order]

    ax = chart(fig, (0.045, BOT, 0.33, TOP - BOT), "R² per indicator",
               "1 = perfect, 0 = predicting the mean", left=1.0, right=0.55,
               bottom=0.95)  # fmt: skip
    fig.text(0.045 + fx(0.3), BOT + fy(0.3), "C_n = HLC × tau_n: effective heat capacity",
             fontsize=10.5, color=MUTED)  # fmt: skip
    hbars(ax, [*order, "C_n"], [*m["R2"], 0.84], [*colors, INK], xmax=1.0)
    ax.axhline(0.5, color=BORDER, lw=1, ls="--")
    ax.set_xticks([0, 0.5, 1.0])

    ax = chart(fig, (0.39, BOT, 0.33, TOP - BOT), "Absolute percentage error",
               "%  ·  mean (MAPE) dark, 95th percentile light", left=1.0, right=0.55,
               bottom=0.95)  # fmt: skip
    y = np.arange(len(order))[::-1]
    ax.barh(y, m["P95_APE_%"], color=colors, alpha=0.25, height=0.6)
    ax.barh(y, m["MAPE_%"], color=colors, height=0.6)
    for yi, a, b in zip(y, m["MAPE_%"], m["P95_APE_%"], strict=True):
        ax.text(a + 0.6, yi, f"{a:.1f}", va="center", fontsize=11, color=INK, weight="semibold")
        ax.text(b + 0.6, yi, f"{b:.0f}", va="center", fontsize=11, color=MUTED)
    ax.set_yticks(y, order, fontsize=11.5, color=INK_2)
    ax.grid(axis="y", visible=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40])

    tiles = [
        ("8.3 %", "mean error on HLC\n95 % of buildings within 21 %", BLUE),
        ("0.734", "standardised MSE\n1.0 = predicting the mean", INK),
        ("±20 %", "calibrated 95 % interval\non HLC (conformal)", AQUA),
    ]
    th = (TOP - BOT - 2 * 0.02) / 3
    for k, (v, label, c) in enumerate(tiles):
        stat(fig, (0.735, TOP - (k + 1) * th - k * 0.02, 0.22, th), v, label, c, size=32)
    save(fig, "04_results.png")


def slide_plateau() -> None:
    fig = slide(
        5,
        "Lack of fit",
        "The models do not fit the data, and it is not the model's fault",
        "No train / validation gap, the same level for every variant, real labels memorised "
        "like noise.",
    )
    h = pd.read_csv(RUNS / "final_tcn_mse/history.csv")
    pw, gap = (0.91 - 2 * 0.015) / 3, 0.015
    ax = chart(fig, (0.045, BOT, pw, TOP - BOT), "Final model, 30 epochs",
               "standardised MSE (1 = predicting the mean)", left=0.75, right=1.3)  # fmt: skip
    ax.fill_between(h["epoch"], h["val_loss"], h["train_loss"], color=BLUE, alpha=0.08, lw=0)
    ax.plot(h["epoch"], h["train_loss"], color=BLUE, lw=2.4)
    ax.plot(h["epoch"], h["val_loss"], color=ORANGE, lw=2.4)
    end_label(ax, 30, h["train_loss"].iloc[-1], "training", BLUE, dy=8)
    end_label(ax, 30, h["val_loss"].iloc[-1], "validation", ORANGE, dy=-8)
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1, 0.99, "predicting the mean", va="top", fontsize=10.5, color=MUTED)
    ax.set(ylim=(0.68, 1.02), xlabel="epoch", xlim=(1, 30))
    ax.annotate("no gap: underfitting,\nnot overfitting", (22, 0.745), xytext=(12, 0.86),
                fontsize=11.5, color=INK, ha="center", linespacing=1.3,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1})  # fmt: skip

    # report section 4: pairs of runs on the same data and budget
    pairs = [
        ("TCN 53k → 186k\nparameters", 0.81, 0.80),
        ("+ system-id features\n(Laguerre, ARX)", 0.811, 0.821),
        ("informative tests\nonly (31 %)", 0.708, 0.710),
    ]
    ax = chart(fig, (0.045 + pw + gap, BOT, pw, TOP - BOT), "Changing the model changes nothing",
               "validation loss  ·  blue: baseline, orange: variant", left=1.85, right=0.4)  # fmt: skip
    for k, (_, a, b) in enumerate(pairs):
        yy = len(pairs) - 1 - k
        ax.plot([a, b], [yy, yy], color="#d5d3cb", lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [yy], s=130, color=BLUE, zorder=2, edgecolor="white", lw=1.5)
        ax.scatter([b], [yy], s=130, color=ORANGE, zorder=2, edgecolor="white", lw=1.5)
        ax.annotate(f"{b - a:+.3f}", (max(a, b), yy), xytext=(14, 0), textcoords="offset points",
                    va="center", fontsize=11, color=INK, weight="semibold")  # fmt: skip
    ax.set_yticks(range(len(pairs))[::-1], [p[0] for p in pairs], fontsize=11, color=INK_2)
    ax.set(xlim=(0.69, 0.85), ylim=(-0.6, 2.6))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    # report section 4: memorisation of 1,024 samples
    epochs = [10, 50, 200]
    memo = [
        ("real labels", [0.748, 0.539, 0.337], BLUE, -8),
        ("shuffled labels", [0.954, 0.762, 0.499], ORANGE, 8),
        ("label computed\nfrom the curve", [0.003, 6e-4, 3e-4], AQUA, 0),
    ]
    ax = chart(fig, (0.045 + 2 * (pw + gap), BOT, pw, TOP - BOT), "Memorising 1,024 samples",
               "training loss (log scale), HLC_mit", left=0.75, right=1.55)  # fmt: skip
    for label, vals, c, dy in memo:
        ax.plot(epochs, vals, color=c, lw=2.4, marker="o", ms=7, mec="white", mew=1.5)
        end_label(ax, 200, vals[-1], label, c, dy=dy)
    ax.set(yscale="log", xlabel="epoch", xlim=(0, 210))
    ax.text(105, 0.06, "real labels are learned\nalmost like pure noise", ha="center",
            fontsize=11.5, color=INK, linespacing=1.3)  # fmt: skip
    save(fig, "05_plateau.png")


def slide_identifiability(d: dict[str, torch.Tensor]) -> None:
    fig = slide(
        6,
        "Why",
        "Different buildings produce the same curve",
        "Under heating the curve is set by the building, but the map building → curve is "
        "many-to-one.",
    )
    # an example pair of "curve twins": same input (always heated), closest curves
    on = (d["U"].mean(1) == 1).nonzero().flatten()
    c = d["dT"][on][:, ::5]
    dist = torch.cdist(c, c)
    dist.fill_diagonal_(float("inf"))
    hlc = d["Y"][on, 0]
    close = dist < torch.quantile(dist.min(1).values, 0.2)
    score = torch.where(close, (hlc[:, None] - hlc[None, :]).abs(), torch.zeros(()))
    i, j = np.unravel_index(int(score.argmax()), score.shape)
    a, b = sorted((int(on[i]), int(on[j])), key=lambda k: float(d["Y"][k, 0]))
    ax = chart(fig, (0.045, BOT, 0.33, TOP - BOT), "Two buildings, same heating",
               "indoor rise ΔT (°C), always heated", left=0.7, right=0.4, bottom=2.0)  # fmt: skip
    ax.plot(HOURS, d["dT"][a].numpy(), color=BLUE, lw=5, alpha=0.35)
    ax.plot(HOURS, d["dT"][b].numpy(), color=ORANGE, lw=2, ls=(0, (5, 3)))
    ax.set(xlabel="time (h)", xlim=(0, 8))
    x, y, w, _ = (0.045, BOT, 0.33, 0)
    cols = [(a, BLUE, "building A"), (b, ORANGE, "building B")]
    for k, (idx, col, name) in enumerate(cols):
        yy = d["Y"][idx]
        xx = x + fx(0.3) + k * w / 2
        fig.text(xx, y + fy(1.25), name, fontsize=11, color=col, weight="semibold")
        fig.text(xx, y + fy(0.85), f"HLC   {yy[0]:.0f} W/K", fontsize=13, color=INK,
                 weight="semibold")  # fmt: skip
        fig.text(xx, y + fy(0.5), f"tau_n  {yy[4]:.1f} h", fontsize=13, color=INK,
                 weight="semibold")  # fmt: skip
        fig.text(xx, y + fy(0.2), f"C_n  {yy[0] * yy[4] / 1000:.1f} kWh/K", fontsize=11,
                 color=MUTED)  # fmt: skip

    # notebook 06: median |Δ target| of curve twins / random pairs, always heated
    names = ["HLC", "HLC_mit", "HLC_ext", "tau_n", "tau_b", "tau_inf", "C_n"]
    ratio = [0.76, 0.86, 0.89, 0.91, 0.96, 0.97, 0.35]
    ax = chart(fig, (0.39, BOT, 0.27, TOP - BOT), "Curve twins differ like strangers",
               "target gap of twins / random pairs\n1 = the curve says nothing", left=0.95,
               right=0.45)  # fmt: skip
    hbars(ax, names, ratio, [*(TC[n] for n in names[:-1]), INK], xmax=1.12)
    ax.axvline(1, color="#c9c7bf", lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks([0, 0.5, 1])

    text_card(
        fig,
        (0.675, BOT, 0.28, TOP - BOT),
        "What the test measures",
        [
            ("The building sets heated curves", ("with the input fixed: 94 % of the variance, "
             "only 20 % without heating.")),
            ("One direction is seen", ("canonical correlations network ↔ targets: 0.97, 0.66, "
             "0.52, 0.15, 0.03.")),
            ("That direction is C_n", ("= HLC × tau_n, the heat stored over a few hours "
             "(R² 0.84).")),
            ("The model is at the limit", "HLC R² 0.42 vs model-free bounds 0.38–0.51."),
        ],
    )  # fmt: skip
    save(fig, "06_identifiability.png")


def slide_excitation() -> None:
    fig = slide(
        7,
        "Best excitation",
        "Which heating protocol is the most informative?",
        "Heat, with a high duty cycle and long on/off steps; test longer than 8 h if HLC and "
        "the time constants matter.",
    )
    # notebook 04: standardised MSE on HLC, PRBS tests, duty-cycle quintile x min switching time
    hlc = np.array([
        [0.608, 0.560, 0.628],
        [0.599, 0.578, 0.639],
        [0.584, 0.570, 0.536],
        [0.586, 0.529, 0.515],
        [0.555, 0.517, 0.518],
    ])  # fmt: skip
    duty = ["0–0.33", "0.33–0.48", "0.48–0.60", "0.60–0.68", "0.68–1"]
    steps = ["< 0.6 h", "0.6–1.3 h", "1.3–2 h"]
    pw, gap = (0.91 - 2 * 0.015) / 3, 0.015
    ax = chart(fig, (0.045, BOT, pw, TOP - BOT), "HLC error on on/off tests",
               "standardised MSE, lower is better\noutlined: the best settings", left=1.3, right=0.35, bottom=0.8)  # fmt: skip
    grid = hlc[::-1]
    ax.imshow(grid, cmap=P.SEQUENTIAL.reversed(), aspect="auto", vmin=0.49, vmax=0.66)
    ax.set_xticks(range(3), steps, fontsize=10.5)
    ax.set_yticks(range(5), duty[::-1], fontsize=10.5)
    ax.set_xlabel("minimum on/off step")
    ax.set_ylabel("duty cycle (share of time heated)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for (r, k), v in np.ndenumerate(grid):
        ax.text(k, r, f"{v:.2f}", ha="center", va="center", fontsize=12, weight="semibold",
                color="white" if v < 0.565 else INK)  # fmt: skip
        if v <= 0.53:
            ax.add_patch(Rectangle((k - 0.46, r - 0.44), 0.92, 0.88, fill=False, ec=ORANGE,
                                   lw=2.5))  # fmt: skip
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=3)
    ax.tick_params(which="minor", length=0)

    # notebook 06: ridge R2 (lower bound of the information) per protocol
    ax = chart(fig, (0.045 + pw + gap, BOT, pw, TOP - BOT), "Information on HLC per protocol",
               "R² read from the curve, model-free", left=1.55, right=0.45)  # fmt: skip
    hbars(ax, ["never heated", "on/off, median", "always heated", "best on/off\nduty 0.55, 1.3 h"],
          [0.126, 0.384, 0.428, 0.508], ["#d5d3cb", BLUE, BLUE, ORANGE], xmax=0.6)  # fmt: skip
    ax.set_xticks([0, 0.2, 0.4, 0.6])

    # notebook 06: information vs test duration (best PRBS protocol)
    hours = [0.5, 1, 2, 3, 4, 6, 8]
    curves = [
        ("C_n", [0.060, 0.748, 0.832, 0.857, 0.861, 0.861, 0.864], INK, 0),
        ("HLC", [0.013, 0.275, 0.304, 0.303, 0.296, 0.319, 0.429], TC["HLC"], 0),
        ("HLC_ext", [0.001, 0.080, 0.077, 0.057, 0.057, 0.095, 0.324], TC["HLC_ext"], 3),
        ("tau_n", [0.009, 0.111, 0.123, 0.125, 0.123, 0.136, 0.232], TC["tau_n"], -3),
    ]
    ax = chart(fig, (0.045 + 2 * (pw + gap), BOT, pw, TOP - BOT), "Would a longer test help?",
               "R² from the first hours (best on/off protocol)", left=0.7, right=1.15)  # fmt: skip
    ax.axvspan(2, 3, color=BLUE_SOFT, lw=0)
    for label, vals, c, dy in curves:
        ax.plot(hours, vals, color=c, lw=2.4, marker="o", ms=6, mec="white", mew=1.2)
        end_label(ax, 8, vals[-1], label, c, dy=dy)
    ax.set(xlabel="test duration (h)", ylim=(0, 1), xlim=(0, 8.3))
    ax.text(2.5, 0.96, "C_n\nsaturated", ha="center", va="top", fontsize=10.5, color=BLUE,
            linespacing=1.2)  # fmt: skip
    ax.annotate("HLC still\nrising at 8 h", (8, 0.43), xytext=(5.3, 0.6), fontsize=10.5,
                color=INK, ha="center", linespacing=1.2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1})  # fmt: skip
    save(fig, "07_excitation.png")


def best_sequence(d: dict[str, torch.Tensor]) -> int:
    """A real heating sequence of the recommended kind: duty 55-65 %, every step >= 1.3 h."""
    u = d["U"]
    duty = u.mean(1)
    for i in ((duty >= 0.55) & (duty <= 0.65)).nonzero().flatten().tolist():
        change = (u[i].diff() != 0).nonzero().flatten() + 1
        edges = torch.cat([torch.tensor([0]), change, torch.tensor([u.shape[1]])])
        steps = edges.diff()
        if len(steps) >= 4 and int(steps.min()) >= 80:
            return i
    return int(((duty >= 0.55) & (duty <= 0.65)).nonzero()[0])


def slide_paper(d: dict[str, torch.Tensor]) -> None:
    fig = slide(
        8,
        "Best excitation and the paper",
        "The excitation we recommend, and how it relates to the paper",
        "Nassif, Guernouti, Stabat & Cohen (SSRN 6928008): the same idea, extended from HLC to "
        "six indicators.",
    )
    # --- left: the recommended protocol, on a real sequence from the data
    rect = (0.045, BOT, 0.3, TOP - BOT)
    i = best_sequence(d)
    text_card(
        fig,
        rect,
        "The excitation we recommend",
        [
            ("Always heat", "no heating, no HLC: R² 0.13 vs 0.43 heated."),
            ("Heat 55–65 % of the time", ("a high duty cycle injects more energy: the HLC "
             "error drops from 0.61 to 0.52.")),
            ("On/off steps of 1.3–4 h", "slow switching reaches the walls; avoid < 40 min."),
            ("Test longer than 8 h", ("HLC and the taus still gain information at 8 h; "
             "C_n is saturated after 2–3 h.")),
        ],
        numbered=True,
        subtitle="a real sequence of this kind (duty "
        f"{float(d['U'][i].mean()):.0%}, steps ≥ 1.3 h)",
        skip=1.45,
    )  # fmt: skip
    x, y, w, h = rect
    ax = fig.add_axes([x + fx(0.85), y + h - fy(1.95), w - fx(1.15), fy(0.8)])
    ax.fill_between(HOURS, d["U"][i].numpy(), step="post", color=BLUE, alpha=0.18, lw=0)
    ax.step(HOURS, d["U"][i].numpy(), where="post", color=BLUE, lw=2)
    ax.set(xlim=(0, 8), ylim=(-0.08, 1.25))
    ax.set_yticks([0, 1], ["off", "7 kW"], fontsize=10)
    ax.set_xticks([0, 2, 4, 6, 8], ["0", "2", "4", "6", "8 h"], fontsize=10)
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    # --- middle: the paper's on-site tests (Table 5, net heating power) and robustness (§3.4)
    rect = (0.36, BOT, 0.24, TOP - BOT)
    ax = chart(fig, rect, "In the paper: within 2 W/K",
               "HAMEX test cell, predicted HLC vs the\n14-day co-heating reference (Table 5)",
               left=1.15, right=0.45, bottom=1.25)  # fmt: skip
    tests = ["6 h PRBS", "8 h sine", "12 h sine", "24 h sine"]
    pred = [21, 22, 23, 23]
    yy = np.arange(len(tests))[::-1]
    ax.axvspan(21, 25, ymax=0.86, color=BLUE_SOFT, lw=0)
    ax.axvline(23, ymax=0.86, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.scatter(pred, yy, s=150, color=[ORANGE, BLUE, BLUE, BLUE], edgecolor="white", lw=1.5,
               zorder=3)  # fmt: skip
    for v, yi in zip(pred, yy, strict=True):
        left = v < 23
        ax.annotate(f"{v} W/K", (v, yi), xytext=(-13 if left else 13, 0),
                    textcoords="offset points", va="center", ha="right" if left else "left",
                    fontsize=10.5, color=INK, weight="semibold")  # fmt: skip
    ax.text(23, 3.6, "reference 23 W/K, ±2 W/K band", ha="center", va="bottom", fontsize=10,
            color=MUTED)  # fmt: skip
    ax.set_yticks(yy, tests, fontsize=11.5, color=INK_2)
    ax.set(xlim=(17.2, 26.3), ylim=(-0.5, 3.9))
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.text(rect[0] + fx(0.3), BOT + fy(0.78), "1,600 simulated campaigns: longer windows",
             fontsize=10.5, color=INK_2)  # fmt: skip
    fig.text(rect[0] + fx(0.3), BOT + fy(0.55), "give more stable HLC (§3.4), as on our data.",
             fontsize=10.5, color=INK_2)  # fmt: skip

    # --- right: paper vs this work
    rect = (0.615, BOT, 0.34, TOP - BOT)
    used = card(fig, rect, "The paper and this work")
    x, y, w, h = rect
    col = [x + fx(0.3), x + fx(1.5), x + fx(3.42)]
    widths = [1.1, 1.75, 1.8]
    yy = y + h - fy(used + 0.2)
    for c, head in zip(col[1:], ("Paper", "This work"), strict=True):
        fig.text(c, yy, head, fontsize=11.5, weight="semibold", color=BLUE, va="top")
    yy -= fy(0.38)
    rows = [
        ("Targets", "HLC only; other indicators listed as future work",
         "HLC, HLC_mit, HLC_ext and 3 time constants"),
        ("Simulations", "test-cell population (±60 %), noise, random initial state",
         "1M simulations of a house, random initial state"),
        ("Choosing the excitation",
         ("k-NN surrogate of RMSE(θ) + Bayesian optimisation: power, switching probability, "
         "minimum duration"),
         "error and information mapped per duty cycle × minimum step"),
        ("Best excitation", "θ* per building and weather; longer windows more stable",
         "heat 55–65 %, steps 1.3–4 h; longer than 8 h"),
        ("Error metric", "P95 grows linearly with RMSE (Fig. 13)",
         "P95 ≈ 2 × RMSE (HLC: 47 vs 24 W/K)"),
        ("Beyond the paper", "—",
         "what 8 h can identify: C_n, not the split into losses and inertia"),
    ]  # fmt: skip
    for k, (aspect, paper, ours) in enumerate(rows):
        cells = [wrap(t, 10.5, wd) for t, wd in zip((aspect, paper, ours), widths, strict=True)]
        for c, lines, color, weight in zip(
            col, cells, (INK, INK_2, INK), ("semibold", "normal", "normal"), strict=True
        ):
            fig.text(c, yy, "\n".join(lines), fontsize=10.5, color=color, weight=weight,
                     va="top", linespacing=1.35)  # fmt: skip
        yy -= fy(max(map(len, cells)) * 10.5 * 1.35 / 72 + 0.16)
        if k < len(rows) - 1:
            fig.add_artist(plt.Line2D([x + fx(0.3), x + w - fx(0.3)], [yy + fy(0.08)] * 2,
                                      color=BORDER, lw=1))  # fmt: skip
    if yy < y + fy(0.15):
        print("  warning: the comparison table overflows")
    save(fig, "08_paper.png")


def slide_conclusion() -> None:
    fig = slide(
        9,
        "Conclusions",
        "What an 8-hour test can and cannot tell",
        "It measures the heat a building stores very well, and how that splits into losses and "
        "inertia much less.",
    )
    tiles = [
        ("8.3 %", "mean error on HLC\n(95 % within 21 %)", BLUE),
        ("R² 0.84", "on the effective capacity\nC_n = HLC × tau_n", INK),
        ("~0.73", "loss plateau of every model:\nthe information limit", ORANGE),
        ("55–65 %", "heated, with 1.3–4 h steps:\nthe most informative tests", AQUA),
    ]
    tw, gap = (0.91 - 3 * 0.015) / 4, 0.015
    for k, (value, label, c) in enumerate(tiles):
        stat(fig, (0.045 + k * (tw + gap), 0.535, tw, TOP - 0.535), value, label, c, size=32)
    text_card(
        fig,
        (0.045, BOT, 0.445, 0.425),
        "What we learned",
        [
            ("HLC is measurable, the taus much less", ("HLC 8.3 %, HLC_ext 9.6 % (MAPE) with "
             "calibrated intervals; tau_b and tau_inf R² ≈ 0.1.")),
            ("The plateau is in the data", ("curve twins, memorisation and model-free bounds "
             "agree: no architecture can move it on these tests.")),
            ("Scores hold for known protocols", "real tests add sensor noise and weather."),
        ],
        numbered=True,
    )  # fmt: skip
    text_card(
        fig,
        (0.51, BOT, 0.445, 0.425),
        "Recommendations for the protocol",
        [
            ("Always heat", "without heating HLC is not measurable (R² 0.13 at best)."),
            ("High duty cycle, long steps", ("heat 55–65 % of the time with 1.3–4 h on/off "
             "steps; avoid steps under ~40 min.")),
            ("Longer tests, and report C_n", ("HLC and the taus still gain information at 8 h; "
             "report C_n with the indicators and their intervals.")),
        ],
        numbered=True,
    )  # fmt: skip
    save(fig, "09_conclusion.png")


def main() -> None:
    style()
    d = load_examples()
    slide_problem(d)
    slide_data()
    slide_approach()
    slide_results()
    slide_plateau()
    slide_identifiability(d)
    slide_excitation()
    slide_paper(d)
    slide_conclusion()


if __name__ == "__main__":
    main()
