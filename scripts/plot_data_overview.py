"""
Two overview figures of the cleaned, usable data, all splits including test (counts only, no model sees them)
1. data_coverage_by_cell.png: per cell type, density map of C-rate against measured cell temperature over every
   usable 1 s row, shows which temperatures and C-rates each dataset covers and where it has none (cold charging)
2. data_hours_by_temperature.png: one row per cell type on a shared temperature axis, usable hours per 2 K bin of
   measured cell temperature by split, shows the warm-to-cold split and each dataset's spread together
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package

from warmtrained import config as cfg
from warmtrained.plotting import CMAP, COL2, LABEL, SPLIT_STYLE, fig_legend  # Agg first

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

OUT = cfg.RESULTS / "figures"
CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50", "Panasonic18650PF"]
SPLIT_GROUP = {"train": "Train", "holdout_profile": "Held-out drive cycles", "select": "Tuning",
               "test": "Test", "used_info": "Not used", "trise": "Test"}  # B6 warming-chamber tests counted as test
T_EDGES = np.arange(-26, 56, 2.0)  # cell temperature bins (degC)
C_EDGES = np.arange(-4.0, 4.05, 0.1)  # C-rate bins


def accumulate():
    # Per cell: 2D counts (C-rate x temperature) and hours per temperature bin per split, read file by file
    m = pd.read_csv(cfg.DATA_PROCESSED / "manifest.csv")
    m = m[m.split != "excluded"]
    grid, hours = {}, {}
    for cell in CELLS:
        nom = cfg.CELLS[cell]["nominal_ah"]
        H = np.zeros((len(C_EDGES) - 1, len(T_EDGES) - 1))
        hs = {k: np.zeros(len(T_EDGES) - 1) for k in SPLIT_STYLE}
        for _, r in m[m.cell == cell].iterrows():
            d = pd.read_parquet(cfg.DATA_PROCESSED / cell / str(r.T_nominal) / (Path(r.file).stem + ".parquet"),
                                columns=["I", "T"])
            d = d[np.isfinite(d["T"])]
            H += np.histogram2d(d.I.values / nom, d["T"].values, bins=[C_EDGES, T_EDGES])[0]
            hs[SPLIT_GROUP[r.split]] += np.histogram(d["T"].values, bins=T_EDGES)[0] / 3600.0  # 1 s rows to hours
        grid[cell], hours[cell] = H, hs
    return grid, hours


def fig_coverage(grid):
    ncol = 3
    fig, axes = plt.subplots(2, ncol, figsize=(COL2, 4.4), sharex=True, sharey=True, constrained_layout=True)
    vmax = max(g.max() for g in grid.values()) / 3600
    for k, (ax, cell) in enumerate(zip(axes.flat, CELLS)):
        H = np.ma.masked_less_equal(grid[cell] / 3600.0, 0)  # hours per map cell
        mesh = ax.pcolormesh(T_EDGES, C_EDGES, H, cmap=CMAP, norm=LogNorm(vmin=1e-3, vmax=vmax), shading="flat")
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(LABEL[cell], loc="left")
        ax.set_xlim(-26, 54)
        if k >= ncol:
            ax.set_xlabel("Temperature (°C)")
        if k % ncol == 0:
            ax.set_ylabel("C-rate (charge > 0)")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("Time (h)")
    cb.ax.grid(False)
    fig.savefig(OUT / "data_coverage_by_cell.png")
    plt.close(fig)


def fig_hours(hours):
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(COL2, 1.05 * len(CELLS) + 0.7), sharex=True,
                             constrained_layout=True)
    left, width = T_EDGES[:-1], np.diff(T_EDGES)
    for ax, cell in zip(axes, CELLS):
        bottom = np.zeros(len(left))
        for group, st in SPLIT_STYLE.items():
            h = hours[cell][group]
            if h.sum() == 0:
                continue
            ax.bar(left, h, width=width, bottom=bottom, align="edge", color=st["color"], hatch=st["hatch"],
                   edgecolor="white", linewidth=0.5, label=group)
            bottom += h
        ax.set_title(f"{LABEL[cell]} ({bottom.sum():,.0f} h)", loc="left")
        ax.set_ylabel("Time (h)")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Temperature (°C)")
    axes[-1].set_xlim(T_EDGES[0], T_EDGES[-1])
    fig_legend(fig, axes, ncol=5, order=list(SPLIT_STYLE))
    fig.savefig(OUT / "data_hours_by_temperature.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid, hours = accumulate()
    fig_coverage(grid)
    fig_hours(hours)
    print("wrote", OUT / "data_coverage_by_cell.png", "and", OUT / "data_hours_by_temperature.png")