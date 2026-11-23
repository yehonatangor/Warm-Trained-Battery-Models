"""
Shared figure settings for every plot script
IEEE printed sizes, Times New Roman at 9 pt, Okabe-Ito colours with a fixed marker and line style per model and cell
(readable in greyscale), one blue scale for every heat map (darker = larger), left and bottom axis lines only,
light grey grid, every line at least 0.5 pt
Importing it selects the Agg backend and sets the rcParams, so import it before matplotlib.pyplot
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")

COL1, COL2 = 3.5, 7.16  # IEEE one- and two-column widths (in)
FS = 9  # every text size (pt)
DPI = 600

# Okabe-Ito palette, grey added for ARX and unused data
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#999999")
GRID = "#d9d9d9"
CMAP = "Blues"

MODEL_STYLE = {  # one look per model in every figure
    "ARX": dict(color=OI["grey"], marker="D", ls="-."),
    "ECM (2 temps)": dict(color=OI["blue"], marker="o", ls="-"),
    "ECM (3 temps)": dict(color=OI["sky"], marker="p", ls=(0, (6, 2))),
    "SINDy": dict(color=OI["orange"], marker="s", ls="--"),
    "LSTM": dict(color=OI["vermillion"], marker="^", ls=":"),
}
CELL_STYLE = {  # one look per cell where cells share an axis
    "Samsung30T": dict(color=OI["black"], marker="o", ls="-"),
    "LGM50": dict(color=OI["orange"], marker="s", ls="--"),
    "LGHG2": dict(color=OI["sky"], marker="^", ls=":"),
    "TUBerlinM50LT": dict(color=OI["green"], marker="D", ls="-."),
    "Tesla2170": dict(color=OI["vermillion"], marker="v", ls=(0, (6, 2))),
    "Panasonic18650PF": dict(color=OI["purple"], marker="P", ls=(0, (3, 1, 1, 1, 1, 1))),
    "SimLGM50": dict(color=OI["blue"], marker="X", ls=(0, (1, 2))),
}
SPLIT_STYLE = {  # bar fill and hatch per data split
    "Train": dict(color=OI["blue"], hatch=""),
    "Held-out drive cycles": dict(color=OI["sky"], hatch="//"),
    "Tuning": dict(color=OI["green"], hatch=".."),
    "Test": dict(color=OI["vermillion"], hatch="xx"),
    "Not used": dict(color=OI["grey"], hatch="--"),
}
LABEL = {"Samsung30T": "B1 Samsung 30T", "LGM50": "B2 LG M50", "LGHG2": "B3 LG HG2",
         "TUBerlinM50LT": "B4 LG M50LT", "Tesla2170": "B5 Tesla 2170", "Panasonic18650PF": "B6 Panasonic 18650PF",
         "SimLGM50": "S1 simulated LG M50"}

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.0, "lines.markersize": 4, "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": DPI, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def log_if_wide(ax, values, axis="y"):
    # Log scale where the plotted positive values span tenfold or more
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) and v.max() >= 10 * v.min():
        (ax.set_yscale if axis == "y" else ax.set_xscale)("log")


def fig_legend(fig, axes, ncol, order=None):
    # One legend below the figure, each label once, in first-seen order or the given order
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    labels = [l for l in order if l in handles] if order else list(handles)
    fig.legend([handles[l] for l in labels], labels, loc="outside lower center", ncol=ncol)


def heatmap(ax, data, cbar_label, **kw):
    # Blue 0 to 1 heat map with 9 pt annotations, no grid
    import seaborn as sns
    sns.heatmap(data, ax=ax, cmap=CMAP, vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                cbar_kws={"label": cbar_label}, annot_kws={"fontsize": FS}, **kw)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.collections[0].colorbar.ax.grid(False)