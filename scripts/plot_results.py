"""
Headline result figures from the saved results (training and tuning scores from results/<model>/summary.csv,
cold-test scores from results/test/), nothing is refitted or rescored here
1. results_error_vs_temperature.png: mean 20 min window voltage error against chamber temperature per cell for ARX,
   ECM on 2 and 3 temperatures, SINDy and LSTM (mean of 5 seeds, band = min to max), grey = test temperatures
2. results_hypotheses.png: share of -20 degC drive-cycle files won for H1 to H4 on B1, B3 and B5, with the verdict
3. results_known_truth.png: the same error curve for the simulated LG M50, where the true model is known
4. results_ensemble_band.png: coverage and width of the ensemble SINDy 5 to 95 % band against temperature
5. results_sindy_terms.png: share of ensemble refits keeping each SINDy term per cell (1 or 0 from the single
   full fit where no ensemble was aggregated)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warmtrained import config as cfg
from warmtrained.plotting import (CELL_STYLE, COL1, COL2, LABEL, MODEL_STYLE, fig_legend, heatmap,  # Agg first
                                  log_if_wide)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = cfg.RESULTS
OUT = R / "figures"
CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50"]
HYP_CELLS = ["Samsung30T", "LGHG2", "Tesla2170"]
MODELS = list(MODEL_STYLE)
# Each model in the training-side summaries (folder, variant) and in results/test/summary.csv
TRAIN_SRC = {"ARX": ("arx", None), "ECM (2 temps)": ("ecm", "single_2T"), "ECM (3 temps)": ("ecm", "single_3T"),
             "SINDy": ("sindy", "full"), "LSTM": ("lstm", None)}
TEST_NAME = {"ARX": "arx", "ECM (2 temps)": "ecm_single_2T", "ECM (3 temps)": "ecm_single_3T",
             "SINDy": "sindy_full"}
SINDY_BLOCKS = ("output", "branch1", "branch2", "thermal")


def train_side():
    # Window error at training and tuning temperatures (held-out drive cycles excluded)
    rows = []
    for model, (folder, variant) in TRAIN_SRC.items():
        s = pd.read_csv(R / folder / "summary.csv")
        if variant is not None:
            s = s[s.variant == variant]
        for r in s[s.split.isin(["train", "select"])].itertuples():
            rows.append(dict(cell=r.cell, model=model, T=r.T_nominal, rmse=r.rmse_V_mV,
                             lo=getattr(r, "rmse_V_min", np.nan), hi=getattr(r, "rmse_V_max", np.nan), test=False))
    return pd.DataFrame(rows)


def test_side():
    # Window error at the test temperatures, LSTM as mean, min and max over seeds
    s = pd.read_csv(R / "test" / "summary.csv")
    rows = [dict(cell=r.cell, model=model, T=r.T_nominal, rmse=r.rmse_V_mV, lo=np.nan, hi=np.nan, test=True)
            for model, name in TEST_NAME.items() for r in s[s.model == name].itertuples()]
    for (cell, T), g in s[s.model.str.startswith("lstm_s")].groupby(["cell", "T_nominal"]):
        rows.append(dict(cell=cell, model="LSTM", T=T, rmse=g.rmse_V_mV.mean(), lo=g.rmse_V_mV.min(),
                         hi=g.rmse_V_mV.max(), test=True))
    return pd.DataFrame(rows)


def error_panels(d, cells, fname, models=MODELS):
    n = len(cells)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL2 if ncol > 1 else COL1, 2.1 * nrow + 0.6), squeeze=False,
                             constrained_layout=True)
    for k, (ax, cell) in enumerate(zip(axes.flat, cells)):
        c = d[(d.cell == cell) & d.model.isin(models)]
        for t in sorted(c.loc[c.test, "T"].unique()):
            ax.axvspan(t - 2.5, t + 2.5, color="0.92", lw=0, zorder=0)
        for m in models:
            g = c[c.model == m].groupby("T", as_index=False).agg(rmse=("rmse", "mean"), lo=("lo", "min"),
                                                                  hi=("hi", "max")).sort_values("T")
            if g.empty:
                continue
            ax.plot(g["T"], g.rmse, label=m, **MODEL_STYLE[m])
            if g.lo.notna().any():
                ax.fill_between(g["T"], g.lo, g.hi, color=MODEL_STYLE[m]["color"], alpha=0.2, lw=0)
        log_if_wide(ax, c[["rmse", "lo", "hi"]].values.ravel())
        ax.set_title(LABEL[cell], loc="left")
        ax.set_xlabel("Temperature (°C)")
        if k % ncol == 0:
            ax.set_ylabel("RMSE (mV)")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig_legend(fig, axes.flat, ncol=len(models) if ncol > 1 else 2)
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_hypotheses():
    h = pd.read_csv(R / "test" / "hypotheses.csv")
    h = h[(h.status == "ok") & (h.block == 10)]
    name = {"H1": "H1 SINDy beats ECM", "H2": "H2 Butler-Volmer terms help", "H3": "H3 ECM beats LSTM",
            "H4": "H4 Two-process law beats one energy"}
    piv = {v: h.pivot(index="hypothesis", columns="cell", values=v)[[c for c in HYP_CELLS if c in h.cell.unique()]]
           for v in ("share", "required", "wins", "files")}
    share = piv["share"]
    text = share.copy().astype(object)
    for i in share.index:
        for c in share.columns:
            ok = share.loc[i, c] >= piv["required"].loc[i, c]
            text.loc[i, c] = (f"{int(piv['wins'].loc[i, c])}/{int(piv['files'].loc[i, c])} files\n"
                              f"{'supported' if ok else 'not supported'}")
    fig, ax = plt.subplots(figsize=(COL2, 2.4), constrained_layout=True)
    heatmap(ax, share, "Share of files won", annot=text.values, fmt="")
    ax.set_yticklabels([name[i] for i in share.index], rotation=0)
    ax.set_xticklabels([LABEL[c] for c in share.columns], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(OUT / "results_hypotheses.png")
    plt.close(fig)


def fig_band():
    rows = []
    for cell in CELLS:
        f = R / "sindy_ensemble" / cell / "band.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(cell=cell, test=False))
        f = R / "test" / cell / "ensemble_band.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(cell=cell, test=True))
    if not rows:
        return
    g = pd.concat(rows, ignore_index=True).groupby(["cell", "T_nominal"], as_index=False)[
        ["win_coverage", "win_width_mV"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), constrained_layout=True)
    for cell, c in g.groupby("cell", sort=False):
        c = c.sort_values("T_nominal")
        axes[0].plot(c.T_nominal, c.win_coverage, label=LABEL[cell], **CELL_STYLE[cell])
        axes[1].plot(c.T_nominal, c.win_width_mV, label=LABEL[cell], **CELL_STYLE[cell])
    axes[0].axhline(0.9, color="k", ls="--", lw=0.6)
    axes[0].text(axes[0].get_xlim()[0], 0.92, " nominal 90 %")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Coverage (fraction)")
    log_if_wide(axes[1], g.win_width_mV)
    axes[1].set_ylabel("Band width (mV)")
    for ax in axes:
        ax.set_xlabel("Temperature (°C)")
    fig_legend(fig, axes, ncol=3)
    fig.savefig(OUT / "results_ensemble_band.png")
    plt.close(fig)


def fig_sindy_terms():
    # Rows: library terms, columns: cells, colour: share of ensemble refits keeping the term
    # (kept 1 or dropped 0 in the single full fit where no ensemble was aggregated)
    cols = {}
    for cell in CELLS + ["SimLGM50"]:
        f = R / "sindy" / cell / "full" / "sindy_params.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        kept = {(eq, k): float(c != 0) for eq in SINDY_BLOCKS for k, c in p[eq].items()}
        inc = R / "sindy_ensemble" / cell / "inclusion.csv"
        if inc.exists():
            d = pd.read_csv(inc)
            kept = {(r.equation, r.term): r.inclusion for r in d.itertuples() if (r.equation, r.term) in kept}
            cols[f"{LABEL[cell]}\n({int(d.n_fits.iat[0])} refits)"] = kept
        else:
            cols[f"{LABEL[cell]}\n(single fit)"] = kept
    if not cols:
        return
    m = pd.DataFrame(cols)
    m.index = [f"{eq}: {t}" for eq, t in m.index]
    fig, ax = plt.subplots(figsize=(COL2, 0.2 * len(m) + 1.0), constrained_layout=True)
    heatmap(ax, m, "Share of refits keeping term", annot=True, fmt=".2f")
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(OUT / "results_sindy_terms.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.concat([train_side(), test_side()], ignore_index=True)
    error_panels(d, CELLS, "results_error_vs_temperature.png")
    error_panels(d, ["SimLGM50"], "results_known_truth.png", models=[m for m in MODELS if m != "LSTM"])
    fig_hypotheses()
    fig_band()
    fig_sindy_terms()
    print("wrote", *(OUT / n for n in ("results_error_vs_temperature.png", "results_hypotheses.png",
                                        "results_known_truth.png", "results_ensemble_band.png",
                                        "results_sindy_terms.png")))