"""
Split conformal prediction intervals for every model, calibrated warm and tested cold (exploratory)
Calibration: absolute voltage error of each model on the selection-temperature windows, per 60 s horizon bin
Interval: prediction +- q, q the conformal quantile of those errors for a nominal 90 % coverage
    standard   q from the raw errors, the same width at every temperature
    arrhenius  errors divided by the ECM total resistance factor c(T_amb) (Arrhenius energies fitted warm),
               interval q c(T_amb): physics says how much the error should grow in the cold
Coverage at the test temperatures shows whether a warm-calibrated interval still holds, and whether the
fitted temperature law is enough to carry it into the cold
Windows overlap and rows are autocorrelated, so coverage is an empirical check, not a guarantee
"""

import argparse

import numpy as np
import pandas as pd

import exploratory_common as ec  # also puts src on the path
from warmtrained import config as cfg
from warmtrained.models.arx import ARXModel
from warmtrained.models.sindy import SINDyModel
from warmtrained.plotting import COL2, LABEL, MODEL_STYLE, fig_legend
import matplotlib.pyplot as plt

OUT = ec.OUT / "conformal"
ALPHA = 0.10
BIN_S = 60
N_BINS = ec.HOR // BIN_S
CAL_ROW_STEP = 5  # keep every 5th measured calibration row to save memory
VARIANTS = ("standard", "arrhenius")


def models(cell):
    R = cfg.RESULTS
    ocv, ecm = ec.ocv_ecm(cell)
    out = {"ARX": ARXModel.load(R / "arx" / cell / "arx_params.json", ocv), "ECM (2 temps)": ecm}

    sindy = R / "sindy" / cell / "full" / "sindy_params.json"
    if sindy.exists():
        out["SINDy"] = SINDyModel.load(sindy, ocv)

    lstm = ec.lstm_files(cell)
    if lstm:
        from warmtrained.models.lstm import LSTMModel  # torch only loaded when needed
        out["LSTM"] = LSTMModel.load(lstm[0], ocv)
    return out, ecm


def errors(model, ecm, df, every):
    # Per piece: absolute error [V], Arrhenius-scaled error, factor c, horizon bin and measured mask per window row
    for _, g in df.groupby("piece", sort=False):
        g = g.reset_index(drop=True)
        w = ec.windows(g, every)
        if w is None:
            continue
        starts, idx, mv = w
        e = np.abs(model.predict_windows(g, starts, ec.HOR)[0] - g.V.values[idx])
        c = ec.resistance_factor(ecm, g.T_amb.values[idx])
        b = np.broadcast_to(np.arange(ec.HOR) // BIN_S, e.shape)
        yield e, e / c, c, b, mv


def quantiles(scores, bins):
    # Conformal quantile per horizon bin, NaN for an empty bin
    q = np.full(N_BINS, np.nan)
    for k in range(N_BINS):
        s = np.sort(scores[bins == k])
        n = len(s)
        if n:
            q[k] = s[min(int(np.ceil((n + 1) * (1 - ALPHA))) - 1, n - 1)]
    return q


def calibrate(model, ecm, cal, every):
    # Quantiles per variant from the calibration windows, None if there are none
    raw, scaled, bins = [], [], []
    for e, es, _, b, mv in errors(model, ecm, cal, every):
        keep = np.flatnonzero(mv.ravel())[::CAL_ROW_STEP]
        raw.append(e.ravel()[keep])
        scaled.append(es.ravel()[keep])
        bins.append(b.ravel()[keep])
    if not raw:
        return None
    bins = np.concatenate(bins)
    return {"standard": quantiles(np.concatenate(raw), bins), "arrhenius": quantiles(np.concatenate(scaled), bins)}


def run(cell, every):
    ms, ecm = models(cell)
    cal = ec.load(cell, *ec.groups(cell)[0])
    if cal is None:
        print(f"{cell}: no calibration data, skipped", flush=True)
        return []

    rows = []
    for name, m in ms.items():
        q = calibrate(m, ecm, cal, every)
        if q is None:
            print(f"{cell} {name}: no calibration windows, skipped", flush=True)
            continue

        for split, t in ec.groups(cell):
            df = ec.load(cell, split, t)
            if df is None:
                continue
            inside = dict.fromkeys(VARIANTS, 0)
            width = dict.fromkeys(VARIANTS, 0.0)
            total = 0
            for e, _, c, b, mv in errors(m, ecm, df, every):
                half = {"standard": q["standard"][b], "arrhenius": q["arrhenius"][b] * c}
                for k, h in half.items():
                    inside[k] += int((e <= h)[mv].sum())
                    width[k] += float((2 * h)[mv].sum())
                total += int(mv.sum())
            if total:
                rows += [dict(cell=cell, model=name, variant=k, split=split, T_nominal=t,
                              coverage=inside[k] / total, width_mV=width[k] / total * 1e3) for k in VARIANTS]
        print(f"{cell} {name} done", flush=True)
    return rows


def figure(d):
    line = {"standard": dict(mfc="white", ls="--"), "arrhenius": {}}
    label = {"standard": "standard", "arrhenius": "Arrhenius-scaled"}
    cells = list(dict.fromkeys(d.cell))
    fig, axes = plt.subplots(2, len(cells), figsize=(COL2, 4.2), sharex="col", sharey="row", layout="constrained",
                             squeeze=False)
    for j, cell in enumerate(cells):
        c = d[d.cell == cell]
        for model, look in MODEL_STYLE.items():
            for variant in VARIANTS:
                r = c[(c.model == model) & (c.variant == variant)].sort_values("T_nominal")
                if len(r):
                    style = {**look, **line[variant]}
                    axes[0, j].plot(r.T_nominal, r.coverage, **style, label=f"{model}, {label[variant]}")
                    axes[1, j].plot(r.T_nominal, r.width_mV, **style)
        axes[0, j].axhline(1 - ALPHA, color="black", lw=0.6, ls=":")
        axes[0, j].set_title(LABEL.get(cell, cell))
        axes[0, j].set_ylim(0, 1)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("Temperature (°C)")
    axes[0, 0].set_ylabel("Coverage")
    axes[1, 0].set_ylabel("Interval width (mV)")
    fig_legend(fig, axes[0], ncol=2)
    fig.savefig(OUT / "conformal_by_temperature.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=ec.MAIN_CELLS)
    ap.add_argument("--every", type=int, default=2, help="use every k-th window")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame([r for c in a.cells for r in run(c, a.every)])
    if d.empty:
        print("no conformal results")
        return

    d.to_csv(OUT / "conformal.csv", index=False)
    figure(d)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()