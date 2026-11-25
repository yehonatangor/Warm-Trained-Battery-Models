"""
SHAP attribution of the LSTM voltage prediction to its inputs, warm against cold (exploratory)
For sampled 20 min windows at each temperature, GradientExplainer (expected gradients) attributes the mean predicted
overpotential over the horizon [mV] to each input channel, summed over the 1501 time steps of the window
Background: windows of the 25 degC training data, so each attribution is the change from a warm reference
If the LSTM barely uses its temperature channels, or uses them in a way that breaks outside 25 to 40 degC,
the temperature share falls or swings in the cold
"""

import argparse

import numpy as np
import pandas as pd
import shap
import torch

import exploratory_common as ec  # also puts src on the path
from warmtrained.models.lstm import ETA_SCALE, SEQ, WARM, LSTMModel
from warmtrained.plotting import COL2, LABEL, OI, fig_legend
import matplotlib.pyplot as plt

OUT = ec.OUT / "shap"
CHANNELS = ["C-rate", "SOC", "Ambient temperature", "Arrhenius theta", "Arrhenius-scaled C-rate",
            "Warm-up voltage", "Warm-up temperature", "Warm-up flag"]
GROUPS = {"Current": [0], "Arrhenius-scaled current": [4], "SOC": [1], "Temperature": [2, 3],
          "Warm-up data": [5, 6, 7]}
GROUP_COLOR = {"Current": OI["grey"], "Arrhenius-scaled current": OI["blue"], "SOC": OI["green"],
               "Temperature": OI["vermillion"], "Warm-up data": OI["orange"]}


class MeanOverpotential(torch.nn.Module):
    # Window input -> mean predicted overpotential over the horizon [mV], one value per window
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out, _ = self.net(x)
        return out[:, WARM + 1:, 0].mean(dim=1, keepdim=True) * ETA_SCALE * 1e3


def sample_inputs(model, df, n, rng):
    # Network inputs of up to n random scorable windows, None if there are none
    if df is None:
        return None
    candidates = []
    for piece, g in df.groupby("piece", sort=False):
        w = ec.windows(g.reset_index(drop=True))
        if w is not None:
            candidates += [(piece, int(s)) for s in w[0]]
    if not candidates:
        return None

    picked = {}
    for k in rng.choice(len(candidates), min(n, len(candidates)), replace=False):
        piece, start = candidates[k]
        picked.setdefault(piece, []).append(start)

    X = []
    for piece, starts in picked.items():
        p = model._piece(df[df.piece == piece].reset_index(drop=True))
        X += [model._window(p, s - WARM, SEQ, WARM + 1) for s in starts]
    return np.array(X, dtype=np.float32)


def explain(explainer, X, device, chunk):
    # SHAP values per window and channel, summed over time [mV]
    out = []
    for a in range(0, len(X), chunk):
        sv = explainer.shap_values(torch.tensor(X[a:a + chunk], device=device))
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 4:  # newer shap adds a trailing output axis
            sv = sv[..., 0]
        out.append(sv.sum(axis=1))
    return np.concatenate(out)


def run(cell, n, n_bg, seed, chunk):
    files = ec.lstm_files(cell)
    if not files:
        print(f"{cell}: no LSTM, skipped", flush=True)
        return None
    match = [f for f in files if f.stem.endswith(f"_s{seed}")]
    if not match:
        print(f"{cell}: no LSTM seed {seed}, skipped", flush=True)
        return None

    ocv, _ = ec.ocv_ecm(cell)
    model = LSTMModel.load(match[0], ocv)
    rng = np.random.default_rng(0)

    bg = sample_inputs(model, ec.load(cell, "train", 25), n_bg, rng)
    if bg is None:
        print(f"{cell}: no 25 degC training windows for the background, skipped", flush=True)
        return None

    wrapper = MeanOverpotential(model.net).eval()
    rows = []
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN refuses LSTM gradients in eval mode
        explainer = shap.GradientExplainer(wrapper, torch.tensor(bg, device=model.device))
        for split, t in [("train", 25)] + ec.groups(cell):
            X = sample_inputs(model, ec.load(cell, split, t), n, rng)
            if X is None:
                continue
            sv = explain(explainer, X, model.device, chunk)
            rows += [dict(cell=cell, split=split, T_nominal=t, window=k, channel=ch, shap_mV=sv[k, c])
                     for k in range(len(sv)) for c, ch in enumerate(CHANNELS)]
            print(f"{cell} {split} {t} degC: {len(sv)} windows explained", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"shap_{cell}.csv", index=False)
    return d


def summary(d):
    # Per cell and temperature: mean |SHAP| of each channel group and its share of the total
    group_of = {CHANNELS[c]: g for g, cs in GROUPS.items() for c in cs}
    d = d.assign(group=d.channel.map(group_of))
    keys = ["cell", "split", "T_nominal"]

    per_window = d.groupby(keys + ["window", "group"]).shap_mV.sum().abs().reset_index()
    s = per_window.groupby(keys + ["group"]).shap_mV.mean().rename("mean_abs_shap_mV").reset_index()
    s["share"] = s.mean_abs_shap_mV / s.groupby(keys).mean_abs_shap_mV.transform("sum")
    return s


def figure(s):
    cells = list(dict.fromkeys(s.cell))
    fig, axes = plt.subplots(1, len(cells), figsize=(COL2, 2.4), sharey=True, layout="constrained", squeeze=False)
    for ax, cell in zip(axes[0], cells):
        c = s[s.cell == cell]
        for group, color in GROUP_COLOR.items():
            r = c[c.group == group].sort_values("T_nominal")
            ax.plot(r.T_nominal, r.share, color=color, marker="o", label=group)
        ax.set_title(LABEL.get(cell, cell))
        ax.set_xlabel("Temperature (°C)")
    axes[0, 0].set_ylabel("Share of mean |SHAP|")
    fig_legend(fig, axes[0], ncol=len(GROUP_COLOR))
    fig.savefig(OUT / "shap_by_temperature.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=ec.MAIN_CELLS)
    ap.add_argument("--windows", type=int, default=200, help="windows explained per temperature")
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="LSTM seed explained")
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    parts = [run(c, a.windows, a.background, a.seed, a.chunk) for c in a.cells]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        print("nothing explained")
        return

    s = summary(pd.concat(parts, ignore_index=True))
    s.to_csv(OUT / "shap_summary.csv", index=False)
    figure(s)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()