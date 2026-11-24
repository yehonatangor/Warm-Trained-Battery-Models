"""
Winter exposure per model: weight the saved error-vs-temperature curves by how often each air
temperature occurs in hourly winter weather in Edmonton, Toronto and London ON. Nothing is refitted

Per city, model and cell (B1, B3, B5, tested down to -20 °C):
  - share of winter hours with 20 min window error above ERR_LIMIT
  - mean and p95 error over the winter
  - share of hours colder than the coldest tested temperature (counted apart, never extrapolated)

Cell temperature != air temperature, so two cases: offset 0 K (cold start after overnight parking)
and 10 K (pack warmed by driving or preheating)
Curves are interpolated linearly in log error and held at the end values outside the scored range
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_results import test_side, train_side
from warmtrained import config as cfg
from warmtrained.plotting import CELL_STYLE, COL1, COL2, LABEL, OI, fig_legend

OUT = cfg.RESULTS / "winter"
FIG = cfg.RESULTS / "figures"
WEATHER = cfg.DATA_RAW / "weather" / "weather_winters_2015_2025.csv"

CELLS = ["Samsung30T", "LGHG2", "Tesla2170"]
MODELS = ["ARX", "ECM (2 temps)", "ECM (3 temps)", "SINDy", "LSTM"]
CITY = {"EDMONTON_INTL_A": "Edmonton", "TORONTO_INTL_A": "Toronto", "LONDON_CS": "London ON"}
CITY_LOOK = {
    "Edmonton": dict(color=OI["blue"], ls="-"),
    "Toronto": dict(color=OI["orange"], ls="--"),
    "London ON": dict(color=OI["green"], ls=":"),
}
OFFSETS = [0, 10]  # [K] cell above air
ERR_LIMIT = 50.0  # [mV]


def load_curves():
    # Mean window RMSE [mV] per cell, model, temperature
    d = pd.concat([train_side(), test_side()], ignore_index=True)
    d = d.groupby(["cell", "model", "T"], as_index=False).rmse.mean()
    return d[d.cell.isin(CELLS) & d.model.isin(MODELS)]


def load_weather():
    w = pd.read_csv(WEATHER, usecols=["station", "time", "T"]).dropna(subset=["T"])
    w["city"] = w.station.map(CITY)
    return w.dropna(subset=["city"])


def exposure(curves, weather):
    rows = []
    for (cell, model), g in curves.groupby(["cell", "model"]):
        g = g.sort_values("T")
        T_scored = g["T"].to_numpy(float)
        log_err = np.log(g.rmse.to_numpy())
        t_min = min(cfg.SPLITS[cell]["test"])

        for city, h in weather.groupby("city"):
            for off in OFFSETS:
                T = h["T"].to_numpy() + off
                inside = T >= t_min
                err = np.exp(np.interp(T[inside], T_scored, log_err))
                rows.append(dict(
                    city=city, offset_K=off, cell=cell, model=model, hours=len(T),
                    share_beyond=float(1 - inside.mean()),
                    share_over_limit=float((err > ERR_LIMIT).sum() / len(T)),
                    mean_err_mV=float(err.mean()) if err.size else np.nan,
                    p95_err_mV=float(np.percentile(err, 95)) if err.size else np.nan,
                    coldest_scored=float(T_scored.min()),
                ))
    return pd.DataFrame(rows)


def plot_exposure(e):
    # Rows: offset, columns: city, bar groups: model, colour: cell
    fig, axes = plt.subplots(len(OFFSETS), len(CITY), figsize=(COL2, 5.0), sharey=True,
                             layout="constrained", squeeze=False)
    x = np.arange(len(MODELS))
    width = 0.8 / len(CELLS)

    for i, off in enumerate(OFFSETS):
        for j, city in enumerate(CITY.values()):
            ax = axes[i, j]
            s = e[(e.city == city) & (e.offset_K == off)]
            for k, cell in enumerate(CELLS):
                share = s[s.cell == cell].set_index("model").reindex(MODELS).share_over_limit
                shift = (k - (len(CELLS) - 1) / 2) * width
                ax.bar(x + shift, share.to_numpy(), width,
                       color=CELL_STYLE[cell]["color"], label=LABEL[cell])
            beyond = s.share_beyond.iat[0] if len(s) else 0.0
            ax.set_title(f"{city}, air + {off} K\n{beyond:.0%} of hours colder than tested")
            ax.set_xticks(x, MODELS, rotation=35, ha="right")
            ax.grid(False, axis="x")
        axes[i, 0].set_ylabel(f"Winter hours over {ERR_LIMIT:.0f} mV")

    fig_legend(fig, axes.flat, ncol=len(CELLS))
    fig.savefig(FIG / "results_winter.png")
    plt.close(fig)


def plot_temperatures(weather):
    fig, ax = plt.subplots(figsize=(COL1, 2.4), layout="constrained")
    bins = np.arange(-46, 27, 1)
    for city, look in CITY_LOOK.items():
        hist, edges = np.histogram(weather.loc[weather.city == city, "T"], bins=bins, density=True)
        ax.step(edges[:-1], hist, where="post", label=city, **look)
    ax.axvspan(-20, 26, color="0.93", lw=0, zorder=0)  # tested range
    ax.set_xlim(-46, 26)
    ax.set_xlabel("Hourly air temperature (°C)")
    ax.set_ylabel("Share of winter hours per °C")
    fig_legend(fig, [ax], ncol=3)
    fig.savefig(FIG / "winter_temperatures.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    weather = load_weather()
    e = exposure(load_curves(), weather)
    e.to_csv(OUT / "winter_exposure.csv", index=False)

    plot_exposure(e)
    plot_temperatures(weather)

    summary = e[e.offset_K == 0].pivot_table(index=["city", "model"], columns="cell",
                                             values="share_over_limit")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()