"""
Fit the 25 degC OCV curve and the ARX floor model for each cell type on its training split,
then score it on the training, validation and held-out-profile splits, test data are never loaded here

Usage:
    python scripts/run_arx.py                      (all cells)
    python scripts/run_arx.py Samsung30T LGHG2     (some cells)

Output:
    results/arx/<cell>/ocv.json, arx_params.json, windows.parquet, freerun.csv, by_file.csv
    results/arx/summary.csv (one row per cell, split and temperature)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package

import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.arx import ARXModel
from warmtrained.models.ocv import fit_ocv

OUT = cfg.RESULTS / "arx"
SCORED_SPLITS = ["train", "select", "holdout_profile"]
GROUP = ["split", "T_nominal"]


def fit(cell, out):
    # Fit the OCV curve (25 degC training data) and the ARX model (all training data), save both
    train = dataset.load(cell, ["train"])
    ocv = fit_ocv(train[train.T_nominal == 25], cfg.MODEL_CELLS[cell]["nominal_ah"])
    ocv.save(out / "ocv.json")

    model = ARXModel(ocv).fit(train)
    model.save(out / "arx_params.json")
    return model


def score(cell, model):
    # Score the model one split and temperature at a time (B5 alone is several million rows)
    m = dataset.manifest()
    rows = m[(m.cell == cell) & m.split.isin(SCORED_SPLITS)]
    groups = rows[GROUP].drop_duplicates().sort_values(GROUP)

    parts_w, parts_f = [], []
    for split, t_nom in zip(groups.split, groups.T_nominal):
        data = dataset.load(cell, [split], temps=[t_nom])
        w, f = harness.evaluate(model, data)
        parts_w.append(w)
        parts_f.append(f)
        del data  # free memory before the next group

    return pd.concat(parts_w, ignore_index=True), pd.concat(parts_f, ignore_index=True)


def summary_table(cell, windows, freerun):
    # One row per split and temperature: mean window scores and median free-run scores
    s = windows.groupby(GROUP).agg(files=("file", "nunique"), windows=("rmse_V_mV", "size"),
                                   rmse_V_mV=("rmse_V_mV", "mean"),
                                   rmse_T_K=("rmse_T_K", "mean")).reset_index()
    f = freerun.groupby(GROUP).agg(free_rmse_V_mV=("rmse_V_mV", "median"),
                                   free_bias_V_mV=("bias_V_mV", "median")).reset_index()
    s = s.merge(f, on=GROUP, how="left")
    s.insert(0, "cell", cell)
    return s


def run(cell):
    # Fit, score and save one cell type, return its summary rows
    t0 = time.time()
    out = OUT / cell
    out.mkdir(parents=True, exist_ok=True)

    model = fit(cell, out)
    print(f"{cell}: fitted in {time.time() - t0:.0f} s  {model.params()}", flush=True)

    windows, freerun = score(cell, model)
    windows.to_parquet(out / "windows.parquet", index=False)
    freerun.to_csv(out / "freerun.csv", index=False)
    harness.summarise(windows, freerun).to_csv(out / "by_file.csv", index=False)

    s = summary_table(cell, windows, freerun)
    print(s.round(2).to_string(index=False), f"\n{cell}: done in {time.time() - t0:.0f} s\n", flush=True)
    return s


def main(cells):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([run(c) for c in cells], ignore_index=True)

    path = OUT / "summary.csv"
    if path.exists():  # keep the rows of cells not rerun
        old = pd.read_csv(path)
        summary = pd.concat([old[~old.cell.isin(cells)], summary], ignore_index=True)
    summary.sort_values(["cell", "split", "T_nominal"]).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(cfg.CELLS))