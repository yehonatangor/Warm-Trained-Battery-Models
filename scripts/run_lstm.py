"""
Train the LSTM for each cell type, then score it with the ARX harness, test data are never loaded here
Grid of four sizes (hidden units x layers) with seed 0, chosen by the early-stopping loss on the selection
temperature (B2: one held-out training cell), then the chosen size retrained with seeds 1 to 4
All five seeds are scored so the spread across seeds can be reported
Resumable: a grid point or seed whose files exist is loaded, not retrained
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.lstm import LSTMModel
from warmtrained.models.ocv import OCVCurve

OUT = cfg.RESULTS / "lstm"
GRID = [(32, 1), (64, 1), (64, 2), (128, 1)]
SEEDS = [0, 1, 2, 3, 4]
SCORED_SPLITS = ["train", "select", "holdout_profile"]
GROUP = ["split", "T_nominal"]
FIXED_SIZE = {"Panasonic18650PF": (64, 2)}  # B6 (Panasonic): size fixed to the B3 choice, so no grid search


def data_for(cell):
    # (training data, early-stopping data, held-out unit or source)
    train = dataset.load(cell, ["train"])
    if cell == "LGM50":  # no selection temperature: hold out one training cell
        unit = sorted(train.unit.unique())[-1]
        return train[train.unit != unit], train[train.unit == unit], unit
    if cell == "Panasonic18650PF":  # no selection temperature: the held-out drive cycles at 25 and 10 degC
        return train, dataset.load(cell, ["holdout_profile"]), "holdout_profile"
    return train, dataset.load(cell, ["select"]), None


def train_one(cell, ocv, E0, train, stop, hidden, layers, seed, epochs):
    # Train one size and seed, or load it if saved
    f = OUT / cell / f"h{hidden}_l{layers}_s{seed}.json"
    if f.exists():
        return LSTMModel.load(f, ocv)
    t0 = time.time()
    print(f"{cell}: hidden {hidden}, layers {layers}, seed {seed}", flush=True)
    m = LSTMModel(ocv, cfg.MODEL_CELLS[cell]["nominal_ah"], E0, hidden, layers, seed=seed).fit(train, stop,
                                                                                                epochs=epochs)
    m.train_seconds = time.time() - t0
    m.save(f)
    print(f"{cell}: done in {m.train_seconds:.0f} s, stop loss {m.best_stop_loss:.3f}", flush=True)
    return m


def score(models, cell):
    # Every seed, one split and temperature at a time to save memory
    m = dataset.manifest()
    rows = m[(m.cell == cell) & m.split.isin(SCORED_SPLITS)]
    groups = rows[GROUP].drop_duplicates().sort_values(GROUP)
    parts_w, parts_f = [], []
    for split, t_nom in zip(groups.split, groups.T_nominal):
        data = dataset.load(cell, [split], temps=[t_nom])
        for seed, model in models.items():
            w, f = harness.evaluate(model, data)
            parts_w.append(w.assign(seed=seed))
            parts_f.append(f.assign(seed=seed))
        del data
    return pd.concat(parts_w, ignore_index=True), pd.concat(parts_f, ignore_index=True)


def summary_table(cell, h, l, windows, freerun):
    # Per split and temperature: mean over seeds of the per-seed scores, with min and max window RMSE
    keys = GROUP + ["seed"]
    per_seed = windows.groupby(keys).agg(rmse_V_mV=("rmse_V_mV", "mean"), rmse_T_K=("rmse_T_K", "mean")).reset_index()
    fr = freerun.groupby(keys).agg(free_rmse_V_mV=("rmse_V_mV", "median"),
                                   free_bias_V_mV=("bias_V_mV", "median")).reset_index()
    per_seed = per_seed.merge(fr, on=keys, how="left")
    s = per_seed.groupby(GROUP).agg(
        rmse_V_mV=("rmse_V_mV", "mean"), rmse_V_min=("rmse_V_mV", "min"), rmse_V_max=("rmse_V_mV", "max"),
        rmse_T_K=("rmse_T_K", "mean"), free_rmse_V_mV=("free_rmse_V_mV", "mean"),
        free_bias_V_mV=("free_bias_V_mV", "mean")).reset_index()
    s.insert(0, "variant", f"h{h}_l{l}")
    s.insert(0, "cell", cell)
    return s


def run(cell, epochs):
    # Grid search, seed retrains, scoring and saving for one cell type, return its summary rows
    out = OUT / cell
    out.mkdir(parents=True, exist_ok=True)
    ocv = OCVCurve.load(cfg.RESULTS / "arx" / cell / "ocv.json")
    E0 = json.loads((cfg.RESULTS / "ecm" / cell / "single_2T" / "ecm_params.json").read_text())["E_J_per_mol"][0]
    train, stop, held = data_for(cell)

    grid = []
    for h, l in ([FIXED_SIZE[cell]] if cell in FIXED_SIZE else GRID):
        m = train_one(cell, ocv, E0, train, stop, h, l, 0, epochs)
        grid.append(dict(hidden=h, layers=l, stop_loss=m.best_stop_loss, held_out_unit=held))
    grid = pd.DataFrame(grid)
    grid.to_csv(out / "grid.csv", index=False)
    best = grid.loc[grid.stop_loss.idxmin()]
    h, l = int(best.hidden), int(best.layers)
    models = {s: train_one(cell, ocv, E0, train, stop, h, l, s, epochs) for s in SEEDS}
    del train, stop

    windows, freerun = score(models, cell)
    windows.to_parquet(out / "windows.parquet", index=False)
    freerun.to_csv(out / "freerun.csv", index=False)
    harness.summarise(windows, freerun).to_csv(out / "by_file.csv", index=False)

    s = summary_table(cell, h, l, windows, freerun)
    print(s.round(2).to_string(index=False), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=list(cfg.CELLS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([run(c, a.epochs) for c in a.cells], ignore_index=True)

    path = OUT / "summary.csv"
    if path.exists():  # keep the rows of cells not rerun
        old = pd.read_csv(path)
        summary = pd.concat([old[~old.cell.isin(a.cells)], summary], ignore_index=True)
    summary.sort_values(["cell", "split", "T_nominal"]).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()