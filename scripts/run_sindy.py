"""
Fit the Arrhenius SINDy model for each cell type, started from its single-E ECM, then score it with the harness,
test data are never loaded here, run scripts/run_ecm.py first

Variants:
    full    whole library, energies from ECM single_2T, training temperatures
    no_bv   Butler-Volmer columns removed
Each fit is timed (fit_seconds sets the ensemble size), the correlation of each Butler-Volmer column with f0 I at
25 degC is saved with the fit, the two are nearly collinear at small currents
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.ocv import OCVCurve
from warmtrained.models.sindy import SINDyModel

OUT = cfg.RESULTS / "sindy"
SCORED_SPLITS = ["train", "select", "holdout_profile"]
GROUP = ["split", "T_nominal"]
VARIANTS = ["full", "no_bv"]
DEFAULT_VARIANTS = VARIANTS


def score(cell, model):
    # One split and temperature at a time to save memory
    m = dataset.manifest()
    rows = m[(m.cell == cell) & m.split.isin(SCORED_SPLITS)]
    groups = rows[GROUP].drop_duplicates().sort_values(GROUP)
    parts_w, parts_f = [], []
    for split, t_nom in zip(groups.split, groups.T_nominal):
        data = dataset.load(cell, [split], temps=[t_nom])
        w, f = harness.evaluate(model, data)
        parts_w.append(w)
        parts_f.append(f)
        del data
    return pd.concat(parts_w, ignore_index=True), pd.concat(parts_f, ignore_index=True)


def summary_table(cell, name, p, windows, freerun):
    # One row per split and temperature: mean window scores, median free-run scores and the selected library
    s = windows.groupby(GROUP).agg(files=("file", "nunique"), windows=("rmse_V_mV", "size"),
                                   rmse_V_mV=("rmse_V_mV", "mean"),
                                   rmse_T_K=("rmse_T_K", "mean")).reset_index()
    f = freerun.groupby(GROUP).agg(free_rmse_V_mV=("rmse_V_mV", "median"),
                                   free_bias_V_mV=("bias_V_mV", "median"),
                                   free_rmse_T_K=("rmse_T_K", "median")).reset_index()
    s = s.merge(f, on=GROUP, how="left")
    s.insert(0, "variant", name)
    s.insert(0, "cell", cell)
    s["fit_seconds"] = p["fit_seconds"]
    s["n_terms"] = sum(len(v) for v in p["selected"].values())
    s["bv_selected"] = p["bv_selected"]
    return s


def run(cell, variants):
    # Fit, score and save every variant of one cell type, return its summary rows
    ocv = OCVCurve.load(cfg.RESULTS / "arx" / cell / "ocv.json")
    summaries = []
    for name in variants:
        ecm_file = cfg.RESULTS / "ecm" / cell / "single_2T" / "ecm_params.json"
        if not ecm_file.exists():
            print(f"{cell} {name}: skipped, {ecm_file.parent.name} ECM not fitted", flush=True)
            continue
        t0 = time.time()
        out = OUT / cell / name
        out.mkdir(parents=True, exist_ok=True)

        train = dataset.load(cell, ["train"])
        model = SINDyModel(ocv, json.loads(ecm_file.read_text()), cfg.MODEL_CELLS[cell]["nominal_ah"], name).fit(train)
        corr = model.bv_correlation(train[train.T_nominal == 25])
        del train
        model.save(out / "sindy_params.json", {"bv_corr_with_f0I_25C": corr})
        p = model.params()
        print(f"{cell} {name}: fitted in {p['fit_seconds']:.0f} s, selected {p['selected']}", flush=True)

        windows, freerun = score(cell, model)
        windows.to_parquet(out / "windows.parquet", index=False)
        freerun.to_csv(out / "freerun.csv", index=False)
        harness.summarise(windows, freerun).to_csv(out / "by_file.csv", index=False)

        s = summary_table(cell, name, p, windows, freerun)
        print(s.round(2).to_string(index=False), f"\n{cell} {name}: done in {time.time() - t0:.0f} s\n", flush=True)
        summaries.append(s)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=list(cfg.CELLS))
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=VARIANTS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([run(c, a.variants) for c in a.cells], ignore_index=True)

    path = OUT / "summary.csv"
    if path.exists():  # keep the rows of cells and variants not rerun
        old = pd.read_csv(path)
        summary = pd.concat([old[~(old.cell.isin(a.cells) & old.variant.isin(a.variants))], summary],
                            ignore_index=True)
    summary.sort_values(["cell", "variant", "split", "T_nominal"]).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()