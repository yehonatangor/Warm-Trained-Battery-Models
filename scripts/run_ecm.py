"""
Fit the Arrhenius ECM variants for each cell type, then score them with the ARX harness, test data are never loaded here
The 25 degC OCV curve is the one run_arx.py saved, fitted again with the same recipe if missing

Variants:
    single_2T              one activation energy per resistance, training temperatures
    single_3T              same law, training plus selection temperatures
    two_E54.8              two-process law, second energy fixed at 54.8 kJ/mol, training temperatures
    two_E44.8, two_E64.8   sensitivity to the fixed energy (+-10 kJ/mol)
    two3T_E*               two-process law, training plus selection temperatures
Cells with no selection temperature skip the 3T variants and the E44.8, E64.8 sensitivity cases
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package

import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.ecm import ECMModel
from warmtrained.models.ocv import OCVCurve, fit_ocv

OUT = cfg.RESULTS / "ecm"
SCORED_SPLITS = ["train", "select", "holdout_profile"]
GROUP = ["split", "T_nominal"]
VARIANTS = {  # name: (law, fixed second energy [J/mol], selection temperature in the fit)
    "single_2T": ("single", None, False),
    "single_3T": ("single", None, True),
    "two_E54.8": ("two_process", 54.8e3, False),
    "two_E44.8": ("two_process", 44.8e3, False),
    "two_E64.8": ("two_process", 64.8e3, False),
    "two3T_E54.8": ("two_process", 54.8e3, True),
    "two3T_E44.8": ("two_process", 44.8e3, True),
    "two3T_E64.8": ("two_process", 64.8e3, True),
}
DEFAULT_VARIANTS = list(VARIANTS)
NO_SELECT = {"LGM50", "Panasonic18650PF"}


def get_ocv(cell):
    f = cfg.RESULTS / "arx" / cell / "ocv.json"
    if f.exists():
        return OCVCurve.load(f)
    return fit_ocv(dataset.load(cell, ["train"], temps=[25]), cfg.MODEL_CELLS[cell]["nominal_ah"])


def make_model(cell, name, ocv):
    law, e2, _ = VARIANTS[name]
    return ECMModel(ocv, law, e2 or 54.8e3, cfg.MODEL_CELLS[cell]["nominal_ah"])


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


def summary_table(cell, name, with_select, windows, freerun):
    # One row per split and temperature: mean window scores and median free-run scores
    s = windows.groupby(GROUP).agg(files=("file", "nunique"), windows=("rmse_V_mV", "size"),
                                   rmse_V_mV=("rmse_V_mV", "mean"),
                                   rmse_T_K=("rmse_T_K", "mean")).reset_index()
    f = freerun.groupby(GROUP).agg(free_rmse_V_mV=("rmse_V_mV", "median"),
                                   free_bias_V_mV=("bias_V_mV", "median"),
                                   free_rmse_T_K=("rmse_T_K", "median")).reset_index()
    s = s.merge(f, on=GROUP, how="left")
    s.insert(0, "variant", name)
    s.insert(0, "cell", cell)
    s["select_in_fit"] = with_select
    return s


def run(cell, variants):
    # Fit, score and save every variant of one cell type, return its summary rows
    ocv = get_ocv(cell)
    summaries = []
    for name in variants:
        with_select = VARIANTS[name][2]
        if cell in NO_SELECT and (with_select or name.endswith(("E44.8", "E64.8"))):
            continue
        t0 = time.time()
        out = OUT / cell / name
        out.mkdir(parents=True, exist_ok=True)

        fit_data = dataset.load(cell, ["train", "select"] if with_select else ["train"])
        model = make_model(cell, name, ocv).fit(fit_data)
        del fit_data
        model.save(out / "ecm_params.json")
        print(f"{cell} {name}: fitted in {time.time() - t0:.0f} s  {model.params()}", flush=True)

        windows, freerun = score(cell, model)
        windows.to_parquet(out / "windows.parquet", index=False)
        freerun.to_csv(out / "freerun.csv", index=False)
        harness.summarise(windows, freerun).to_csv(out / "by_file.csv", index=False)

        s = summary_table(cell, name, with_select, windows, freerun)
        print(s.round(2).to_string(index=False), f"\n{cell} {name}: done in {time.time() - t0:.0f} s\n", flush=True)
        summaries.append(s)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=list(cfg.CELLS))
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
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