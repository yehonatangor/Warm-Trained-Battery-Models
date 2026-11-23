"""
Runs the single test run and scores hypotheses H1 to H4

Loads every fitted model, scores it on each cell's test split with the same harness, then gives the verdicts of
H1 to H4 on the -20 degC drive-cycle test files of B1, B3 and B5, nothing is refitted
B4 (35 and 5 degC), B2 (0 degC) and the simulated cell (30, 35, 0, -10 degC) are scored the same way, no hypothesis
If ensemble SINDy fits exist, the coverage and width of its 5 to 95 % band are computed on the test files too

Models: ARX, ECM single_2T, single_3T, two3T_E54.8 (and the E2 sensitivity fits if present), SINDy full and no_bv,
LSTM all five seeds (median LSTM seed of H3: per file, the seed whose mean window RMSE is the median of the five)
A model whose files are missing is skipped and reported

Hypotheses (A wins against B on a file per warmtrained.evaluation.stats):
    H1  SINDy full against ECM single_2T: wins on >= 75 % of the files of each cell type
    H2  SINDy full against SINDy no_bv: >= 75 %
    H3  ECM single_2T against the median LSTM seed: all files of B1 and B3, >= 90 % of B5
    H4  ECM two3T_E54.8 against ECM single_3T (both fitted on 40, 25, 10 degC): >= 75 %
Block 10 is the main rule, block 20 the sensitivity case

Runs once: writes results/test/RUN_LOG.txt and refuses to start again unless --again is given (reruns are logged)
--dry-run scores the selection split instead into results/test_dry/ with mock verdicts on the 10 degC files
(not evidence), so the pipeline can be checked without touching test data
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_sindy_ensemble

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness, stats
from warmtrained.models.arx import ARXModel
from warmtrained.models.ecm import ECMModel
from warmtrained.models.ocv import OCVCurve
from warmtrained.models.sindy import SINDyModel

CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50", "SimLGM50"]
HYP_CELLS = ["Samsung30T", "LGHG2", "Tesla2170"]
HYP_T = -20
ECM_VARIANTS = ["single_2T", "single_3T", "two3T_E54.8", "two3T_E44.8", "two3T_E64.8"]
NOT_DRIVE = {"HPPC", "Charge", "Charge 1C", "C/20", "C/20 at 40 degC", "CC discharge", "Capacity 1C", "Other",
             "Pause cycle", "Mixed cycle", "LG M50 rate test", "C/20 charge"}
HYPOTHESES = {  # name: (model A, model B, required win share per cell type)
    "H1": ("sindy_full", "ecm_single_2T", {c: 0.75 for c in HYP_CELLS}),
    "H2": ("sindy_full", "sindy_no_bv", {c: 0.75 for c in HYP_CELLS}),
    "H3": ("ecm_single_2T", "lstm_median", {"Samsung30T": 1.0, "LGHG2": 1.0, "Tesla2170": 0.90}),
    "H4": ("ecm_two3T_E54.8", "ecm_single_3T", {c: 0.75 for c in HYP_CELLS}),
}


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def load_models(cell):
    # Every fitted model of one cell that exists, and the names of the missing ones
    R = cfg.RESULTS
    ocv = OCVCurve.load(R / "arx" / cell / "ocv.json")
    files = {"arx": (ARXModel, R / "arx" / cell / "arx_params.json")}
    files.update({f"ecm_{v}": (ECMModel, R / "ecm" / cell / v / "ecm_params.json") for v in ECM_VARIANTS})
    files.update({f"sindy_{v}": (SINDyModel, R / "sindy" / cell / v / "sindy_params.json") for v in ("full", "no_bv")})

    grid = R / "lstm" / cell / "grid.csv"
    if grid.exists():
        from warmtrained.models.lstm import LSTMModel  # torch only needed when LSTM results exist
        g = pd.read_csv(grid)
        best = g.loc[g.stop_loss.idxmin()]
        files.update({f"lstm_s{s}": (LSTMModel, R / "lstm" / cell / f"h{int(best.hidden)}_l{int(best.layers)}_s{s}.json")
                      for s in range(5)})

    models, missing = {}, []
    for k, (cls, f) in files.items():
        if f.exists():
            models[k] = cls.load(f, ocv)
        else:
            missing.append(k)
    if not grid.exists():
        missing.append("lstm")
    return models, missing


def test_temps(cell, dry):
    if dry:
        return ("train", [10]) if cell == "LGM50" else ("select", cfg.SPLITS[cell]["select"])
    return "test", cfg.SPLITS[cell]["test"]


def median_seed(lstm):
    # Per file, the windows of the seed whose mean window RMSE is the median of the seeds
    all_w = pd.concat([w.assign(seed=k) for k, w in lstm.items()], ignore_index=True)
    per = all_w.groupby(["file", "seed"]).rmse_V_mV.mean().reset_index()
    pick = per.sort_values("rmse_V_mV").groupby("file").apply(lambda d: d.seed.iloc[len(d) // 2],
                                                              include_groups=False)
    return pd.concat([all_w[(all_w.file == f) & (all_w.seed == s)] for f, s in pick.items()], ignore_index=True)


def score_cell(cell, out, dry):
    # Score every model of one cell, returns (windows per model, summary rows)
    models, missing = load_models(cell)
    if not models:
        print(f"{cell}: no models to score; missing {missing}", flush=True)
        return {}, pd.DataFrame()
    split, temps = test_temps(cell, dry)
    print(f"{cell}: {len(models)} models on {split} {temps}; missing {missing}", flush=True)

    ens = None
    if (cfg.RESULTS / "sindy_ensemble" / cell / "fits").exists():
        from run_sindy_ensemble import band_table, ensemble_models
        ens = ensemble_models(cell)[0]

    parts_w, parts_f, band = {k: [] for k in models}, {k: [] for k in models}, []
    for t in temps:
        data = dataset.load(cell, [split], temps=[t], allow_test=not dry)
        for k, m in models.items():
            w, f = harness.evaluate(m, data)
            parts_w[k].append(w)
            parts_f[k].append(f)
        if ens:
            band.append(band_table(ens, data))
        del data
        print(f"  {cell} {t} degC scored", flush=True)

    (out / cell).mkdir(parents=True, exist_ok=True)
    windows, rows = {}, []
    for k in models:
        w, f = pd.concat(parts_w[k], ignore_index=True), pd.concat(parts_f[k], ignore_index=True)
        w.to_parquet(out / cell / f"{k}_windows.parquet", index=False)
        f.to_csv(out / cell / f"{k}_freerun.csv", index=False)
        windows[k] = w
        s = w.groupby("T_nominal").agg(files=("file", "nunique"), windows=("rmse_V_mV", "size"),
                                       rmse_V_mV=("rmse_V_mV", "mean"), rmse_T_K=("rmse_T_K", "mean"))
        fr = f.groupby("T_nominal").agg(free_rmse_V_mV=("rmse_V_mV", "median"), free_bias_V_mV=("bias_V_mV", "median"))
        rows.append(s.join(fr).reset_index().assign(cell=cell, model=k))
    if band:
        pd.concat(band, ignore_index=True).to_csv(out / cell / "ensemble_band.csv", index=False)

    lstm = {k: w for k, w in windows.items() if k.startswith("lstm_s")}
    if lstm:
        windows["lstm_median"] = median_seed(lstm)
    return windows, pd.concat(rows, ignore_index=True)


def verdicts(all_windows, out, t_hyp=HYP_T):
    # Per-file comparisons and verdicts of every hypothesis, cell and block
    file_rows, verdict_rows = [], []
    for h, (a, b, need) in HYPOTHESES.items():
        for cell in HYP_CELLS:
            W = all_windows.get(cell, {})
            if a not in W or b not in W:
                verdict_rows.append(dict(hypothesis=h, cell=cell, model_a=a, model_b=b, status="model missing"))
                continue
            wa, wb = W[a], W[b]
            files = set(wa[(wa.T_nominal == t_hyp) & ~wa.type.isin(NOT_DRIVE)].file)
            for block in (10, 20):
                r = stats.compare_models(wa, wb, files, block=block)
                file_rows.append(r.assign(hypothesis=h, cell=cell, model_a=a, model_b=b, block=block))
                share = float(r.a_wins.mean()) if len(r) else np.nan
                verdict_rows.append(dict(hypothesis=h, cell=cell, model_a=a, model_b=b, block=block, files=len(r),
                                         wins=int(r.a_wins.sum()) if len(r) else 0, share=share, required=need[cell],
                                         supported=bool(share >= need[cell]) if len(r) else None, status="ok"))
    pd.concat(file_rows, ignore_index=True).to_csv(out / "hypotheses_files.csv", index=False)
    v = pd.DataFrame(verdict_rows)
    v.to_csv(out / "hypotheses.csv", index=False)
    print(v.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=CELLS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--again", action="store_true")
    a = ap.parse_args()
    out = cfg.RESULTS / ("test_dry" if a.dry_run else "test")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "RUN_LOG.txt"

    if not a.dry_run:
        if log.exists() and not a.again:
            raise SystemExit(f"the test run has already started ({log}); pass --again on purpose to rerun")
        with open(log, "a") as fh:
            fh.write(f"{now()} test run started, cells {a.cells}\n")

    all_windows, summaries = {}, []
    for c in a.cells:
        all_windows[c], s = score_cell(c, out, a.dry_run)
        summaries.append(s)
    pd.concat(summaries, ignore_index=True).to_csv(out / "summary.csv", index=False)

    if a.dry_run:  # exercises the verdict code on the 10 degC selection files, not evidence
        verdicts(all_windows, out, t_hyp=10)
    else:
        verdicts(all_windows, out)
        with open(log, "a") as fh:
            fh.write(f"{now()} test run finished\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()