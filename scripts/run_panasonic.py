"""
Panasonic 18650PF (B6), pre-registered, scored once: a cell no design choice has touched

Models, all fitted on 25 and 10 degC by the usual scripts first (run_arx, run_ecm single_2T, run_sindy no_bv,
run_lstm), every setting taken from the other cells unchanged:
    arx, ecm, sindy (no_bv), lstm_s0 to lstm_s4 and lstm_median (per file, the seed with the median mean window RMSE)
Data: the test split (0, -10, -20 degC drive cycles) and the trise split (chamber warms during the drive)

Claim on the -20 degC test files, hypotheses' win rule (block 10, block 20 reported as sensitivity):
    P3  ecm beats lstm_median on >= 75 % of files
Logs its start in results/panasonic/RUN_LOG.txt and refuses a second run unless --force (logged as a reproduction)
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_test

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness, stats
from warmtrained.models.arx import ARXModel
from warmtrained.models.ecm import ECMModel
from warmtrained.models.ocv import OCVCurve
from warmtrained.models.sindy import SINDyModel
from run_test import median_seed

CELL = "Panasonic18650PF"
R = cfg.RESULTS
OUT = R / "panasonic"
T_CLAIM = -20
CLAIMS = {"P3": ("ecm", "lstm_median", 0.75)}  # name: (model A, model B, required win share)
GROUP = ["split", "T_nominal"]


def load_models():
    from warmtrained.models.lstm import LSTMModel  # torch only needed here
    ocv = OCVCurve.load(R / "arx" / CELL / "ocv.json")
    models = {"arx": ARXModel.load(R / "arx" / CELL / "arx_params.json", ocv),
              "ecm": ECMModel.load(R / "ecm" / CELL / "single_2T" / "ecm_params.json", ocv),
              "sindy": SINDyModel.load(R / "sindy" / CELL / "no_bv" / "sindy_params.json", ocv)}
    for s in range(5):
        models[f"lstm_s{s}"] = LSTMModel.load(R / "lstm" / CELL / f"h64_l2_s{s}.json", ocv)
    return models


def score(models):
    # Windows and free runs of every model on the test and trise splits
    W, F = {k: [] for k in models}, {k: [] for k in models}
    for split in ("test", "trise"):
        for t in cfg.SPLITS[CELL][split]:
            try:
                data = dataset.load(CELL, [split], temps=[t], allow_test=True)
            except ValueError:
                continue
            if data.empty:
                continue
            for k, m in models.items():
                w, f = harness.evaluate(m, data)
                W[k].append(w)
                F[k].append(f)
            print(f"{split} {t} degC scored", flush=True)
            del data
    windows = {k: pd.concat(v, ignore_index=True) for k, v in W.items()}
    free = {k: pd.concat(v, ignore_index=True) for k, v in F.items()}
    windows["lstm_median"] = median_seed({k: w for k, w in windows.items() if k.startswith("lstm_s")})
    return windows, free


def summary_table(windows, free):
    # One row per model, split and temperature, windows saved per model
    rows = []
    for k, w in windows.items():
        w.to_parquet(OUT / f"{k}_windows.parquet", index=False)
        s = w.groupby(GROUP).agg(files=("file", "nunique"), windows=("rmse_V_mV", "size"),
                                 rmse_V_mV=("rmse_V_mV", "mean"), rmse_T_K=("rmse_T_K", "mean")).reset_index()
        if k in free:
            fr = free[k].groupby(GROUP).agg(free_rmse_V_mV=("rmse_V_mV", "median")).reset_index()
            s = s.merge(fr, on=GROUP, how="left")
        rows.append(s.assign(model=k))
    return pd.concat(rows, ignore_index=True)


def claims_table(windows):
    # Per-file comparisons and verdicts of every claim and block
    claim_rows, file_rows = [], []
    files = set(windows["ecm"].query("split == 'test' and T_nominal == @T_CLAIM").file)
    for name, (ma, mb, need) in CLAIMS.items():
        for block in (10, 20):
            r = stats.compare_models(windows[ma], windows[mb], files, block=block)
            file_rows.append(r.assign(claim=name, model_a=ma, model_b=mb, block=block))
            share = float(r.a_wins.mean()) if len(r) else np.nan
            claim_rows.append(dict(claim=name, model_a=ma, model_b=mb, block=block, files=len(r),
                                   wins=int(r.a_wins.sum()) if len(r) else 0, share=share, required=need,
                                   supported=bool(share >= need)))
    pd.concat(file_rows, ignore_index=True).to_csv(OUT / "claims_files.csv", index=False)
    return pd.DataFrame(claim_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun the frozen models (a reproduction)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "RUN_LOG.txt"
    if log.exists() and not a.force:
        sys.exit(f"{log} exists: the Panasonic test has been run. Pass --force only to reproduce it.")
    with open(log, "a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} "
                 f"{'reproduction (--force)' if a.force else 'first run'}\n")

    windows, free = score(load_models())
    summary = summary_table(windows, free)
    summary.to_csv(OUT / "summary.csv", index=False)
    claims = claims_table(windows)
    claims.to_csv(OUT / "claims.csv", index=False)

    print(summary.pivot_table(index=GROUP, columns="model", values="rmse_V_mV").round(1).to_string())
    print(claims.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()