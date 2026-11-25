"""
Shared helpers for the exploratory analyses
These run after run_test.py, so the test files have already been scored once and nothing here is pre-registered
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.ecm import ECMModel
from warmtrained.models.ocv import OCVCurve

OUT = cfg.RESULTS / "exploratory"
MAIN_CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT"]  # cells with a selection temperature
HOR = int(harness.HORIZON_S)


def groups(cell):
    # (split, temperature) pairs: selection temps first (warm reference), then test temps
    s = cfg.SPLITS[cell]
    return [("select", t) for t in s["select"]] + [("test", t) for t in s["test"]]


def load(cell, split, t):
    # Data for one split and temperature, None if no usable file
    try:
        return dataset.load(cell, [split], temps=[t], allow_test=True)
    except ValueError:
        return None


def windows(g, every=1):
    # Scored windows (every k-th start): returns starts, predicted row indices, measured mask; None if none qualify
    V, meas = g.V.values, g.measured.values
    starts = harness.window_starts(len(g))[::every]
    if not len(starts):
        return None

    idx = starts[:, None] + np.arange(1, HOR + 1)
    ok = meas[idx] & np.isfinite(V[idx])
    keep = ok.sum(axis=1) >= harness.MIN_MEASURED
    if not keep.any():
        return None
    return starts[keep], idx[keep], ok[keep]


def arrhenius(T_c, E):
    # exp(-E * (1/T_ref - 1/T) / R), equals 1 at 25 degC
    theta = 1.0 / cfg.T_REF_K - 1.0 / (np.asarray(T_c, dtype=float) + 273.15)
    return np.exp(-E * theta / cfg.R_GAS)


def resistance_factor(ecm, T_c):
    # Total resistance R0 + R1 + R2 at T relative to 25 degC
    R25, E = ecm.p["R25"], ecm.p["E"]
    return sum(R25[k] * arrhenius(T_c, E[k]) for k in range(3)) / R25.sum()


def ocv_ecm(cell):
    # 25 degC OCV curve and the single_2T ECM built on it
    ocv = OCVCurve.load(cfg.RESULTS / "arx" / cell / "ocv.json")
    ecm = ECMModel.load(cfg.RESULTS / "ecm" / cell / "single_2T" / "ecm_params.json", ocv)
    return ocv, ecm


def lstm_files(cell):
    # Parameter files of the 5 seeds at the best grid size, [] if no LSTM for this cell
    folder = cfg.RESULTS / "lstm" / cell
    grid = folder / "grid.csv"
    if not grid.exists():
        return []

    g = pd.read_csv(grid).dropna(subset=["stop_loss"])
    if g.empty:
        return []
    best = g.loc[g.stop_loss.idxmin()]

    files = [folder / f"h{int(best.hidden)}_l{int(best.layers)}_s{seed}.json" for seed in range(5)]
    return [f for f in files if f.exists()]