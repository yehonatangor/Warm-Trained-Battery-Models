"""
Paired comparison of two models on one test file, and the win rule of the frozen hypotheses
Both models are scored on the same 20 min windows, d_k = RMSE_B,k - RMSE_A,k (positive when A is better)
Windows overlap (20 min every 2 min), so the 95 % interval of the mean d comes from a moving-block bootstrap
A wins when the whole interval is above zero (files with at least MIN_WINDOWS windows),
or when the mean d exceeds MARGIN_MV (shorter files, where a bootstrap is unreliable)
"""

import numpy as np
import pandas as pd

MIN_WINDOWS = 30  # fewest windows for the bootstrap rule
MARGIN_MV = 5.0  # win margin for shorter files [mV]
BLOCK = 10  # consecutive windows per bootstrap block
N_BOOT = 2000  # bootstrap repeats


def block_bootstrap_mean(d, block=BLOCK, n_boot=N_BOOT, seed=0):
    # Mean of d and its 95 % moving-block bootstrap interval (2.5 and 97.5 percentiles)
    d = np.asarray(d, dtype=float)
    n = len(d)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)

    # Each repeat draws k block starts, each block covers block consecutive windows
    k = int(np.ceil(n / block))
    starts = rng.choice(np.arange(n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)

    means = d[idx[:, :n]].mean(axis=1)  # trim each repeat to the series length
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compare_file(wa, wb, block=BLOCK, seed=0):
    # Paired comparison of model A (wa) against model B (wb), window tables of one file from the harness
    key = ["piece", "t_start"]
    m = wa[key + ["rmse_V_mV"]].merge(wb[key + ["rmse_V_mV"]], on=key, suffixes=("_a", "_b"))
    m = m.dropna().sort_values(key)
    d = (m.rmse_V_mV_b - m.rmse_V_mV_a).values
    n = len(d)

    if n == 0:
        return dict(n_windows=0, mean_diff_mV=np.nan, lo_mV=np.nan, hi_mV=np.nan, rule="none", a_wins=False)

    if n >= MIN_WINDOWS:
        mean, lo, hi = block_bootstrap_mean(d, block, seed=seed)
        return dict(n_windows=n, mean_diff_mV=mean, lo_mV=lo, hi_mV=hi,
                    rule=f"bootstrap_block{block}", a_wins=bool(lo > 0))

    mean = float(d.mean())
    return dict(n_windows=n, mean_diff_mV=mean, lo_mV=np.nan, hi_mV=np.nan,
                rule=f"margin_{MARGIN_MV}mV", a_wins=bool(mean > MARGIN_MV))


def compare_models(wa, wb, files=None, block=BLOCK):
    # compare_file for every file both models scored (optionally only files), one row per file
    # The seed is the file's position among all shared files, so a subset gives the same result per file
    rows = []
    shared = sorted(set(wa.file) & set(wb.file))
    for k, f in enumerate(shared):
        if files is not None and f not in files:
            continue
        r = compare_file(wa[wa.file == f], wb[wb.file == f], block, seed=k)
        rows.append(dict(file=f, **r))
    return pd.DataFrame(rows)