"""
One evaluation harness for every model, scored on measured rows only
1. Window score: 20 min open-loop windows started every 2 min, each after 5 min of warm-up and ending inside
   the same piece. The model may use measured warm-up data, then runs on current, SOC and ambient temperature only
   A window is scored if it holds at least MIN_MEASURED measured rows: voltage RMSE [mV], temperature RMSE [K]
2. Free run: whole piece simulated from its first row, voltage RMSE split into bias (mean error) and
   shape (error std), so an offset can be told apart from wrong dynamics

Every model implements:
    predict_windows(piece, starts, horizon) -> (V_pred, T_pred), shape (len(starts), horizon),
        row j of the window starting at index s predicts row s + 1 + j
    free_run(piece) -> (V_pred, T_pred), the length of the piece
"""

import numpy as np
import pandas as pd

WARMUP_S = 300  # measured data the model may use before a window [s]
HORIZON_S = 1200  # window length [s]
STRIDE_S = 120  # time between window starts [s]
MIN_MEASURED = 60  # measured rows a score needs

KEYS = ["file", "unit", "T_nominal", "type", "split"]  # one test file


# Helpers

def window_starts(n_rows, step_s=1.0):
    # Start index (last warm-up row) of every window that fits in a piece of n_rows rows
    warm, hor, stride = int(WARMUP_S / step_s), int(HORIZON_S / step_s), int(STRIDE_S / step_s)
    last = n_rows - 1 - hor
    if last < warm:
        return np.zeros(0, dtype=int)
    return np.arange(warm, last + 1, stride)


def rmse_rows(err, mask):
    # RMSE of each row over its masked entries, NaN where none
    n = mask.sum(axis=1)
    sq = np.where(mask, err ** 2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sq / n)


def piece_meta(g, piece_id):
    # Test info repeated on every score row
    return dict(file=g.file.iat[0], unit=g["unit"].iat[0], T_nominal=g.T_nominal.iat[0], type=g.type.iat[0],
                split=g.split.iat[0], piece=piece_id, T_amb=g.T_amb.iat[0])


# Scores

def free_run_score(model, g, meta):
    # Free-run scores of one piece, None if it has too few measured rows
    V, T, meas = g.V.values, g["T"].values, g.measured.values
    Vf, Tf = model.free_run(g)

    m = meas & np.isfinite(V)
    if m.sum() < MIN_MEASURED:
        return None

    e = (Vf - V)[m] * 1e3  # voltage error [mV]
    mt = meas & np.isfinite(T)
    rmse_T = float(np.sqrt(np.mean((Tf - T)[mt] ** 2))) if mt.any() else np.nan
    return dict(meta, n_measured=int(m.sum()), rmse_V_mV=float(np.sqrt(np.mean(e ** 2))),
                bias_V_mV=float(np.mean(e)), shape_V_mV=float(np.std(e)), rmse_T_K=rmse_T)


def window_scores(model, g, meta, step_s):
    # Scores of every scorable window in one piece
    hor = int(HORIZON_S / step_s)
    V, T, meas = g.V.values, g["T"].values, g.measured.values

    starts = window_starts(len(g), step_s)
    if len(starts) == 0:
        return []

    idx = starts[:, None] + np.arange(1, hor + 1)[None, :]  # rows each window predicts
    mv = meas[idx] & np.isfinite(V[idx])
    keep = mv.sum(axis=1) >= MIN_MEASURED
    if not keep.any():
        return []
    starts, idx, mv = starts[keep], idx[keep], mv[keep]

    Vp, Tp = model.predict_windows(g, starts, hor)
    err_V = (Vp - V[idx]) * 1e3  # [mV]
    mt = meas[idx] & np.isfinite(T[idx])
    rv = rmse_rows(err_V, mv)
    rt = rmse_rows(Tp - T[idx], mt)
    with np.errstate(invalid="ignore"):
        bias = np.where(mv, err_V, 0.0).sum(axis=1) / mv.sum(axis=1)

    return [dict(meta, t_start=float(g.t.iat[s]), n_measured=int(mv[k].sum()),
                 rmse_V_mV=float(rv[k]), bias_V_mV=float(bias[k]), rmse_T_K=float(rt[k]))
            for k, s in enumerate(starts)]


def evaluate(model, df, step_s=1.0):
    # Score a model on a table from dataset.load
    # Returns (windows, freerun): one row per scored window, one row per piece
    win_rows, free_rows = [], []
    for piece_id, g in df.groupby("piece", sort=False):
        g = g.reset_index(drop=True)
        meta = piece_meta(g, piece_id)

        free = free_run_score(model, g, meta)
        if free is not None:
            free_rows.append(free)
        win_rows += window_scores(model, g, meta, step_s)

    return pd.DataFrame(win_rows), pd.DataFrame(free_rows)


def summarise(windows, freerun):
    # One row per test file: mean window RMSE (headline score), window count,
    # and free-run scores pooled over the file's pieces (weighted by measured rows)
    w = None
    if len(windows):
        w = windows.groupby(KEYS).agg(n_windows=("rmse_V_mV", "size"),
                                      window_rmse_V_mV=("rmse_V_mV", "mean"),
                                      window_rmse_T_K=("rmse_T_K", "mean")).reset_index()

    f = None
    if len(freerun):
        f = freerun.assign(sq=freerun.rmse_V_mV ** 2 * freerun.n_measured,
                           bw=freerun.bias_V_mV * freerun.n_measured)
        f = f.groupby(KEYS).agg(n=("n_measured", "sum"), sq=("sq", "sum"), bw=("bw", "sum")).reset_index()
        f["free_rmse_V_mV"] = np.sqrt(f.sq / f.n)
        f["free_bias_V_mV"] = f.bw / f.n
        f = f[KEYS + ["free_rmse_V_mV", "free_bias_V_mV"]]

    if w is not None and f is not None:
        out = w.merge(f, on=KEYS, how="outer")
    else:
        out = w if f is None else f
    return out.sort_values(["split", "T_nominal", "file"]).reset_index(drop=True)