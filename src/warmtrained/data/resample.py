"""
Putting every test on one 1 s grid
The cycler holds the current constant between rows and logs a row at each new step,
but the Digatron sometimes logs that row up to ~2 min late, so insert_step_starts adds a row at the true step start
Data logged faster than 1 s are averaged over each 1 s window (no aliasing of short spikes)
Data logged slower than 1 s are filled: current held, V, T and SOC interpolated in time
Every 1 s row says whether it holds at least one real sample (measured), metrics use measured rows only
A test is split into segments at real gaps, nothing is filled across a segment break
"""

import numpy as np
import pandas as pd

from warmtrained import config as cfg


def insert_step_starts(df: pd.DataFrame) -> pd.DataFrame:
    # Add a row at the true start of every step whose first logged row came more than 0.5 s late (Digatron step_time)
    # The new row carries the new step's current, V = T = NaN (interpolated later) and measured = False
    # Files without step_time pass through
    df = df.copy()
    df["measured"] = True
    if "step_time" not in df or df["step_time"].isna().all():
        return df

    t, step, st = df.t.values, df.step.values, df.step_time.values
    t_start = t - st
    late = np.r_[False, step[1:] != step[:-1]] & (st > 0.5)
    late[1:] &= t_start[1:] > t[:-1]  # the step must have begun after the previous logged row
    if not late.any():
        return df

    idx = np.flatnonzero(late)
    rows = df.iloc[idx].copy()
    rows["t"] = t_start[idx]
    rows[["V", "T"]] = np.nan
    rows["measured"] = False
    rows["step_time"] = 0.0
    return pd.concat([df, rows]).sort_values("t", kind="stable").reset_index(drop=True)


def unlogged_gaps(t: np.ndarray, V: np.ndarray, measured: np.ndarray) -> np.ndarray:
    # True on the first row after a gap between two real samples longer than GAP_SPLIT_S
    # with a voltage move above GAP_SPLIT_DV: the cell did something the log did not record
    m = np.flatnonzero(measured)
    flag = np.zeros(len(t), dtype=bool)
    if len(m) < 2:
        return flag
    gap = (np.diff(t[m]) > cfg.GAP_SPLIT_S) & (np.abs(np.diff(V[m])) > cfg.GAP_SPLIT_DV)
    flag[m[:-1][gap] + 1] = True
    return flag


def segments(t, I, V, measured, cut=None, min_gap_s: float = 5.0) -> np.ndarray:
    # Segment number per row, a new segment starts at:
    # 1) a gap longer than max(min_gap_s, 3 x median step) that is not bridgeable
    # 2) an unlogged gap (unlogged_gaps)
    # 3) a row where cut is True (rows just before were removed by the warm-up rule)
    # Bridgeable: both rows at rest and <= GAP_SPLIT_S apart, same current and <= 1 h apart, or <= 120 s apart
    dt = np.diff(t)
    if not len(dt):
        return np.zeros(len(t), dtype=int)

    gap = max(min_gap_s, 3.0 * float(np.median(dt)))
    rest = np.abs(I) < cfg.REST_CURRENT_A
    both_rest = rest[:-1] & rest[1:]
    same_current = np.abs(I[1:] - I[:-1]) <= np.maximum(0.02, 0.01 * np.abs(I[:-1]))
    bridgeable = (both_rest & (dt <= cfg.GAP_SPLIT_S)) | (same_current & (dt <= 3600.0)) | (dt <= 120.0)

    new = ((dt > gap) & ~bridgeable) | unlogged_gaps(t, V, measured)[1:]
    if cut is not None:
        new |= np.asarray(cut, dtype=bool)[1:]
    return np.r_[0, np.cumsum(new)]


def to_common_step(df: pd.DataFrame, step_s: float = cfg.RESAMPLE_S, align: str = "floor") -> pd.DataFrame:
    # Resample one test to a fixed step
    # Input: t, V, I, T, soc, measured (optional cut_before)
    # Output: t (bin centre), V, I, T, soc, segment, measured (+ chamber if present)
    # align="floor": bins [k, k+1) s, for logs sampled faster or slower than 1 s
    # align="round": bins centred on whole seconds, for data already on a 1 s grid with ~1 ms jitter (Tesla 2170)
    # Floor bins would split jittered stamps across edges (two samples in some bins, none in others, up to 0.7 % charge bias)
    # The grid is first shifted by the file's mean sub-second phase, so stamps near x.5 s round cleanly too
    df = df.copy()
    cut = df["cut_before"].values if "cut_before" in df else None
    df["segment"] = segments(df.t.values, df.I.values, df.V.values, df.measured.values, cut)

    phase = 0.0
    if align == "round" and len(df):
        # Circular mean of the sub-second phase: 0.999 s and 1.0003 s average to ~0, not 0.5
        t = df.t.values
        regular = np.r_[True, np.diff(t) >= step_s / 2]  # skip dense stretches (chamber ramps)
        phase = float(np.angle(np.mean(np.exp(2j * np.pi * t[regular] / step_s)))) * step_s / (2 * np.pi)

    scaled = (df.t.values - phase) / step_s
    df["bin"] = (np.rint(scaled) if align == "round" else np.floor(scaled)).astype(np.int64)

    extra = ["chamber"] if "chamber" in df else []  # measured chamber temperature (Panasonic), carried to the grid
    num = ["V", "T", "soc"] + extra
    cols = ["t", "V", "I", "T", "soc", "segment", "measured"] + extra

    parts = []
    for seg, g in df.groupby("segment", sort=True):
        bins = g.groupby("bin")
        full = np.arange(g["bin"].min(), g["bin"].max() + 1)

        out = bins[num + ["I"]].mean().reindex(full)  # 1 s window means (NaN V of inserted rows ignored)
        out[num] = out[num].interpolate(method="index", limit_area="inside")  # fill slow-logged stretches in time
        out["I"] = out["I"].fillna(bins["I"].last().reindex(full).ffill())  # empty bins hold the last logged current
        out["measured"] = bins["measured"].any().reindex(full, fill_value=False).values
        out["segment"] = seg

        if len(out) >= cfg.MIN_SEGMENT_S / step_s:  # drop fragments too short for any model step
            parts.append(out)

    if not parts:
        return pd.DataFrame(columns=cols)

    res = pd.concat(parts).rename_axis("bin").reset_index()
    res["t"] = (res["bin"] + (0.0 if align == "round" else 0.5)) * step_s + phase
    return res[cols]