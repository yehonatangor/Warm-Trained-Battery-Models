"""
Cleaning rules
Each rule returns a boolean mask of rows to keep (or a flag), so the pipeline can log what every rule removed
"""

import numpy as np
import pandas as pd

from fastcharge import config as cfg


# Helpers

def stretches(x):
    # (first row, end row) of every run of equal values, end row exclusive
    starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
    return list(zip(starts, np.r_[starts[1:], len(x)]))


def first(mask, start):
    # First row at or after start where mask is True, len(mask) if none
    hits = np.flatnonzero(mask[start:])
    return start + hits[0] if len(hits) else len(mask)


# Rules

def range_mask(df: pd.DataFrame, v_range=None) -> np.ndarray:
    # Rows inside believable voltage, temperature and current ranges
    # v_range overrides the default voltage window (LFP cells)
    v_lo, v_hi = v_range or cfg.VOLTAGE_RANGE_V
    t_lo, t_hi = cfg.TEMPERATURE_RANGE_C
    return (df.V.between(v_lo, v_hi)
            & df["T"].between(t_lo, t_hi)
            & (df.I.abs() <= cfg.MAX_ABS_CURRENT_A)).values


def interruption_mask(df: pd.DataFrame) -> np.ndarray:
    # Drop rows where the cycler status reports an operator or host interruption
    return ~df.status.astype(str).str.contains("Interr|BTS", regex=True).values


def duplicate_time_mask(df: pd.DataFrame) -> np.ndarray:
    # Keep a row only if its time is later than every earlier row
    # Drops repeated time stamps and rows written out of order (time jumping backwards)
    t = df.t.values
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    earlier_max = np.r_[-np.inf, np.maximum.accumulate(t)[:-1]]
    return t > earlier_max


def warmup_mask(df: pd.DataFrame, t_nominal: float) -> np.ndarray:
    # Drop room-temperature rests and charges
    # Each row is a discharge (-1), rest (0) or charge (1), a stretch is a run of one kind
    # In rest or charge stretches of at least WARMUP_MIN_SEGMENT_S, rows off the set point by more than WARMUP_BAND_K are dropped
    # Short stretches (HPPC pulses, regen, drive cycle pauses) and all discharge rows are kept, so cold self-heating stays
    t, I = df.t.values, df.I.values
    T = df["T"].bfill().values  # rows added at a step start have no temperature, use the next real one
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    kind = (I > cfg.REST_CURRENT_A).astype(int) - (I < -cfg.REST_CURRENT_A)
    off = np.abs(T - t_nominal) > cfg.WARMUP_BAND_K  # False where the temperature is missing
    runs = stretches(kind)

    keep = np.ones(len(df), dtype=bool)
    warm = np.zeros(len(runs), dtype=bool)  # long stretch that had rows dropped

    for k, (a, b) in enumerate(runs):
        end = t[b] if b < len(t) else t[-1] + dt  # start of the next stretch, last stretch: one more step
        if kind[a] != -1 and end - t[a] >= cfg.WARMUP_MIN_SEGMENT_S:
            keep[a:b] = ~off[a:b]
            warm[k] = off[a:b].any()

    # A short rest next to a dropped warm stretch (e.g. the last rows after a room-temperature charge)
    # belongs to that warm period, so its off-temperature rows go too
    padded = np.r_[False, warm, False]
    for k, (a, b) in enumerate(runs):
        if kind[a] == 0 and (padded[k] or padded[k + 2]):
            keep[a:b] &= ~off[a:b]

    return keep


def setpoint_mask(df: pd.DataFrame, t_nominal: float) -> np.ndarray:
    # Keep rows logged while the chamber set point was within WARMUP_BAND_K of the folder temperature
    # Tesla 2170: drops charges run at 25 degC and impossible set points (e.g. -1308 degC)
    # Rows with no logged set point are kept and left to the warm-up rule
    # A file whose set point channel was not recorded is kept whole (see setpoint_channel_valid)
    if not setpoint_channel_valid(df, t_nominal):
        return np.ones(len(df), dtype=bool)
    amb = df["amb"].values
    return ~(np.isfinite(amb) & (np.abs(amb - t_nominal) > cfg.WARMUP_BAND_K))


def soak_mask(df: pd.DataFrame, t_nominal: float) -> np.ndarray:
    # Used instead of the warm-up rule when the set point is logged
    # Each time the set point enters the folder band (file start, or back from a 25 degC charge),
    # drop every row until the cell first comes within WARMUP_BAND_K of the set point (cell not soaked yet)
    # Rest rows before the first load that drift out of the band are dropped too
    # After the first load, self-heating and the cooling rests after it are kept
    # No usable set point channel: chamber assumed at the folder temperature, so only the file start can be unsoaked
    keep = np.ones(len(df), dtype=bool)
    if len(df) == 0:
        return keep

    if setpoint_channel_valid(df, t_nominal):
        amb = pd.Series(df["amb"].values).ffill().bfill().values
        inband = np.abs(amb - t_nominal) <= cfg.WARMUP_BAND_K
    else:
        inband = np.ones(len(df), dtype=bool)

    soaked = np.abs(df["T"].values - t_nominal) <= cfg.WARMUP_BAND_K
    loaded = np.abs(df.I.values) > cfg.REST_CURRENT_A
    entries = np.flatnonzero(inband & ~np.r_[False, inband[:-1]])

    for e in entries:
        j = first(soaked, e)  # first soaked row
        keep[e:j] = False
        L = first(loaded, j)  # first load after soaking
        keep[j:L] &= soaked[j:L] | ~inband[j:L]  # rows off the set point are the set point rule's job

    return keep


def setpoint_channel_valid(df: pd.DataFrame, t_nominal: float) -> bool:
    # True if the file logs a set point and at least one value is within WARMUP_BAND_K of the folder temperature
    # A constant channel must also agree with the cell: no rest of 10 min or more may have a median
    # cell temperature more than 10 K from it (two m448-N files at 40 degC log 0 degC throughout)
    # A constant 0 degC can be a real set point at the 0 degC stage, so it is checked against the rests
    if "amb" not in df:
        return False

    amb = df["amb"].values
    logged = amb[np.isfinite(amb)]
    if len(logged) == 0 or not np.any(np.abs(logged - t_nominal) <= cfg.WARMUP_BAND_K):
        return False

    if np.all(logged == logged[0]):
        t, T = df.t.values, df["T"].values
        rest = np.abs(df.I.values) < cfg.REST_CURRENT_A
        for a, b in stretches(rest.astype(int)):
            long_rest = rest[a] and t[b - 1] - t[a] >= 600
            if long_rest and abs(np.nanmedian(T[a:b]) - logged[0]) > 10.0:
                return False

    return True


def label_mismatch(t_soaked: float, t_nominal: float) -> bool:
    # True when a test never started at its folder temperature:
    # the soaked rest temperature before the test (see clean_dataset.ambient) is more than LABEL_BAND_K off
    # Self-heating under load does not count, so a fast cold discharge that warms the cell is kept
    return bool(np.isfinite(t_soaked) and abs(t_soaked - t_nominal) > cfg.LABEL_BAND_K)