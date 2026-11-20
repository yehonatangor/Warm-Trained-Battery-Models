"""
State of charge by coulomb counting
SOC(t) = SOC(t_a) + (Q(t) - Q(t_a)) / Q_ref, Q = charge flowed in (charge positive), t_a = anchor where the cell is full
Anchors are only full states reached at ANCHOR_MIN_T_C or warmer, since a cold CV hold ends before the cell is full:
- row 0 of a test that starts full (McMaster tests are fully charged at room temperature before every test,
  TU Berlin drive pieces start after a CV hold and a 2 h rest), accepted when V >= 4.05 V and |I| <= C/10
- the last row of a CV hold at >= 4.19 V whose current tapered below 2 x C/20 and whose cell is at least ANCHOR_MIN_T_C
TU Berlin lowers the temperature limit (its C/50 CV ends fall at most 0.4 % SOC short of full at 5 degC)
Rows before the first anchor are counted backwards from it
Training files: the closure error at each new anchor is spread linearly back to the previous one (no SOC jumps)
Select and test files: the count only runs forward
After an unlogged gap SOC is NaN until the next anchor
Q_ref is measured warm, never at a test temperature: a 25 degC slow-discharge capacity per cell type,
the 25 degC 1C capacity (TU Berlin), or each Tesla 2170 cell's 40 degC C/20 capacity per stage (clean_dataset.stage_capacity)
"""

import numpy as np
import pandas as pd

from fastcharge import config as cfg

START_FULL_V = 4.05
START_FULL_UNDER_LOAD_V = 3.95
CV_V = 4.19
ANCHOR_MIN_T_C = 15.0


def charge_integral(df: pd.DataFrame) -> np.ndarray:
    # Charge flowed in since the first row [Ah]
    # Between two rows the current is the earlier row's value (the cycler holds it until it logs a change)
    # Steps longer than MAX_DT_COUNT_S are capped
    dt = np.clip(np.diff(df.t.values), 0.0, cfg.MAX_DT_COUNT_S)
    return np.r_[0.0, np.cumsum(df.I.values[:-1] * dt)] / 3600.0


def full_anchors(df: pd.DataFrame, nominal_ah: float, start_is_full: bool, t_nominal: float,
                 min_T_c: float = ANCHOR_MIN_T_C, start_under_load: bool = False, volts=None) -> np.ndarray:
    # Row indices where the cell is known to be full
    # Missing cell temperature (LG M50 cell 791): the chamber set point decides if a CV end is warm enough
    # min_T_c can be lowered for a cell whose cold CV end is shown to be full (TU Berlin)
    # start_under_load: Panasonic drive files whose first row (0.1 s in) is already under load after a full charge,
    # row 0 is accepted at any current if V >= START_FULL_UNDER_LOAD_V (IR drop at -20 degC and 2.8 A)
    # volts: (start full, start full under load, CV) thresholds for another chemistry (LFP)
    V, I, T = df.V.values, df.I.values, df["T"].values
    v_start, v_load, v_cv = volts or (START_FULL_V, START_FULL_UNDER_LOAD_V, CV_V)

    rested_full = V[0] >= v_start and abs(I[0]) <= nominal_ah / 10
    loaded_full = start_under_load and V[0] >= v_load
    start = [0] if start_is_full and (rested_full or loaded_full) else []

    # Last row of every CV hold, kept if tapered and warm
    in_cv = (V >= v_cv) & (I > 0)
    ends = np.flatnonzero(in_cv[:-1] & ~in_cv[1:])
    temp = np.where(np.isfinite(T[ends]), T[ends], t_nominal)
    cv_ends = ends[(I[ends] <= 2 * nominal_ah / 20) & (temp >= min_T_c)]

    return np.unique(np.r_[start, cv_ends]).astype(int)


def coulomb_count(df: pd.DataFrame, q_ref_ah: float, nominal_ah: float, t_nominal: float, start_is_full: bool = True,
                  spread_closure: bool = False, start_soc: float | None = None, gap_rows=None,
                  min_T_c: float = ANCHOR_MIN_T_C, start_under_load: bool = False, volts=None):
    # Returns (soc array, anchor indices, largest closure error)
    # start_soc given (LG M50 logs chained from the previous temperature): count starts there, no anchors
    # gap_rows marks the first row after an unlogged gap, SOC is NaN from there until the next anchor
    q = charge_integral(df)
    t = df.t.values
    n = len(df)

    if start_soc is not None:
        soc = start_soc + (q - q[0]) / q_ref_ah
        anchors = np.array([], dtype=int)
        closure = 0.0
    else:
        anchors = full_anchors(df, nominal_ah, start_is_full, t_nominal, min_T_c, start_under_load, volts)
        if len(anchors) == 0:
            return np.full(n, np.nan), anchors, np.nan

        soc = 1.0 + (q - q[anchors[0]]) / q_ref_ah  # also counts rows before the first anchor backwards
        errors = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            err = soc[b] - 1.0  # how far the count from anchor a misses full at anchor b
            gap_between = gap_rows is not None and bool(np.any(gap_rows[a + 1:b + 1]))

            if spread_closure and not gap_between:
                soc[a:b + 1] -= err * (t[a:b + 1] - t[a]) / (t[b] - t[a])
                soc[b + 1:] -= err
            else:
                soc[b:] -= err  # forward only: re-anchor at b

            if not gap_between:
                errors.append(err)
        closure = max(errors, key=abs) if errors else 0.0

    if gap_rows is not None and np.any(gap_rows):
        # A backward count from the first anchor cannot cross an unlogged gap
        first_gap = int(np.argmax(gap_rows))
        if len(anchors) and first_gap <= anchors[0]:
            soc[:first_gap] = np.nan

        # From each gap until the next anchor SOC is unknown: gaps = 1, anchors = 2, carry the last mark forward
        mark = np.full(n, np.nan)
        mark[gap_rows] = 1.0
        mark[anchors] = 2.0
        soc[pd.Series(mark).ffill().values == 1.0] = np.nan

    return soc, anchors, float(closure)


def slow_discharge_capacity(df: pd.DataFrame, i_max_a: float) -> float:
    # Charge removed in the first continuous slow discharge of a test [Ah]
    I = df.I.values
    dis = (I < -cfg.REST_CURRENT_A) & (I > -i_max_a)
    if not dis.any():
        return float("nan")

    a = int(np.argmax(dis))
    after = np.flatnonzero(~dis[a:])
    b = a + after[0] if len(after) else len(df)

    dt = np.clip(np.diff(df.t.values[a:b + 1]), 0, cfg.MAX_DT_COUNT_S)  # one fewer step if the file ends discharging
    return float(-np.sum(I[a:a + len(dt)] * dt) / 3600.0)