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
Validation and test files: the count only runs forward
After an unlogged gap SOC is NaN until the next anchor
Q_ref is measured warm, never at a test temperature: a 25 degC slow-discharge capacity per cell type,
the 25 degC 1C capacity (TU Berlin), or each Tesla 2170 cell's 40 degC C/20 capacity per stage (stage_capacity)
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import io

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

# Reference capacity (Q_ref), never measured at a test temperature

TESLA_STAGES = [40, 25, 10, 0, -10, -20]  # B5 test order, hot to cold


def file_time(name):
    # Start time in a B5 file name (MM-DD-YY_HH.MM), None if there is none
    m = re.search(r"(\d\d-\d\d-\d\d)_(\d\d)\.(\d\d)", name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}", "%m-%d-%y %H:%M")


def longest_slow_discharge(d, nominal_ah):
    # Charge removed in the longest continuous slow discharge [Ah]
    # C/30 or C/22 on these small LFP cells is below REST_CURRENT_A, so a lower threshold is used
    I = d.I.values
    dis = (I < -0.005) & (I > -0.2 * nominal_ah)
    edges = np.flatnonzero(np.diff(np.r_[0, dis.astype(int), 0]))
    a, b = max(zip(edges[::2], edges[1::2]), key=lambda ab: ab[1] - ab[0])
    dt = np.clip(np.diff(d.t.values[a:b]), 0, cfg.MAX_DT_COUNT_S)
    return float(-np.sum(I[a:b - 1] * dt) / 3600.0)


def reference_capacity(cell, meta):
    # B1, B3, B6, B7: the 25 degC C/20 discharge
    # B2: median 0.1C discharge at 25 degC
    # B4: median 1C discharge of the three 25 degC capacity tests (the only capacity tests in the set)
    # B5: {"unit|T": {"Ah", "start"}}, each cell's 40 degC C/20 discharge at the start of each stage (see stage_capacity)
    # B8, B9: slow discharge of the 25 degC OCV test (B8: C/30 script 1, B9: each cell's low-current test)
    root = cfg.DATA_RAW / meta["root"]
    nominal = meta["nominal_ah"]
    slow = 0.2 * nominal  # largest current of a slow discharge

    if cell in ("A123_26650", "CALCE_A123"):
        files = [f for f in sorted(root.glob(meta["q_ref_glob"])) if not f.name.startswith("~$")]
        return float(np.median([longest_slow_discharge(io.read_arbin(f), nominal) for f in files]))

    if cell == "Panasonic18650PF":
        f = next(root.glob(meta["q_ref_glob"]))
        return slow_discharge_capacity(io.read_panasonic_mat(f), i_max_a=slow)

    if cell == "Tesla2170":
        caps = {}
        for f in sorted(root.glob("*/*/*/2_40C_Discharge*.mat")):
            unit, stage = f.parts[-4], int(f.parts[-3][:-1])  # stage folder "40C" -> 40
            ah = slow_discharge_capacity(io.read_tesla_mat(f), i_max_a=slow)
            caps[f"{unit}|{stage}"] = {"Ah": ah, "start": file_time(f.name).isoformat()}
        return caps

    if cell == "TUBerlinM50LT":
        caps = {}
        for f in sorted(root.glob(meta["capacity_glob"])):
            d = io.read_basytec_txt(f)
            caps[f.name[:6].upper()] = slow_discharge_capacity(d, i_max_a=1.5 * nominal)  # 1C
        return {"Ah": float(np.median(list(caps.values()))), "per_cell_1C_Ah": caps}

    folder = root / meta["folders"][25]

    if cell == "LGM50":
        caps = []
        for f in sorted(folder.glob("*_0p1C_*.csv")):
            d = io.read_basytec(f)
            if len(d) <= 100:
                continue
            dis = d[(d.I < -0.3) & (d.I > -0.7)]  # 0.1C = 0.5 A
            if len(dis):
                caps.append(float(dis.Ah.max() - dis.Ah.min()) if dis.Ah.notna().any() else np.nan)
        return float(np.nanmedian(caps))

    f = next(folder.glob("*C20*.csv"))
    return slow_discharge_capacity(io.read_digatron(f), i_max_a=slow)


def stage_capacity(caps, unit, t_nom, name, split):
    # B5 capacity for one test
    # Test stages use the capacity measured at the start of their own stage (only earlier information)
    # Training and validation stages interpolate in time between the start of their stage and the next one,
    # since capacity falls by up to 5 % within a stage (25 to 10 degC)
    here = caps.get(f"{unit}|{t_nom}")
    if here is None:
        return float("nan")

    k = TESLA_STAGES.index(t_nom)
    nxt = caps.get(f"{unit}|{TESLA_STAGES[k + 1]}") if k + 1 < len(TESLA_STAGES) else None
    t = file_time(name)
    if split == "test" or nxt is None or t is None:
        return here["Ah"]

    t0, t1 = datetime.fromisoformat(here["start"]), datetime.fromisoformat(nxt["start"])
    w = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return here["Ah"] + w * (nxt["Ah"] - here["Ah"])


def test_q_ref(q_ref, cell, unit, t_nom, name, split):
    # One capacity for this test (B5: per stage, B4: the median of its dict)
    if isinstance(q_ref, dict) and cell == "Tesla2170":
        return stage_capacity(q_ref, unit, t_nom, name, split)
    if isinstance(q_ref, dict):
        return q_ref["Ah"]
    return q_ref