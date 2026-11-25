"""
Independent checks on the cleaned data, run after scripts/clean_dataset.py
Each check prints PASS or FAIL with the number it measured
The checks recompute quantities from the raw logs or from the data owners' own columns,
so they do not just repeat the cleaning code
"""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import clean, io, resample, tuberlin

PROC = cfg.DATA_PROCESSED
TESLA_DRIVES = ("REORDERED", "HWCUST", "HWGRADE")
OFF_BAND_K = cfg.WARMUP_BAND_K + 0.5  # off-temperature limit for the checks, a little looser than the cleaning band
results = []  # True or False for every check


# Helpers

def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load(r):
    # Processed 1 s table of one manifest row
    return pd.read_parquet(PROC / r.cell / str(r.T_nominal) / (Path(r.file).stem + ".parquet"))


def fingerprint(df):
    # Hash of V, I and T, the same test stored twice gives the same hash
    return hashlib.sha1(np.round(df[["V", "I", "T"]].to_numpy(dtype=float), 6).tobytes()).hexdigest()


def off_temperature(g, t_nom):
    # True if any measured row sits more than OFF_BAND_K from the set point
    return bool((g.measured & ((g["T"] - t_nom).abs() > OFF_BAND_K)).any())


@lru_cache(maxsize=2)
def b4_pieces(path, unit):
    # All pieces of one B4 log, read once and reused for its eight tests
    return {s: p for s, _, p in tuberlin.split_log(io.read_basytec_txt(Path(path)), unit)}


def tesla_raw_path(r):
    # Raw .mat file of one B5 test
    meta = cfg.CELLS["Tesla2170"]
    pattern = meta["glob"].format(folder=meta["folders"][r.T_nominal])
    return next(p for p in (cfg.DATA_RAW / meta["root"]).glob(pattern) if p.name == r.file)


def raw_frame(r):
    # Raw rows of one test after the row validity rules only (no warm-up, no resampling)
    meta = cfg.CELLS[r.cell]
    root = cfg.DATA_RAW / meta["root"]

    if r.cell == "TUBerlinM50LT":
        log, suffix = r.file.split("__")
        path = root / meta["folders"][r.T_nominal] / (log + ".txt")
        df = b4_pieces(str(path), r.unit)[suffix]
    else:
        if r.split == "trise":  # B6 warming-chamber tests live in their own folders
            patterns = [meta["trise_glob"].format(folder=f) for f in meta["trise_folders"][r.T_nominal]]
        else:
            patterns = [g.format(folder=meta["folders"][r.T_nominal], t=r.T_nominal)
                        for g in meta.get("globs", [meta.get("glob", "{folder}/*.csv")])]

        fname, _, piece = r.file.partition("__")  # B9: one test of a three-test log
        path = next(p for pat in patterns for p in root.glob(pat)
                    if p.name == r.file or (piece and p.stem == fname))
        if piece:
            df = dict(io.split_calce_dynamic(io.read_arbin(path)))[piece]
        else:
            df = io.read_any(path, meta["format"])

    keep = (clean.range_mask(df, meta.get("voltage_range"))
            & clean.interruption_mask(df)
            & clean.duplicate_time_mask(df))
    return resample.insert_step_starts(df[keep].reset_index(drop=True))


# Per-file measures for check 1

def warm_charge(d, r):
    # Charge [Ah] taken above 15 degC in a cold file: room-temperature charging left in
    # Only cells whose charges ran warm (B1, B3, B5)
    # B5 drive cycles left out: HWGRADE regenerates for minutes on a cell self-heated above 15 degC at 0 and 10 degC
    cold_warm_charged = r.T_nominal < 15 and r.cell in ("Samsung30T", "LGHG2", "Tesla2170")
    tesla_drive = r.cell == "Tesla2170" and r.type in TESLA_DRIVES
    if not cold_warm_charged or tesla_drive:
        return 0.0
    return float(np.clip(d.I.values, 0, None)[d["T"].values > 15].sum() / 3600)


def unsoaked_start(d, t_nom):
    # B5: 1 if the stretch of 2 min or more before the first load sits off the set point (unsoaked), else 0
    loaded = np.abs(d.I.values) > cfg.REST_CURRENT_A
    first = int(np.argmax(loaded)) if loaded.any() else len(d)
    pre = d.iloc[:first]
    return int(len(pre) >= cfg.WARMUP_MIN_SEGMENT_S and off_temperature(pre, t_nom))


def off_stretches(d, t_nom):
    # Number of long rest or charge stretches with rows far from the set point
    # Only B1 and B3: B4's chamber holds the set point (a warm stretch is self-heating),
    # B2's cool-down rests are handled separately, B5's logged set point replaces this check (see unsoaked_start)
    I = d.I.values
    kind = np.where(I < -cfg.REST_CURRENT_A, -1, np.where(I > cfg.REST_CURRENT_A, 1, 0))
    starts = np.r_[0, np.flatnonzero(np.diff(kind) != 0) + 1]
    bad = 0
    for a, b in zip(starts, np.r_[starts[1:], len(d)]):
        if kind[a] >= 0 and b - a >= cfg.WARMUP_MIN_SEGMENT_S:
            bad += int(off_temperature(d.iloc[a:b], t_nom))
    return bad


# Checks

def check_files(usable):
    # 1. Grid, ranges and missing values in every usable file
    bad_step = nan_rows = stretch_bad = 0
    soc_lo, soc_hi, warm_ah = 1.0, 0.0, 0.0

    for _, r in usable.iterrows():
        d = load(r)
        for _, g in d.groupby("segment"):
            bad_step += int((np.abs(np.diff(g.t.values) - cfg.RESAMPLE_S) > 1e-9).any())
        if r.cell != "LGM50":  # one B2 thermocouple is excluded on purpose
            nan_rows += int(d.loc[d.measured, ["V", "T", "soc"]].isna().any(axis=1).sum())
        soc_lo, soc_hi = min(soc_lo, float(d.soc.min())), max(soc_hi, float(d.soc.max()))
        warm_ah += warm_charge(d, r)
        if r.cell == "Tesla2170":
            stretch_bad += unsoaked_start(d, r.T_nominal)
        if r.cell in ("Samsung30T", "LGHG2"):
            stretch_bad += off_stretches(d, r.T_nominal)

    report("constant 1 s step inside segments", bad_step == 0, f"{bad_step} segments with an irregular step")
    report("missing V, T or SOC in measured rows", nan_rows == 0, f"{nan_rows}")
    report("SOC range", soc_lo >= cfg.SOC_RANGE[0] and soc_hi <= cfg.SOC_RANGE[1], f"{soc_lo:.3f} to {soc_hi:.3f}")
    report("warm charge in cold files", warm_ah < 0.5, f"{warm_ah:.2f} Ah")
    report("long rest/charge stretches off temperature", stretch_bad == 0, f"{stretch_bad}")
    report("no missing ambient temperature", not usable.T_amb.isna().any(), f"{int(usable.T_amb.isna().sum())} files")


def check_duplicates(usable):
    # 1b. No test stored twice among the usable files
    prints = {}
    for _, r in usable.iterrows():
        prints.setdefault(fingerprint(load(r)), []).append(r.file)
    dupes = [v for v in prints.values() if len(v) > 1]
    detail = f"{len(dupes)} groups" + (f": {dupes[:2]}" if dupes else "")
    report("no duplicate tests among usable files", not dupes, detail)


def check_rounded_grid(usable, cells):
    # 1c. Files put on the grid by rounding (B5) keep exactly one published sample per second
    # A share below 1 can only come from a second missing in the published data (filled by holding the current),
    # two samples in one second (the bias rounding removes) would show up in charge conservation instead
    rnd = [c for c in cells if cfg.CELLS[c].get("grid_align") == "round"]
    if not rnd:
        return
    frac = usable[usable.cell.isin(rnd)].measured_frac
    report("one sample per second on the rounded grid (B5)", bool((frac >= 0.999).all()),
           f"min measured share {frac.min():.4f}")


def check_charge_conservation(usable):
    # 2. Raw against processed charge, the 1 s averaging must not create or lose charge
    # Only tests with no rows removed by the set-point, warm-up or unknown-SOC rules
    setpoint_removed = usable["rows_removed_setpoint"].fillna(0) if "rows_removed_setpoint" in usable else 0
    sel = usable[(usable.cell != "LGM50")
                 & (usable.rows_removed_warmup == 0)
                 & (setpoint_removed == 0)
                 & (usable.rows_removed_soc_unknown.fillna(0) == 0)]

    for cell, g in sel.groupby("cell"):
        worst, n = 0.0, 0
        for _, r in g.iterrows():
            raw, d = raw_frame(r), load(r)
            dt = np.diff(raw.t.values)
            q_raw = float(np.sum(raw.I.values[:-1] * dt) / 3600)
            through = float(np.sum(np.abs(raw.I.values[:-1]) * dt) / 3600)  # total charge moved either way
            if through > 0.1:
                q_proc = d.I.sum() * cfg.RESAMPLE_S / 3600
                worst, n = max(worst, abs(q_raw - q_proc) / through), n + 1
        # B9 logs about once per second, so a current step shares its 1 s bin with the old value (about 0.3 %)
        limit = 0.005 if cell == "CALCE_A123" else 0.002
        report(f"charge conservation {cell} (worst of {n} files, share of throughput)", worst < limit,
               f"{100 * worst:.3f} %")


def check_b4_owner_soc(u):
    # 3a. B4 drive-cycle charge against the TU Berlin preprocessed files (owners' coulomb count)
    # Our Q_ref differs from theirs, so the charge removed is compared, which does not depend on the capacity
    root = cfg.DATA_RAW / "tuberlin_m50lt/Lithium-Ion Battery Drive Cycle Dataset/2_preprocessed"
    worst, n = 0.0, 0
    for _, r in u[~u.type.str.startswith("Charge")].iterrows():
        f = next(root.glob(f"JY_SOC_{r.T_nominal}deg/*_{r.type}_{r.T_nominal}deg.csv"))
        own = pd.read_csv(f)
        q_own = float(np.sum(own.Current.values[:-1] * np.diff(own.Time.values)) / 3600)
        q_ours = float(load(r).I.sum() * cfg.RESAMPLE_S / 3600)
        worst, n = max(worst, abs(q_ours - q_own) / abs(q_own)), n + 1
    report(f"B4 drive-cycle charge against the owners' preprocessed files (worst of {n})", worst < 0.01,
           f"{100 * worst:.2f} %")


def check_b5_owner_soc(u):
    # 3b. B5 drive-cycle SOC drop against the owners' SOC in the .mat files (their capacity interpolated
    # between stages), both drops should agree within the capacity difference (a few % of the drop)
    worst, n = 0.0, 0
    for _, r in u[u.type.isin(TESLA_DRIVES)].iterrows():
        meas = io.load_meas(tesla_raw_path(r))
        own = np.asarray(meas.SOC, dtype=float)
        t_own = np.asarray(meas.Time, dtype=float)
        t_own = t_own - t_own[0]
        d = load(r)

        # Compare at the last kept second: the set-point rule can cut the end of a cycle whose chamber was
        # already switched to 25 degC, so both drops cover the same stretch
        t_end = float(d.t.iloc[-1])
        drop_own = own[0] - float(np.interp(t_end, t_own, own))
        drop_ours = 1.0 - float(d.soc.iloc[-1])
        worst, n = max(worst, abs(drop_ours - drop_own) / drop_own), n + 1
    report(f"B5 drive-cycle SOC drop against the owners' SOC (worst of {n})", worst < 0.03, f"{100 * worst:.2f} %")


def check_b5_setpoint(u):
    # 3c. Every kept B5 row was logged while the chamber set point was within the band of its test temperature
    # (raw set point read again and matched to the processed time stamps)
    bad, n = 0, 0
    for _, r in u[u.setpoint_channel == "used"].iterrows():
        raw = io.read_tesla_mat(tesla_raw_path(r))
        d = load(r)
        tr, td = raw.t.values, d.t.values

        # Nearest raw sample to each processed second
        k = np.clip(np.searchsorted(tr, td), 1, len(raw) - 1)
        k = np.where(np.abs(tr[k - 1] - td) <= np.abs(tr[k] - td), k - 1, k)
        amb = raw.amb.values[k]
        off = np.flatnonzero(np.isfinite(amb) & (np.abs(amb - r.T_nominal) > cfg.WARMUP_BAND_K))

        # A 1 s bin at a cut may sit nearest to a dropped sample, it is fine if it holds a kept one
        for i in off:
            lo, hi = np.searchsorted(tr, td[i] - 0.5), np.searchsorted(tr, td[i] + 0.5)
            bad += int(not np.any(np.abs(raw.amb.values[lo:hi] - r.T_nominal) <= cfg.WARMUP_BAND_K))
        n += 1
    report(f"B5 kept rows logged at the test set point ({n} files)", bad == 0, f"{bad} rows off the set point")


# Running

def main(cells):
    m = pd.read_csv(PROC / "manifest.csv")
    m = m[m.cell.isin(cells)]
    usable = m[m.split != "excluded"]
    print(usable.groupby(["cell", "split"]).agg(files=("file", "size"), hours=("hours_kept", "sum")).round(1))

    check_files(usable)
    check_duplicates(usable)
    check_rounded_grid(usable, cells)
    check_charge_conservation(usable)

    # 3. SOC against the data owners' own coulomb count (independent code, different capacity choice)
    if "TUBerlinM50LT" in cells:
        check_b4_owner_soc(usable[usable.cell == "TUBerlinM50LT"])
    if "Tesla2170" in cells:
        tesla = usable[usable.cell == "Tesla2170"]
        check_b5_owner_soc(tesla)
        check_b5_setpoint(tesla)

    print(f"\n{sum(results)} of {len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:] or list(cfg.CELLS)) else 1)