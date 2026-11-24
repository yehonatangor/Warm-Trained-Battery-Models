"""
Clean every raw test: validity rules, late-logged step starts, set-point, soak and warm-up rules,
SOC, one 1 s grid and ambient temperature
Writes one Parquet file per test, the manifest (one row per test) and the reference capacities
"""

import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import clean, io, manifest, resample, soc, tuberlin

TRISE_TYPES = {"US06", "HWFET", "UDDS", "LA92", "NN", "Mixed cycle"}  # drive tests kept from the B6 Trise folders
AMBIENT_MEASURED = ("soaked_rest", "log_rest_end", "first_temperature")  # sources a filled ambient can borrow from


# Helpers

def unit_of(cell, f):
    # Physical cell a raw file belongs to
    if cell == "LGM50":
        return f.name[4:7]
    if cell in ("TUBerlinM50LT", "CALCE_A123"):
        return f.name[:6]  # e.g. NMC063, A1-007
    if cell == "Tesla2170":
        return f.parts[-4]
    return cell  # one cell per type


def test_type_of(cell, name):
    if cell == "LGM50":
        return "LG M50 rate test"
    if cell == "Tesla2170":
        return io.tesla_test_type(name)
    if cell == "Panasonic18650PF":
        return io.panasonic_test_type(name)
    if cell in ("A123_26650", "CALCE_A123"):
        return io.lfp_test_type(name)
    return io.test_type(name)


def content_hash(df):
    # Fingerprint of a raw test's V, I and T, catches the same test stored twice
    return hashlib.sha1(np.round(df[["V", "I", "T"]].to_numpy(dtype=float), 6).tobytes()).hexdigest()


def theta(T_c):
    # Dimensionless temperature input: T_ref * (1/T_ref - 1/T)
    return 1.0 - cfg.T_REF_K / (T_c + 273.15)


def write_parquet(out, path):
    # Write to a temporary file, then rename, so an interrupted run never leaves half a file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def exclude(rec, flag, end_soc=None):
    # Mark a test as excluded, nothing is written for it
    rec.update(flags=flag, split="excluded")
    return rec, None, end_soc


def fill_ambient(rec, out, log):
    # Ambient for a test with none of its own: median of the usable tests of the same cell type and set point
    # whose ambient was measured, the same physical cell first, fails loudly if there is none
    donors = [r for r in log
              if r["cell"] == rec["cell"] and r["T_nominal"] == rec["T_nominal"]
              and r.get("split") not in (None, "excluded")
              and r.get("T_amb_source") in AMBIENT_MEASURED]
    same_unit = [r["T_amb"] for r in donors if r.get("unit") == rec.get("unit")]
    values = same_unit or [r["T_amb"] for r in donors]
    if not values:
        raise RuntimeError(f"no measured ambient to fill {rec['file']}")

    rec["T_amb"] = float(np.median(values))
    out["T_amb"] = rec["T_amb"]
    out["theta_dimless"] = theta(rec["T_amb"])


# Cleaning steps

def early_exclusion(rec, meta, name, df, test, trise, seen):
    # Reason to skip a test before cleaning it, None if it goes on
    if name in meta.get("exclude_files", {}):
        return meta["exclude_files"][name]
    if test in meta.get("exclude_types", {}):
        return meta["exclude_types"][test]
    if trise and test not in TRISE_TYPES:
        return "trise_not_drive"  # charges and pauses between the Trise drives
    if len(df) < 2 or rec["hours_raw"] < 0.1:  # duration, not rows: a 60 s log of 25 min has only 27 rows
        return "too_short"
    if seen is not None:
        h = content_hash(df)
        if h in seen:  # same test stored twice under two names (B5 -20 degC C/20), keep the first
            return f"duplicate_of:{seen[h]}"
        seen[h] = name
    return None


def drop_invalid(df, rec, meta, temperature_excluded):
    # Validity rules: drop rows that cannot be real, log how many each rule removed
    checked = df.fillna({"T": 20.0}) if temperature_excluded else df  # missing temperature must not fail the range
    masks = {
        "range": clean.range_mask(checked, meta.get("voltage_range")),
        "interruption": clean.interruption_mask(df),
        "duplicate_time": clean.duplicate_time_mask(df),
    }
    keep = np.ones(len(df), bool)
    for key, m in masks.items():
        rec[f"rows_removed_{key}"] = int((keep & ~m).sum())  # only rows no earlier rule removed
        keep &= m
    return df[keep].reset_index(drop=True)


def temperature_keep(df, rec, t_nom, rule):
    # Rows kept by the set-point rule plus the soak or warm-up rule, and whether the set point channel is usable
    # Default: warm-up rule, B4: none (chamber held at the set point, warmer rows are self-heating)
    # B5 with a usable set point channel: soak rule, without one: warm-up rule
    channel_ok = clean.setpoint_channel_valid(df, t_nom)
    if "amb" in df:
        rec["setpoint_channel"] = "used" if channel_ok else "ignored"

    sp = clean.setpoint_mask(df, t_nom)
    rec["rows_removed_setpoint"] = int((~sp).sum())

    if rule == "soak" and channel_ok:
        wm = clean.soak_mask(df, t_nom)
    elif rule:
        wm = clean.warmup_mask(df, t_nom)
    else:
        wm = np.ones(len(df), bool)
    rec["rows_removed_warmup"] = int((sp & ~wm).sum())

    return wm & sp, channel_ok


def count_soc(df, meta, cell, t_nom, test, q_ref, split, start_soc, start_is_full, gap_rows):
    # SOC on the whole record, returns (soc, anchors, closure, start source)
    # B2 below 25 degC: chained from the end of the same cell's previous log, no anchors
    nominal = meta["nominal_ah"]
    if cell == "LGM50" and t_nom != 25:
        s, anchors, closure = soc.coulomb_count(df, q_ref, nominal, t_nom,
                                                start_soc=np.nan if start_soc is None else start_soc)
        return s, anchors, closure, "chained" if start_soc is not None else "missing"

    if start_is_full is None:
        start_is_full = cell != "LGM50"
    s, anchors, closure = soc.coulomb_count(
        df, q_ref, nominal, t_nom,
        start_is_full=start_is_full,
        spread_closure=split == "train",  # training files spread closure errors
        gap_rows=gap_rows,
        min_T_c=meta.get("anchor_min_T_c", soc.ANCHOR_MIN_T_C),
        start_under_load=meta.get("start_full_under_load", False) and test in TRISE_TYPES,
        volts=meta.get("anchor_volts"),
    )
    return s, anchors, closure, "anchor" if len(anchors) else "none"


def process_frame(cell, meta, t_nom, name, df, test, q_ref, unit, start_soc=None, start_is_full=None,
                  t_amb_measured=None, seen=None, trise=False):
    # Clean one test, returns (manifest record, 1 s table or None, SOC at the end of the raw log)
    # start_is_full: may row 0 be a full anchor (default: every cell but B2, B4: drive pieces only)
    # t_amb_measured: ambient measured outside this test (B4: end of the log's rests)
    # seen: content hashes so far, to catch duplicates
    # trise: B6 warming-chamber test, no warm-up rule, ambient = measured chamber temperature row by row,
    # its mixed cycles are kept since the reason for leaving them out only applies to training data
    t0 = time.time()
    nominal = meta["nominal_ah"]
    rec = dict(cell=cell, unit=unit, T_nominal=t_nom, file=name, type=test,
               rows_raw=len(df), hours_raw=float(df.t.iloc[-1] / 3600) if len(df) else 0.0)
    flags = []

    reason = early_exclusion(rec, meta, name, df, test, trise, seen)
    if reason:
        return exclude(rec, reason)

    nominal_split = "trise" if trise else manifest.assign_split(cell, t_nom, test, [])
    q_ref = soc.test_q_ref(q_ref, cell, unit, t_nom, name, nominal_split)
    if cell == "Tesla2170" and not np.isfinite(q_ref):
        return exclude(rec, "no_stage_capacity")
    rec["q_ref_Ah"] = q_ref

    # 1. Validity rules
    temperature_excluded = cell == "LGM50" and int(name[4:7]) in cfg.LGM50_EXCLUDE_TEMPERATURE_CELLS
    if temperature_excluded:
        df["T"] = np.nan  # offset thermocouple: voltage and current kept, temperature unusable
    df = drop_invalid(df, rec, meta, temperature_excluded)
    if len(df) < 2:
        return exclude(rec, "no_valid_rows")

    # 2. Rows at the true start of late-logged steps, so the current between rows is right
    df = resample.insert_step_starts(df)
    rec["rows_inserted"] = int((~df.measured).sum())

    # 3. Temperature rules, then unlogged gaps
    rule = False if trise else meta.get("warmup_rule", True)
    wm, channel_ok = temperature_keep(df, rec, t_nom, rule)
    gap_rows = resample.unlogged_gaps(df.t.values, df.V.values, df.measured.values)
    rec["unlogged_gaps"] = int(gap_rows.sum())

    # 4. SOC on the whole record, before the temperature rules drop rows
    s, anchors, closure, source = count_soc(df, meta, cell, t_nom, test, q_ref, nominal_split,
                                            start_soc, start_is_full, gap_rows)
    df["soc"] = s
    rec["soc_start_source"] = source
    rec["soc_anchors"] = len(anchors)
    rec["closure_max"] = closure
    if source in ("missing", "none"):
        flags.append("soc_unanchored")
    finite = s[np.isfinite(s)]
    end_soc = float(finite[-1]) if len(finite) else None

    # 5. Drop the temperature-rule rows, the next kept row starts a new segment
    start_trimmed = not wm[0]
    df["cut_before"] = np.r_[False, ~wm[:-1]] | gap_rows
    df = df[wm].reset_index(drop=True)

    unknown = df.soc.isna().values  # SOC unknown after an unlogged gap, these rows cannot be model inputs
    partly_unknown = unknown.any() and not unknown.all()
    rec["rows_removed_soc_unknown"] = int(unknown.sum()) if partly_unknown else 0
    if partly_unknown:
        df["cut_before"] |= np.r_[False, unknown[:-1]]
        df = df[~unknown].reset_index(drop=True)

    kept_s = float(df.t.iloc[-1] - df.t.iloc[0]) if len(df) > 1 else 0.0
    if kept_s < cfg.KEPT_MIN_S:
        flags.append("unsoaked_or_off_setpoint" if rule == "soak" and channel_ok else "too_short_after_warmup")
        rec["hours_kept"] = 0.0
        return exclude(rec, ";".join(flags), end_soc)

    # 6. Test flags and ambient temperature
    dt = np.diff(df.t.values, append=df.t.values[-1])
    if float(dt[np.abs(df.I.values) > cfg.REST_CURRENT_A].sum()) < cfg.MIN_LOAD_S:
        flags.append("no_load")  # rest only, or a few stray rows of current: nothing to learn
    if cell == "TUBerlinM50LT" and test == "Charge 1C" and df.V.iloc[0] >= cfg.TOP_UP_START_V:
        flags.append("top_up")  # first charge at 45 degC starts near full, not a full 1C charge
    T_p02 = float(np.nanpercentile(df["T"], 2)) if df["T"].notna().any() else np.nan

    if trise:
        t_amb, src = float(np.nanmedian(df["chamber"])), "chamber_measured"  # summary only, rows get the series
    elif t_amb_measured is not None:
        t_amb, src = t_amb_measured, "log_rest_end"
    else:
        t_amb, src = clean.ambient(df, nominal)

    if not trise and clean.label_mismatch(t_amb, t_nom):
        flags.append("label_mismatch")  # soaked starting temperature is off, self-heating does not count

    if cell == "Tesla2170" and (start_trimmed or src != "soaked_rest"):
        # Unsoaked start was trimmed, or the test starts under load: the first temperatures are not the ambient,
        # filled in clean_cell from usable soaked tests of the same cell at this stage
        t_amb, src = float("nan"), "filled"
    if not np.isfinite(t_amb):
        src = "filled"  # B2 cell 791 (no thermocouple) and the B5 cases above
    rec["T_amb_source"] = src

    soc_known = df.soc.values[np.isfinite(df.soc.values)]
    if len(soc_known) and (soc_known.min() < cfg.SOC_RANGE[0] or soc_known.max() > cfg.SOC_RANGE[1]):
        flags.append("soc_out_of_range")

    # 7. One 1 s grid, ambient temperature and the dimensionless temperature inputs
    out = resample.to_common_step(df, cfg.RESAMPLE_S, meta.get("grid_align", "floor"))
    out["T_amb"] = out["chamber"].values if trise else t_amb
    out = out.drop(columns=["chamber"], errors="ignore")
    out["theta_dimless"] = theta(out["T_amb"])
    out["theta_cell"] = theta(out["T"])  # same from the measured cell temperature

    # 8. Summary for the manifest
    soc_seen = out.soc.notna().any()
    rec["hours_kept"] = len(out) * cfg.RESAMPLE_S / 3600
    rec["measured_frac"] = float(out.measured.mean())
    rec["charge_Ah_kept"] = float(np.clip(out.I.values, 0, None).sum() * cfg.RESAMPLE_S / 3600)
    rec["T_cell_p02"] = T_p02
    rec["soc_min"] = float(np.nanmin(out.soc)) if soc_seen else np.nan
    rec["soc_max"] = float(np.nanmax(out.soc)) if soc_seen else np.nan
    rec["soc_nan_frac"] = float(out.soc.isna().mean())
    rec["T_amb"] = t_amb
    rec["segments"] = int(out.segment.nunique())
    rec["rows_processed"] = len(out)

    if trise:
        rec["split"] = "excluded" if flags else "trise"
    else:
        rec["split"] = manifest.assign_split(cell, t_nom, test, flags)

    if temperature_excluded:
        flags.append("temperature_excluded")  # informational only, voltage and current are still used
    rec["flags"] = ";".join(flags)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, out, end_soc


# Files to tests

def process_file(cell, meta, t_nom, f, q_ref, start_soc=None, seen=None, trise=False):
    # Read one raw file and clean it, files in an exclude_dirs folder are recorded as excluded without reading
    unit, test = unit_of(cell, f), test_type_of(cell, f.name)
    skip = meta.get("exclude_dirs", {}).get(f.parent.name)
    if skip:
        return exclude(dict(cell=cell, unit=unit, T_nominal=t_nom, file=f.name, type=test), skip)

    df = io.read_any(f, meta["format"])
    return process_frame(cell, meta, t_nom, f.name, df, test, q_ref, unit, start_soc, seen=seen, trise=trise)


def tests_of(cell, meta, t_nom, f, q_ref, start_soc, seen, trise=False):
    # Yields (record, output, end SOC) for every test in raw file f
    # One test per file, except B9 dynamic logs (three) and B4 logs (eight)
    unit = unit_of(cell, f)

    if meta["format"] == "calce_dynamic" and "OCV" not in f.name:
        for kind, piece in io.split_calce_dynamic(io.read_arbin(f)):
            name = f"{f.stem}__{kind}"
            yield process_frame(cell, meta, t_nom, name, piece, test_type_of(cell, name), q_ref, unit, seen=seen)
        return

    if meta["format"] != "basytec_txt":
        yield process_file(cell, meta, t_nom, f, q_ref, start_soc, seen, trise)
        return

    log = io.read_basytec_txt(f)
    t_rest = clean.log_rest_temperature(log)
    for suffix, kind, piece in tuberlin.split_log(log, unit):
        # A drive piece starts at the end of the rest after a CV hold (full)
        # A charge piece starts where the previous drive ended, so its only anchor is its own CV end
        yield process_frame(cell, meta, t_nom, f"{f.stem}__{suffix}", piece, kind, q_ref, unit,
                            start_is_full=kind != "Charge 1C", t_amb_measured=t_rest, seen=seen)


# Running

def clean_cell(cell, temps, log):
    # Clean every test of one cell, append the records to log, return its Q_ref
    meta = cfg.CELLS[cell]
    if cell == "LGM50" and temps is not None:
        raise ValueError("LG M50 SOC is chained from 25 to 10 to 0 degC: run all its temperatures together")

    folders = {t: fo for t, fo in meta["folders"].items() if temps is None or t in temps}
    for t_nom in folders:  # no stale files from an earlier run
        shutil.rmtree(cfg.DATA_PROCESSED / cell / str(t_nom), ignore_errors=True)

    q_ref = soc.reference_capacity(cell, meta)
    print(f"{cell}: Q_ref = {q_ref}", flush=True)

    end_soc = {}  # B2: SOC at the end of each cell's previous log (tested 25, then 10, then 0 degC)
    pending = []  # outputs waiting for an ambient temperature
    written, seen = set(), {}

    root = cfg.DATA_RAW / meta["root"]
    jobs = [(t, g.format(folder=fo, t=t), False)
            for t, fo in folders.items()
            for g in meta.get("globs", [meta.get("glob", "{folder}/*.csv")])]
    jobs += [(t, meta["trise_glob"].format(folder=fo), True)  # B6 warming-chamber tests
             for t, fos in meta.get("trise_folders", {}).items() if t in folders
             for fo in fos]

    for t_nom, pattern, trise in jobs:
        files = sorted(p for p in root.glob(pattern) if not p.name.startswith("~$"))  # skip Excel lock files
        for f in files:
            cell_id = f.name[:7]  # B2 cell id, links its logs across temperatures
            try:
                results = list(tests_of(cell, meta, t_nom, f, q_ref, end_soc.get(cell_id), seen, trise))
            except Exception:  # recorded with its traceback, never dropped silently
                print(traceback.format_exc(), flush=True)
                rec = dict(cell=cell, unit=unit_of(cell, f), T_nominal=t_nom, file=f.name)
                results = [exclude(rec, "error")]

            for rec, out, last in results:
                if cell == "LGM50":
                    end_soc[cell_id] = last

                path = cfg.DATA_PROCESSED / cell / str(t_nom) / (Path(rec["file"]).stem + ".parquet")
                if out is not None:
                    # An excluded copy may share a name with the kept one (B6 dis5_10p files), a kept one may not
                    if path in written:
                        raise RuntimeError(f"two tests map to {path}")
                    written.add(path)
                    if np.isnan(rec["T_amb"]):
                        pending.append((rec, out, path))
                    else:
                        write_parquet(out, path)

                log.append(rec)
                print(cell, t_nom, rec["file"], rec.get("split"), rec.get("flags", ""), flush=True)

    for rec, out, path in pending:  # donors are measured tests only, so the filling order does not matter
        fill_ambient(rec, out, log)
        write_parquet(out, path)

    return q_ref


def save_capacities(caps):
    # A run on some cells keeps the other cells' capacities
    path = cfg.DATA_PROCESSED / "reference_capacity.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(caps)
    path.write_text(json.dumps(old, indent=2))


def save_manifest(log, cells, temps):
    # Write manifest.csv, keeping the rows of cells and temperatures not rerun
    m = pd.DataFrame(log)
    path = cfg.DATA_PROCESSED / "manifest.csv"
    if path.exists():
        prev = pd.read_csv(path)
        rerun = prev.cell.isin(cells) & (prev.T_nominal.isin(temps) if temps is not None else True)
        m = pd.concat([prev[~rerun], m], ignore_index=True)

    m = m.sort_values(["cell", "T_nominal", "file"], ascending=[True, False, True]).reset_index(drop=True)
    m.to_csv(path, index=False)
    return m


def main(cells, temps=None):
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    log = []
    caps = {cell: clean_cell(cell, temps, log) for cell in cells}
    save_capacities(caps)

    m = save_manifest(log, cells, temps)
    summary = m[m.cell.isin(cells)].groupby(["cell", "split"]).agg(files=("file", "size"),
                                                                     hours=("hours_kept", "sum"))
    print(summary.round(1))


if __name__ == "__main__":
    args = sys.argv[1:]
    temps = None
    if "--temps" in args:
        k = args.index("--temps")
        temps = [int(x) for x in args[k + 1:]]
        args = args[:k]
    main(args or list(cfg.CELLS), temps)