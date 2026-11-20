"""
Readers for the raw battery logs
Every reader returns a DataFrame with t [s], V [V], I [A, charge positive], T [degC], status and step
"""

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CELLS, DATA_RAW

logger = logging.getLogger(__name__)


# Helpers

def finish(out, required=("t", "V", "I", "T")):
    # Drop rows with missing values and start the time at zero
    out = out.dropna(subset=list(required)).reset_index(drop=True)
    if len(out) > 0:
        out["t"] -= out["t"].iloc[0]
    return out


def to_seconds(series):
    # "hh:mm:ss" text to seconds
    return pd.to_timedelta(series.astype(str).str.strip(), errors="coerce").dt.total_seconds().values


def to_num(series):
    return pd.to_numeric(series, errors="coerce").values


def find_col(columns, prefix):
    # First column whose name starts with the prefix, None if there is none
    return next((c for c in columns if c.startswith(prefix)), None)


def load_meas(path):
    # Only the .mat readers need scipy
    from scipy.io import loadmat
    return loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]


def short_path(path, cell):
    # Path relative to the cell root, for log messages
    try:
        return path.relative_to(DATA_RAW / CELLS[cell]["root"])
    except ValueError:
        return path


# Readers

def read_digatron(path):
    # McMaster Digatron CSV (B1, B3, B7)
    # Header block, then a "Time Stamp" line, a units line and the data rows
    with open(path, errors="replace") as f:
        lines = f.readlines()

    h = next(i for i, line in enumerate(lines) if line.startswith("Time Stamp"))
    rows = [line for line in lines[h + 2:] if line.strip() and line[0].isdigit()]

    d = pd.read_csv(io.StringIO(lines[h] + "".join(rows)), low_memory=False, index_col=False)
    d.columns = d.columns.str.strip()

    out = pd.DataFrame({
        "t": to_seconds(d["Prog Time"]),
        "V": to_num(d["Voltage"]),
        "I": to_num(d["Current"]),
        "T": to_num(d["Temperature"]),
        "status": d["Status"].astype(str).str.strip().values,
        "step": to_num(d["Step"]),
        "step_time": to_seconds(d["Step Time"]),  # time since the step began, finds late step changes
    })
    return finish(out)


def read_basytec(path):
    # Brosa Planella LG M50 log (B2), 14 header lines
    # Temperature from LogTempMid if it exists, else LogTemp001
    d = pd.read_csv(path, skiprows=14, encoding="latin-1", low_memory=False)
    d.columns = d.columns.str.strip().str.lstrip("~")
    cols = d.columns

    time_col = find_col(cols, "Prog Time")
    temp_col = find_col(cols, "LogTempMid") or find_col(cols, "LogTemp001")

    t = to_seconds(d[time_col])
    if np.isnan(t).mean() > 0.5:
        # Some files store the time as plain seconds
        t = to_num(d[time_col])

    out = pd.DataFrame({
        "t": t,
        "V": to_num(d[find_col(cols, "Voltage")]),
        "I": to_num(d[find_col(cols, "Current")]),
        "T": to_num(d[temp_col]),
        "Ah": to_num(d[find_col(cols, "AhAccu")]),
        "status": d["Status"].astype(str).str.strip().values,
        "step": to_num(d["Step"]),
    })
    return finish(out, required=("t", "V", "I"))


def read_basytec_txt(path):
    # TU Berlin LG M50LT log (B4), BaSyTec text export
    # Column names come from the "~Time[h]" line
    # Capacity files have a DateTime column written as two tokens
    # Extra columns: cmd (Pause, Charge, Discharge, Table = drive cycle), step = test plan line
    with open(path, encoding="latin-1") as f:
        header = next(line for line in f if line.startswith("~Time"))

    names = header.lstrip("~").split()
    if "DateTime" in names:
        k = names.index("DateTime")
        names[k:k + 1] = ["Date", "Clock"]

    d = pd.read_csv(path, sep=r"\s+", comment="~", header=None, names=names,
                    encoding="latin-1", low_memory=False)

    out = pd.DataFrame({
        "t": to_num(d[find_col(names, "Time[h]")]) * 3600.0,
        "V": to_num(d[find_col(names, "U[V]")]),
        "I": to_num(d[find_col(names, "I[A]")]),
        "T": to_num(d[find_col(names, "T1")]),
        "status": "",
        "step": to_num(d[find_col(names, "Line")]),
        "step_time": to_num(d[find_col(names, "t-Step")]) * 3600.0,
        "cmd": d[find_col(names, "Command")].astype(str).values,
    })
    return finish(out, required=("t", "V", "I"))


def read_tesla_mat(path):
    # McMaster Tesla 2170 .mat file (B5), 1 Hz, field name case changes between files
    # Extra column: amb, the chamber set point (NaN if not logged)
    meas = load_meas(path)
    fields = {k.lower(): k for k in meas._fieldnames}

    def get(name):
        key = fields.get(name.lower())
        return None if key is None else np.asarray(getattr(meas, key), dtype=float)

    t = get("Time")
    amb = get("Ambient_Temp_degC")

    out = pd.DataFrame({
        "t": t,
        "V": get("Voltage"),
        "I": get("Current"),
        "T": get("Battery_Temp_degC"),
        "status": "",
        "step": np.nan,
        "amb": amb if amb is not None else np.full(len(t), np.nan),
    })
    return finish(out)


def read_panasonic_mat(path):
    # UW-Madison Panasonic 18650PF .mat file (B6)
    # Drive cycles logged every 0.1 s, charges and pauses every 10 to 60 s
    # Extra column: chamber, measured chamber temperature (ambient for the Trise tests)
    meas = load_meas(path)

    def get(name):
        # atleast_1d because 1-row files load as a single number
        return np.atleast_1d(np.asarray(getattr(meas, name), dtype=float))

    out = pd.DataFrame({
        "t": get("Time"),
        "V": get("Voltage"),
        "I": get("Current"),
        "T": get("Battery_Temp_degC"),
        "status": "",
        "step": np.nan,
        "chamber": get("Chamber_Temp_degC"),
    })
    return finish(out)


def read_arbin(path):
    # Arbin export, .xls or .xlsx (B8, B9)
    # Data is on the "Channel_..." sheets, long tests are split over several
    # Temperature from the first "Temperature" column, NaN if there is none
    x = pd.ExcelFile(path)
    sheets = [s for s in x.sheet_names if s.startswith("Channel")]
    d = pd.concat([x.parse(s) for s in sheets], ignore_index=True)
    d = d.sort_values("Test_Time(s)", kind="stable")

    temp_col = next((c for c in d.columns if "Temperature" in c), None)

    out = pd.DataFrame({
        "t": to_num(d["Test_Time(s)"]),
        "V": to_num(d["Voltage(V)"]),
        "I": to_num(d["Current(A)"]),
        "T": to_num(d[temp_col]) if temp_col else np.nan,
        "status": "",
        "step": to_num(d["Step_Index"]),
    })
    return finish(out, required=("t", "V", "I"))


def split_calce_dynamic(log, min_charge_s=600.0):
    # Cut a CALCE "DST-US06-FUDS" log into its three tests
    # Each test is a 1C CC-CV charge, a 2 min rest and one drive profile down to 2.0 V
    # A piece runs from its charge to the next charge so its own CV end anchors the SOC
    charging = (log.I > 0.5).values
    t = log.t.values

    # First and last row of every charging stretch
    starts = np.flatnonzero(charging & ~np.r_[False, charging[:-1]])
    ends = np.flatnonzero(charging & ~np.r_[charging[1:], False])

    # Profiles have short charge pulses, so a real charge needs at least 10 min of CC
    keep = [s for s, e in zip(starts, ends) if t[e] - t[s] >= min_charge_s]

    bounds = keep + [len(log)]
    pieces = []
    for k, name in enumerate(["DST", "US06", "FUDS"]):
        if k + 1 < len(bounds):
            p = log.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
            p["t"] -= p["t"].iloc[0]
            pieces.append((name, p))
    return pieces


# Test type labels

TEST_TYPES = [
    ("HPPC", "HPPC"), ("Cap_1C", "Capacity 1C"), ("C20", "C/20"), ("Dis_", "CC discharge"),
    ("UDDS", "UDDS"), ("HWFET", "HWFET"), ("LA92", "LA92"), ("US06", "US06"),
    ("Mixed", "Mixed cycle"), ("Charge", "Charge"), ("PausCycl", "Pause cycle"),
]

TESLA_TYPES = [
    ("REORDERED", "REORDERED"), ("HWCUST", "HWCUST"), ("HWGRADE", "HWGRADE"), ("HPPC", "HPPC"),
    ("40C_Discharge", "C/20 at 40 degC"), ("C_20_Discharge", "C/20"), ("CC_CV_charge", "Charge"),
    ("_Discharge", "CC discharge"), ("Other", "Other"),
]

# The 25 degC highway cycle was split in two (HWFTa and HWFTb)
PANASONIC_CYCLES = {
    "US06": "US06", "HWFET": "HWFET", "HWFTa": "HWFET", "HWFTb": "HWFET",
    "UDDS": "UDDS", "LA92": "LA92", "NN": "NN",
}

PANASONIC_OTHER = [
    ("HPPC", "HPPC"), ("C20", "C/20"), ("Dis1C", "CC discharge"), ("dis5_10p", "CC discharge"),
    ("DisPulse", "CC discharge"), ("PreChg", "Charge"), ("Charge", "Charge"), ("Pause", "Rest"),
]


def first_match(text, table, default="Other"):
    # Label of the first key found in the text
    return next((label for key, label in table if key in text), default)


def test_type(filename):
    # McMaster, Brosa Planella and TU Berlin file names
    return first_match(filename, TEST_TYPES)


def tesla_test_type(filename):
    # Only look between the step number and the date
    # The schedule name later in the file name is not the test type
    head = re.sub(r"^\d+_", "", filename)
    head = re.split(r"_\d\d-\d\d-\d\d_", head)[0]
    return first_match("_" + head, TESLA_TYPES)


def panasonic_test_type(filename):
    # Two or more cycle names = a combined file, the same cycles are also stored split
    # Cycle_1 to Cycle_4 are random mixes of the five cycles
    tokens = re.split(r"[_ .]", filename)
    cycles = {PANASONIC_CYCLES[t] for t in tokens if t in PANASONIC_CYCLES}

    if len(cycles) > 1:
        return "Combined drive cycles"
    if len(cycles) == 1:
        return cycles.pop()
    if "Cycle" in tokens:
        return "Mixed cycle"

    lower_table = [(key.lower(), label) for key, label in PANASONIC_OTHER]
    return first_match(filename.lower(), lower_table)


def lfp_test_type(filename):
    # B8 and B9: the slow OCV test takes the place of the C/20 test
    # Plett "DYN" tests are scaled UDDS profiles
    # CALCE pieces get their names from split_calce_dynamic
    if "OCV" in filename:
        return "C/20"
    if "_DYN_" in filename:
        return "UDDS"
    return next((k for k in ["DST", "US06", "FUDS"] if filename.endswith("__" + k)), "Other")


# Format table, keyed by CELLS[cell]["format"]
# glob is only the default for cells with no glob in the config

FORMATS = {
    "digatron":      {"reader": read_digatron,      "label": test_type,           "glob": "{folder}/**/*.csv"},
    "basytec":       {"reader": read_basytec,       "label": test_type,           "glob": "{folder}/**/*.csv"},
    "basytec_txt":   {"reader": read_basytec_txt,   "label": test_type,           "glob": "{folder}/*.txt"},
    "tesla_mat":     {"reader": read_tesla_mat,     "label": tesla_test_type,     "glob": "{folder}/**/*.mat"},
    "panasonic_mat": {"reader": read_panasonic_mat, "label": panasonic_test_type, "glob": "{folder}/**/*.mat"},
    "arbin":         {"reader": read_arbin,         "label": lfp_test_type,       "glob": "{folder}/*.xlsx"},
    "calce_dynamic": {"reader": read_arbin,         "label": lfp_test_type,       "glob": "{folder}/*.xlsx"},
}


def read_any(path, fmt):
    return FORMATS[fmt]["reader"](Path(path))


def read_cell_file(path, cell):
    return read_any(path, CELLS[cell]["format"])


def label_file(filename, cell):
    return FORMATS[CELLS[cell]["format"]]["label"](filename)


# Finding the files of a cell

def get_patterns(cfg):
    if "globs" in cfg:
        return cfg["globs"]
    return [cfg.get("glob", FORMATS[cfg["format"]]["glob"])]


def exclude_reason(path, cfg):
    # Reason the config gives for skipping this file, None if it is kept
    if path.name in cfg.get("exclude_files", {}):
        return cfg["exclude_files"][path.name]

    exclude_dirs = cfg.get("exclude_dirs", {})
    bad_dir = next((part for part in path.parts if part in exclude_dirs), None)
    if bad_dir:
        return exclude_dirs[bad_dir]

    label = FORMATS[cfg["format"]]["label"](path.name)
    return cfg.get("exclude_types", {}).get(label)


def cell_files(cell, include_trise=True):
    # Returns (temperature, path, kind) for every usable file, kind is "main" or "trise"
    # Skipped files are logged with the reason from the config
    cfg = CELLS[cell]
    root = DATA_RAW / cfg["root"]

    jobs = [(temp, folder, "main", pattern)
            for temp, folder in cfg["folders"].items()
            for pattern in get_patterns(cfg)]

    if include_trise:
        trise_glob = cfg.get("trise_glob", "{folder}/*.mat")
        jobs += [(temp, folder, "trise", trise_glob)
                 for temp, folders in cfg.get("trise_folders", {}).items()
                 for folder in folders]

    files = []
    seen = set()
    for temp, folder, kind, pattern in jobs:
        matches = sorted(root.glob(pattern.format(folder=folder, t=temp)))
        if not matches:
            logger.warning("%s %+d degC: no files for %s", cfg["label"], temp, pattern.format(folder=folder, t=temp))

        for path in matches:
            if path in seen or not path.is_file():
                continue
            seen.add(path)

            reason = exclude_reason(path, cfg)
            if reason:
                logger.info("%s skip %s (%s)", cfg["label"], short_path(path, cell), reason)
                continue

            files.append((temp, path, kind))

    logger.info("%s (%s): %d files found", cfg["label"], cell, len(files))
    return files


def iter_tests(cell, include_trise=True):
    # Yields (temperature, test type, name, log, kind) for each test
    # CALCE dynamic logs are cut into their three tests first
    cfg = CELLS[cell]
    for temp, path, kind in cell_files(cell, include_trise):
        logger.debug("%s read %s", cfg["label"], short_path(path, cell))
        log = read_any(path, cfg["format"])

        if cfg["format"] == "calce_dynamic" and "dynamic" in path.parts:
            for piece, p in split_calce_dynamic(log):
                name = path.stem + "__" + piece
                yield temp, label_file(name, cell), name, p, kind
        else:
            yield temp, label_file(path.name, cell), path.stem, log, kind