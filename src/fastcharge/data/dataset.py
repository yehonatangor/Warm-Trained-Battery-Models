"""
Model-ready data: one table per request, with built-in data rules
Rules:
1. Never return excluded files
2. Test splits (test, fresh, trise) need allow_test=True, meant for the single final test run
3. used_info (B4 1C charges at 35 and 5 degC) needs allow_used_info=True and is never scored,
   it holds information at test temperatures, so no model may train or validate on it
4. Every row keeps measured (score on measured rows only) and piece (one continuous stretch of one file,
   a model never steps from one piece into the next)
"""

from pathlib import Path

import pandas as pd

from fastcharge import config as cfg

TEST_SPLITS = {"test", "fresh", "trise"}


def manifest() -> pd.DataFrame:
    # Measured files (manifest.csv, from the cleaning) plus simulated ones (manifest_sim.csv, if present)
    parts = [pd.read_csv(cfg.DATA_PROCESSED / "manifest.csv")]
    sim = cfg.DATA_PROCESSED / "manifest_sim.csv"
    if sim.exists():
        parts.append(pd.read_csv(sim))
    return pd.concat(parts, ignore_index=True)


def check_access(splits, allow_test, allow_used_info):
    # Guard against loading data a model must not see by accident
    if "excluded" in splits:
        raise ValueError("excluded files are never used")
    if splits & TEST_SPLITS and not allow_test:
        raise PermissionError("test data are for the single final test run: pass allow_test=True on purpose")
    if "used_info" in splits and not allow_used_info:
        raise PermissionError("used_info is never scored: pass allow_used_info=True on purpose")


def load(cell: str, splits=("train",), allow_test: bool = False, types=None,
         allow_used_info: bool = False, temps=None) -> pd.DataFrame:
    # Stack every usable file of the cell whose split is in splits into one table
    # types / temps: optional filters on test type and chamber temperature (temps saves memory for B5)
    # q_ref_Ah is the capacity SOC was counted with (changes between B5 stages, models in Ah must use it)
    splits = set(splits)
    check_access(splits, allow_test, allow_used_info)

    m = manifest()
    rows = m[(m.cell == cell) & m.split.isin(splits)]
    if types is not None:
        rows = rows[rows.type.isin(types)]
    if temps is not None:
        rows = rows[rows.T_nominal.isin(temps)]

    parts = []
    for _, r in rows.iterrows():
        stem = Path(r.file).stem  # file name without extension (B4 pieces have none)
        d = pd.read_parquet(cfg.DATA_PROCESSED / cell / str(r.T_nominal) / (stem + ".parquet"))

        # File info for every row
        d["piece"] = stem + "#" + d.segment.astype(str)
        d = d.assign(
            file=r.file,
            unit=r.get("unit", ""),  # physical cell (several per type for B2, B4, B5)
            T_nominal=r.T_nominal,
            type=r.type,
            split=r.split,
            q_ref_Ah=r.get("q_ref_Ah", float("nan")),
            T_amb_source=r.get("T_amb_source", ""),
        )
        parts.append(d)

    if not parts:
        raise ValueError(f"no usable files for {cell} in {sorted(splits)}")
    return pd.concat(parts, ignore_index=True)


def pieces(df: pd.DataFrame):
    # Continuous stretches one at a time: (piece id, rows in time order)
    return df.groupby("piece", sort=False)