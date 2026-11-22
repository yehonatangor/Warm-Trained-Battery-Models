"""
Open-circuit voltage curve U_oc(SOC) at 25 degC, one per cell type, shared by all its models
1. Keep 25 degC training rows with a small current (|I| <= OCV_MAX_C x nominal capacity)
2. Median voltage per 1 % SOC bin, separately for discharge/rest (I <= 0) and charge (I > 0)
3. Average the two branches where both exist (cancels most overpotential and hysteresis)
4. Fill empty bins, smooth (5-bin running median), force non-decreasing in SOC
Source rows: C/20 tests (B1, B3, B5), 0.1C discharges (B2), rests and drive pauses (B4, a pseudo-OCV)
Its error is a shared bias across all models of the cell (proposal, Section 3)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OCV_MAX_C = 0.11  # small-current limit as a fraction of the nominal capacity in A (C/20 and 0.1C pass)
BIN = 0.01  # SOC bin width
MIN_ROWS = 20  # a bin needs this many rows per branch


class OCVCurve:
    # OCV as a lookup table: SOC grid and voltage at each grid point

    def __init__(self, soc_grid, v_grid):
        self.soc = np.asarray(soc_grid, dtype=float)
        self.v = np.asarray(v_grid, dtype=float)

    def __call__(self, soc):
        # Voltage at the given SOC, flat beyond the ends of the measured range
        return np.interp(np.asarray(soc, dtype=float), self.soc, self.v)

    def save(self, path):
        Path(path).write_text(json.dumps({"soc": self.soc.tolist(), "v": self.v.tolist()}, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["soc"], d["v"])


def branch_medians(part, edges):
    # Median voltage in each SOC bin, NaN where a bin has fewer than MIN_ROWS rows
    idx = np.digitize(part.soc.values, edges) - 1
    med = np.full(len(edges) - 1, np.nan)
    for k in range(len(med)):
        v = part.V.values[idx == k]
        if len(v) >= MIN_ROWS:
            med[k] = np.median(v)
    return med


def fit_ocv(df, nominal_ah):
    # Fit the curve from a table with columns V, I, soc, measured (already only 25 degC training data)
    d = df[df.measured & df.soc.notna() & (df.I.abs() <= OCV_MAX_C * nominal_ah)]
    edges = np.arange(0.0, 1.0 + BIN, BIN)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Discharge or rest branch and charge branch
    both = np.vstack([branch_medians(d[d.I <= 0], edges), branch_medians(d[d.I > 0], edges)])

    # Mean of the branches present in each bin, NaN if neither
    n = np.isfinite(both).sum(axis=0)
    v = np.where(n > 0, np.nansum(both, axis=0) / np.maximum(n, 1), np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        raise ValueError("too few small-current rows to build an OCV curve")

    v = np.interp(centres, centres[ok], v[ok])  # fill empty bins inside, flat outside
    v = pd.Series(v).rolling(5, center=True, min_periods=1).median().values  # smooth
    v = np.maximum.accumulate(v)  # non-decreasing in SOC

    # Keep only the measured SOC range
    keep = (centres >= centres[ok].min()) & (centres <= centres[ok].max())
    return OCVCurve(centres[keep], v[keep])