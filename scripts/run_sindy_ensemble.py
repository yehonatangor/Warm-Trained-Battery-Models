"""
Ensemble SINDy: refit SINDy (selection included) on bootstrap resamples of the training pieces, then report each
library term's inclusion probability and a 5 to 95 % prediction band

Resampling: pieces drawn with replacement, each draw weighted as its own file
100 full-size resamples if the single full fit took under 10 min, else 30 resamples of half the pieces
Each refit starts from the full fit's coefficients with the whole library on (dropped terms at zero)
Resumable: refit b saved as fits/fit_###.json and skipped if present, seed 1000 + b

Aggregation (--aggregate) on the selection split (LGM50: its 10 degC training data), test data never loaded:
    inclusion.csv      share of refits keeping each term, plus the Butler-Volmer group as one
    band.csv           per piece: share of measured voltage inside the band and mean band width, free run and windows
    band_summary.csv   the same averaged per temperature
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package

import numpy as np
import pandas as pd

from warmtrained import config as cfg
from warmtrained.data import dataset
from warmtrained.evaluation import harness
from warmtrained.models.ocv import OCVCurve
from warmtrained.models.sindy import BV_IDX, DYN_NAMES, OUT_NAMES, TH_NAMES, SINDyModel

OUT = cfg.RESULTS / "sindy_ensemble"
LIMIT_S = 600.0  # single-fit time below which the full ensemble is run
BLOCKS = (("out", "output", OUT_NAMES), ("d1", "branch1", DYN_NAMES), ("d2", "branch2", DYN_NAMES),
          ("th", "thermal", TH_NAMES))


def plan(cell):
    # (number of resamples, share of pieces per resample)
    full = json.loads((cfg.RESULTS / "sindy" / cell / "full" / "sindy_params.json").read_text())
    return (100, 1.0) if full["fit_seconds"] < LIMIT_S else (30, 0.5)


def warm_model(cell, ocv):
    # Whole library active, coefficients from the full fit
    ecm = json.loads((cfg.RESULTS / "ecm" / cell / "single_2T" / "ecm_params.json").read_text())
    full = SINDyModel.load(cfg.RESULTS / "sindy" / cell / "full" / "sindy_params.json", ocv)
    m = SINDyModel(ocv, ecm, cfg.MODEL_CELLS[cell]["nominal_ah"], "full")
    m.co, m.c1, m.c2, m.ct = full.co.copy(), full.c1.copy(), full.c2.copy(), full.ct.copy()
    return m


def resample(train, b, frac):
    # Bootstrap draw of pieces, each draw renamed so it counts as its own piece and file
    rng = np.random.default_rng(1000 + b)
    pieces = train.piece.unique()
    draw = rng.choice(pieces, size=max(1, int(round(frac * len(pieces)))), replace=True)
    groups = dict(tuple(train.groupby("piece", sort=False)))
    parts = []
    for k, p in enumerate(draw):
        g = groups[p].copy()
        g["piece"] = f"{p}#b{k}"
        g["file"] = f"{g.file.iat[0]}#b{k}"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def fit(cell, max_fits, reverse=False):
    # Fit the missing refits, at most max_fits new ones
    n, frac = plan(cell)
    out = OUT / cell / "fits"
    out.mkdir(parents=True, exist_ok=True)
    ocv = OCVCurve.load(cfg.RESULTS / "arx" / cell / "ocv.json")
    train = dataset.load(cell, ["train"])
    done = 0
    print(f"{cell}: {n} resamples of {frac:.0%} of the training pieces", flush=True)
    for b in (range(n - 1, -1, -1) if reverse else range(n)):
        f = out / f"fit_{b:03d}.json"
        if f.exists():
            continue
        if max_fits is not None and done >= max_fits:
            break
        t0 = time.time()
        m = warm_model(cell, ocv).fit(resample(train, b, frac))
        m.save(f, {"resample": b, "fraction": frac})
        done += 1
        print(f"{cell} fit {b:03d}: {time.time() - t0:.0f} s, terms {sum(a.sum() for a in m.active.values())}",
              flush=True)
    print(f"{cell}: {len(glob.glob(str(out / 'fit_*.json')))} of {n} fits present", flush=True)


def selection_data(cell):
    if cell == "LGM50":
        return dataset.load(cell, ["train"], temps=[10])
    return dataset.load(cell, ["select"])


def ensemble_models(cell):
    # Saved refits of one cell as models, and their parameter records
    ocv = OCVCurve.load(cfg.RESULTS / "arx" / cell / "ocv.json")
    files = sorted(glob.glob(str(OUT / cell / "fits" / "fit_*.json")))
    return [SINDyModel.load(Path(f), ocv) for f in files], [json.loads(Path(f).read_text()) for f in files]


def band_table(models, data):
    # Per piece: share of measured voltage inside the 5 to 95 % band and mean band width, free run and windows
    band = []
    for piece, g in data.groupby("piece", sort=False):
        g = g.reset_index(drop=True)
        V, meas = g.V.values, g.measured.values & np.isfinite(g.V.values)
        if meas.sum() < harness.MIN_MEASURED:
            continue
        lo, hi = np.percentile([m.free_run(g)[0] for m in models], [5, 95], axis=0)
        r = dict(piece=piece, file=g.file.iat[0], T_nominal=g.T_nominal.iat[0], type=g.type.iat[0],
                 free_coverage=float(np.mean((V[meas] >= lo[meas]) & (V[meas] <= hi[meas]))),
                 free_width_mV=float(np.mean(hi[meas] - lo[meas]) * 1e3))
        starts = harness.window_starts(len(g))
        if len(starts):
            hor = int(harness.HORIZON_S)
            W = np.array([m.predict_windows(g, starts, hor)[0] for m in models])  # (fits, windows, horizon)
            wl, wh = np.percentile(W, [5, 95], axis=0)
            idx = starts[:, None] + 1 + np.arange(hor)[None, :]
            Vw, mw = V[idx], meas[idx]
            r["win_coverage"] = float(((Vw >= wl) & (Vw <= wh))[mw].mean())
            r["win_width_mV"] = float((wh - wl)[mw].mean() * 1e3)
        band.append(r)
    return pd.DataFrame(band)


def aggregate(cell):
    models, params = ensemble_models(cell)
    if not models:
        raise SystemExit(f"{cell}: no fits yet")

    rows = [dict(equation=name, term=t, inclusion=np.mean([t in p["selected"][key] for p in params]))
            for key, name, names in BLOCKS for t in names]
    rows.append(dict(equation="output", term="BV group",
                     inclusion=np.mean([any(OUT_NAMES[j] in p["selected"]["out"] for j in BV_IDX) for p in params])))
    inc = pd.DataFrame(rows)
    inc.insert(0, "n_fits", len(models))
    inc.to_csv(OUT / cell / "inclusion.csv", index=False)

    band = band_table(models, selection_data(cell))
    band.to_csv(OUT / cell / "band.csv", index=False)
    summ = band.groupby("T_nominal")[["free_coverage", "free_width_mV", "win_coverage", "win_width_mV"]].mean()
    summ.insert(0, "n_fits", len(models))
    summ.to_csv(OUT / cell / "band_summary.csv")
    print(inc[inc.inclusion > 0].to_string(index=False))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--max-fits", type=int, default=None)
    ap.add_argument("--reverse", action="store_true", help="fit from the last resample down")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    for c in a.cells:
        aggregate(c) if a.aggregate else fit(c, a.max_fits, a.reverse)