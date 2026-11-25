"""
Run the whole pipeline in order and time every step (per cell for model scripts)
Each step runs as its own process with this Python (uv run python scripts/run_all.py uses the project's .venv)
Timings are appended to results/timings.csv as each step finishes, so a crash loses nothing already timed
Model scripts run once per cell, so their times are fit plus scoring per model per cell

    uv run python scripts/run_all.py                 core pipeline
    uv run python scripts/run_all.py --exploratory   core, then exploratory analyses
    uv run python scripts/run_all.py --from 7        start at step 7 (numbers match the run guide in reproduce.ipynb)
    uv run python scripts/run_all.py --list          print the steps and exit
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from warmtrained import config as cfg  # noqa: E402

OUT = cfg.RESULTS / "timings.csv"

MODEL_CELLS = list(cfg.MODEL_CELLS)
LSTM_CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50", "Panasonic18650PF"]
ENS_CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50"]

# (step, script, args, cells to run one at a time or None)
CORE = [
    ("1", "clean_dataset.py", [], list(cfg.CELLS)),
    ("2a", "verify_cleaning.py", [], None),
    ("2b", "simulate_known_truth.py", [], None),
    ("3", "run_arx.py", [], MODEL_CELLS),
    ("4", "run_ecm.py", [], MODEL_CELLS),
    ("5a", "run_sindy.py", [], MODEL_CELLS),
    ("5b", "run_lstm.py", [], LSTM_CELLS),
    ("6", "run_sindy_ensemble.py", [], ENS_CELLS),
    ("6", "run_sindy_ensemble.py", ENS_CELLS + ["--aggregate"], None),
    ("6", "sindy_equations.py", ["--variant", "full"], None),
    ("6", "sindy_equations.py", ["--variant", "no_bv"], None),
    ("7", "run_test.py", [], None),
    ("8", "run_panasonic.py", [], None),
    ("9", "plot_data_overview.py", [], None),
    ("9", "plot_results.py", [], None),
    ("9", "download_weather.py", [], None),
    ("9", "winter_exposure.py", [], None),
]
EXPLORATORY = [
    ("10", "lstm_shap.py", [], None),
    ("10", "lstm_coverage.py", [], None),
    ("10", "conformal_coverage.py", [], None),
    ("10", "ecm_neural.py", [], None),
    ("10", "ecm_adaptation.py", [], None),
    ("10", "eis_ecm.py", [], None),
]


def step_key(step):
    # "2b" -> (2, "b"), so steps sort numerically; bad input raises ValueError
    digits = "".join(c for c in step if c.isdigit())
    return int(digits), step[len(digits):]


def expand(steps):
    # One job per cell, cell goes first in the args
    for step, script, args, cells in steps:
        for cell in cells or [None]:
            yield step, script, ([cell] if cell else []) + args, cell


def log_timing(row):
    new = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "script", "cell", "args", "start", "seconds", "ok"])
        w.writerow(row)


def run(step, script, args, cell):
    print(f"\n=== step {step}: {script} {' '.join(args)}", flush=True)
    start = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    ok = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT).returncode == 0
    sec = time.perf_counter() - t0

    log_timing([step, script, cell or "", " ".join(args), start, round(sec, 1), ok])
    print(f"=== {script} {cell or ''}: {sec / 60:.1f} min, {'ok' if ok else 'FAILED'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=step_key, default="1", help="first step, e.g. 5a or 7")
    ap.add_argument("--exploratory", action="store_true", help="also run exploratory analyses (step 10)")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    a = ap.parse_args()

    steps = CORE + (EXPLORATORY if a.exploratory else [])
    steps = [s for s in steps if step_key(s[0]) >= a.start]

    if a.list:
        for step, script, args, _ in expand(steps):
            print(f"{step:>3}  {script} {' '.join(args)}")
        return

    for job in expand(steps):
        if not run(*job) and not a.keep_going:
            print(f"stopped at step {job[0]}; fix it, then rerun with --from {job[0]}")
            sys.exit(1)

    print(f"\nall steps done, timings in {OUT}")


if __name__ == "__main__":
    main()