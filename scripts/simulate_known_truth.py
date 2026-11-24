"""
Simulated LG M50 data with a known true model, for the known-truth check

Truth: PyBaMM DFN, lumped thermal model, O'Regan et al. (2022) LG M50 parameters (temperature-dependent kinetics,
diffusion and conductivities, Butler-Volmer charge transfer)
The cell sits in a chamber at T_amb, starts soaked (cell temperature = T_amb) and full: at rest at 4.2 V, the LG M50
charge limit (the parameter set's own SOC 1 sits at its 4.4 V cut-off, above the real cell's range)

Profiles (current only, the simulator computes voltage and temperature):
    UDDS, HWFET    measured current of the B4 LG M50LT drive cycles at 25 degC (unit NMC068, FTP-72 = UDDS, HWFET),
                   scaled from the M50LT capacity to 5.0 Ah, realistic input statistics from a near-identical cell
    HPPC           25 and 40 degC only: at every 10 % SOC a 10 s 2C discharge pulse, 40 s rest, 10 s 1C charge pulse,
                   then 0.5C discharge to the next level and a 20 min rest, shows the Butler-Volmer nonlinearity
    C/20 discharge and charge   25 degC only, for the OCV curve (mean of the branches) and the SOC reference
                   capacity, the charge starts at rest at 2.55 V
Temperatures: 25 and 40 (train), 10 (selection), 30 and 35 (interpolation test), 0 and -10 degC (extrapolation test)

Measurement: 1 s simulator output plus Gaussian noise (1 mV, 0.05 K, fixed seed), every row measured
SOC coulomb counted with the 25 degC C/20 capacity, a run stops at 2.5 V or at the end of its profile
--force rewrites every run, otherwise runs already written are skipped
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pybamm

from warmtrained import config as cfg

CELL = "SimLGM50"
Q_NOM = cfg.SIM_CELLS[CELL]["nominal_ah"]
OUT = cfg.DATA_PROCESSED / CELL
PROFILE_SRC = {"UDDS": "NMC068_25deg__04_FTP-72", "HWFET": "NMC068_25deg__08_HWFET"}
TEMPS = [25, 40, 10, 30, 35, 0, -10]
NOISE_V, NOISE_T = 1e-3, 0.05
B4_Q = 4.85  # LG M50LT nominal capacity [Ah], scales its currents to the M50


def b4_profile(stem):
    # First segment of a B4 drive-cycle test, charge > 0, 1 s grid
    d = pd.read_parquet(cfg.DATA_PROCESSED / "TUBerlinM50LT" / "25" / f"{stem}.parquet")
    return d[d.segment == d.segment.iloc[0]].I.values * Q_NOM / B4_Q


def hppc_profile():
    I = []
    for _ in range(9):
        I += [-2 * Q_NOM] * 10 + [0.0] * 40 + [1 * Q_NOM] * 10 + [0.0] * 60
        I += [-0.5 * Q_NOM] * int(0.1 * 3600 / 0.5) + [0.0] * 1200
    return np.array(I)


def cover20_profile(sign=-1):
    return np.full(int(20 * 3600 * 1.05), sign * Q_NOM / 20)


def simulate(I, T_c, start="4.2 V", v_max=5.0):
    # DFN on a 1 s current profile (charge > 0) at ambient T_c [degC], frame up to the cut-off
    p = pybamm.ParameterValues("ORegan2022")
    TK = T_c + 273.15
    t = np.arange(len(I), dtype=float)
    p.update({"Ambient temperature [K]": TK, "Initial temperature [K]": TK,
              "Upper voltage cut-off [V]": v_max,  # 5 V lets cold regeneration peaks near full pass
              "Total heat transfer coefficient [W.m-2.K-1]": 25.0,  # chamber with forced air (default 10)
              "Current function [A]": pybamm.Interpolant(t, -I, pybamm.t, interpolator="linear")})
    sim = pybamm.Simulation(pybamm.lithium_ion.DFN({"thermal": "lumped"}), parameter_values=p,
                            solver=pybamm.IDAKLUSolver())
    sol = sim.solve(t_eval=[0, t[-1]], t_interp=t, initial_soc=start)
    n = len(sol["Time [s]"].entries)
    return pd.DataFrame({"t": sol["Time [s]"].entries, "V": sol["Voltage [V]"].entries,
                         "I": -sol["Current [A]"].entries,
                         "T": sol["X-averaged cell temperature [K]"].entries - 273.15}).iloc[:n]


def to_processed(df, T_c, q_ref, seed, soc0=1.0):
    # Simulator output in the processed format, with measurement noise and coulomb-counted SOC
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"t": df.t.values + 0.5})
    out["V"] = df.V.values + rng.normal(0, NOISE_V, len(df))
    out["I"] = df.I.values
    out["T"] = df["T"].values + rng.normal(0, NOISE_T, len(df))
    out["soc"] = soc0 + np.cumsum(np.r_[0.0, df.I.values[:-1]]) / 3600.0 / q_ref
    out["segment"] = 0
    out["measured"] = True
    out["T_amb"] = float(T_c)
    out["theta_dimless"] = cfg.T_REF_K * (1 / cfg.T_REF_K - 1 / (out.T_amb + 273.15))
    out["theta_cell"] = cfg.T_REF_K * (1 / cfg.T_REF_K - 1 / (out["T"] + 273.15))
    return out


def split_of(T_c):
    s = cfg.SPLITS[CELL]
    return next(k for k in ("train", "select", "test") if T_c in s[k])


def main(force):
    profiles = {name: b4_profile(stem) for name, stem in PROFILE_SRC.items()}
    runs = [(25, "C/20", cover20_profile()), (25, "C/20 charge", cover20_profile(+1))]
    runs += [(T, "HPPC", hppc_profile()) for T in (25, 40)]
    runs += [(T, name, I) for T in TEMPS for name, I in profiles.items()]

    man_path = cfg.DATA_PROCESSED / "manifest_sim.csv"
    rows = pd.read_csv(man_path).to_dict("records") if man_path.exists() and not force else []
    have = {r["file"] for r in rows}
    q_ref = soc_empty = None

    for k, (T_c, typ, I) in enumerate(runs):
        name = f"sim_{T_c}C_{typ.replace('/', '').replace(' ', '')}"
        f = OUT / str(T_c) / f"{name}.parquet"
        done = name in have and f.exists()
        if done and typ != "C/20":  # the C/20 run always reruns, it sets q_ref
            continue
        t0 = time.time()
        raw = simulate(I, T_c, "2.55 V", 4.2) if typ == "C/20 charge" else simulate(I, T_c)
        soc0 = 1.0
        if typ == "C/20":
            q_ref = float(-np.sum(raw.I.values) / 3600.0)  # capacity delivered to 2.5 V at 25 degC [Ah]
            dis = np.cumsum(-raw.I.values) / 3600.0
            soc_empty = 1.0 - float(np.interp(-2.55, -raw.V.values, dis)) / q_ref  # SOC at the 2.55 V rest
            if done:
                continue
        if typ == "C/20 charge":
            soc0 = soc_empty
        if q_ref is None:
            raise RuntimeError("the 25 degC C/20 run must come first")

        d = to_processed(raw, T_c, q_ref, seed=k, soc0=soc0)
        f.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(f, index=False)
        rows = [r for r in rows if r["file"] != name]
        rows.append(dict(cell=CELL, unit="sim", T_nominal=T_c, file=name, type=typ, q_ref_Ah=q_ref,
                         T_amb_source="set", T_amb=float(T_c), hours_kept=len(d) / 3600, measured_frac=1.0,
                         soc_min=float(d.soc.min()), soc_max=float(d.soc.max()), segments=1,
                         rows_processed=len(d), split=split_of(T_c)))
        pd.DataFrame(rows).to_csv(man_path, index=False)
        print(f"{name}: {len(d) / 3600:.2f} h, V {d.V.min():.3f} to {d.V.max():.3f}, T rise "
              f"{d['T'].max() - T_c:.2f} K, {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {man_path} ({len(rows)} runs, q_ref {q_ref:.3f} Ah)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)