"""
Equivalent-circuit model with Arrhenius resistances, the physics baseline

Electrical, series resistance plus two RC branches:
    V = U_oc(SOC) + R0(T) I + v1 + v2,   dv_i/dt = -v_i / (R_i(T) C_i) + I / C_i
Thermal, one lumped temperature (measured surface temperature is the target, never an input):
    C_th dT/dt = I (V - U_oc) - hA (T - T_amb)
Temperature law per resistance, theta = 1/T_ref - 1/T, T the predicted cell temperature (sees self-heating):
    single:       R_k = R_k,25 exp(-E_k theta / R_g)
    two_process:  R_k = R_k,25 [(1 - w_k) exp(-E_k theta / R_g) + w_k exp(-E_2 theta / R_g)], E_2 fixed,
                  w_k in [0, 1], E_k in [0, E_2]
Capacitances constant, so tau_i = R_i C_i varies through R_i, branches ordered tau_1 < tau_2 at 25 degC, SOC an input

1 s grid: exact zero-order-hold RC update with resistances at the step start, explicit Euler for temperature
Output-error fit, files weighted equally, in three stages:
    A thermal alone, heated by the measured I (V - U_oc)
    B electrical alone, measured temperature in the Arrhenius factors
    C fully coupled, starting from A and B
Residuals in 10 mV and 1 K units, so 10 mV weighs as much as 1 K
Windows: RC states fitted by least squares to the last 5 min of measured voltage (measured temperature),
temperature starts from its measured value, then open loop over the horizon on current, SOC and ambient
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

from warmtrained import config as cfg

R_GAS = cfg.R_GAS
T_REF = cfg.T_REF_K
V_SCALE = 0.010  # voltage residual unit, 10 mV
T_SCALE = 1.0  # temperature residual unit, 1 K
MAX_RESIDUALS = 400_000
WARMUP = 300  # window warm-up [s]


# Numba kernels

@njit(cache=True, error_model="numpy")
def _factor(T_c, E, w, E2):
    # Arrhenius factor of one resistance at T_c [degC], single energy E or two-process mix with E2, 1 at 25 degC
    T_c = min(max(T_c, -60.0), 120.0)  # a diverging predicted temperature must not crash the run
    th = 1.0 / T_REF - 1.0 / (T_c + 273.15)
    return (1.0 - w) * np.exp(-E * th / R_GAS) + w * np.exp(-E2 * th / R_GAS)


@njit(cache=True, error_model="numpy")
def _simulate(I, U, T_amb, Tm, R25, E, w, E2, C1, C2, Cth, hA, v10, v20, T0, use_measured_T, dt):
    # Simulate one piece, returns V, T and the two RC voltages
    n = len(I)
    V, T, v1, v2 = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    v1[0], v2[0], T[0] = v10, v20, T0
    for k in range(n):
        Tk = Tm[k] if use_measured_T else T[k]
        R0 = R25[0] * _factor(Tk, E[0], w[0], E2)
        R1 = R25[1] * _factor(Tk, E[1], w[1], E2)
        R2 = R25[2] * _factor(Tk, E[2], w[2], E2)
        V[k] = U[k] + R0 * I[k] + v1[k] + v2[k]
        if k + 1 < n:
            a1 = np.exp(-dt / (R1 * C1))
            a2 = np.exp(-dt / (R2 * C2))
            v1[k + 1] = a1 * v1[k] + R1 * (1.0 - a1) * I[k]
            v2[k + 1] = a2 * v2[k] + R2 * (1.0 - a2) * I[k]
            T[k + 1] = T[k] + dt / Cth * (I[k] * (V[k] - U[k]) - hA * (T[k] - T_amb[k]))
    return V, T, v1, v2


@njit(cache=True, error_model="numpy")
def _thermal(I, Vm, U, T_amb, Cth, hA, T0, dt):
    # Thermal part alone, heated by the measured overpotential (stage A)
    n = len(I)
    T = np.empty(n)
    T[0] = T0
    for k in range(n - 1):
        T[k + 1] = T[k] + dt / Cth * (I[k] * (Vm[k] - U[k]) - hA * (T[k] - T_amb[k]))
    return T


@njit(cache=True, error_model="numpy")
def _windows(starts, horizon, warm, I, U, T_amb, Vm, Tm, meas, R25, E, w, E2, C1, C2, Cth, hA, dt):
    # Open-loop predictions for every window start
    nw = len(starts)
    Vout, Tout = np.empty((nw, horizon)), np.empty((nw, horizon))
    for j in range(nw):
        s = starts[j]
        a = max(s - warm, 0)
        n = s - a + 1
        # Warm-up: forced response from zero states and decay of a unit initial state, measured temperature
        F1, F2, D1, D2 = np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)
        for k in range(n - 1):
            i = a + k
            R1 = R25[1] * _factor(Tm[i], E[1], w[1], E2)
            R2 = R25[2] * _factor(Tm[i], E[2], w[2], E2)
            a1 = np.exp(-dt / (R1 * C1))
            a2 = np.exp(-dt / (R2 * C2))
            F1[k + 1] = a1 * F1[k] + R1 * (1.0 - a1) * I[i]
            F2[k + 1] = a2 * F2[k] + R2 * (1.0 - a2) * I[i]
            D1[k + 1] = a1 * D1[k]
            D2[k + 1] = a2 * D2[k]
        # Least squares for initial states x1, x2 on measured warm-up rows (small ridge for rests)
        s11, s12, s22, b1, b2 = 1e-9, 0.0, 1e-9, 0.0, 0.0
        for k in range(n):
            i = a + k
            if meas[i]:
                R0 = R25[0] * _factor(Tm[i], E[0], w[0], E2)
                r = Vm[i] - U[i] - R0 * I[i] - F1[k] - F2[k]
                s11 += D1[k] * D1[k]
                s12 += D1[k] * D2[k]
                s22 += D2[k] * D2[k]
                b1 += D1[k] * r
                b2 += D2[k] * r
        det = s11 * s22 - s12 * s12
        if not det > 0.0:  # degenerate warm-up (identical branches): zero initial states
            det = np.inf
        x1 = (s22 * b1 - s12 * b2) / det
        x2 = (s11 * b2 - s12 * b1) / det
        v1 = x1 * D1[n - 1] + F1[n - 1]
        v2 = x2 * D2[n - 1] + F2[n - 1]
        T = Tm[s]
        # Open loop over the horizon, predicted temperature
        for h in range(horizon + 1):
            i = s + h
            R0 = R25[0] * _factor(T, E[0], w[0], E2)
            R1 = R25[1] * _factor(T, E[1], w[1], E2)
            R2 = R25[2] * _factor(T, E[2], w[2], E2)
            Vk = U[i] + R0 * I[i] + v1 + v2
            if h > 0:
                Vout[j, h - 1] = Vk
                Tout[j, h - 1] = T
            a1 = np.exp(-dt / (R1 * C1))
            a2 = np.exp(-dt / (R2 * C2))
            v1 = a1 * v1 + R1 * (1.0 - a1) * I[i]
            v2 = a2 * v2 + R2 * (1.0 - a2) * I[i]
            T = T + dt / Cth * (I[i] * (Vk - U[i]) - hA * (T - T_amb[i]))
    return Vout, Tout


# Model

class ECMModel:
    # law: "single" or "two_process" (E2 [J/mol] fixed)

    def __init__(self, ocv, law="single", E2=54.8e3, nominal_ah=3.0):
        self.ocv, self.law, self.E2, self.nominal = ocv, law, float(E2), nominal_ah
        # Start: resistances scale with 1/capacity, heat capacity ~23 J/K per Ah
        # two_process starts at w = 0.3, a start on the w = 0 bound can stall at the single law
        q = nominal_ah
        self.p = dict(R25=np.array([0.05, 0.02, 0.03]) / q, tau=np.array([30.0, 1000.0]), E=np.full(3, 20e3),
                      w=np.full(3, 0.3) if law == "two_process" else np.zeros(3), Cth=23.0 * q, hA=0.05)
        self.dt = cfg.RESAMPLE_S

    # Parameter vector <-> dict

    def _names(self, part):
        el = ["R0", "R1", "R2", "tau1", "tau2", "E0", "E1", "E2"]
        if self.law == "two_process":
            el += ["w0", "w1", "w2"]
        th = ["Cth", "hA"]
        return {"el": el, "th": th, "all": el + th}[part]

    def _spec(self):
        # Packed parameter: name -> (key in p, index, log-scaled, lo, hi), E packed in kJ/mol
        top = self.E2 / 1e3 if self.law == "two_process" else 150.0
        s = {f"R{k}": ("R25", k, True, 1e-5, 2.0) for k in range(3)}
        s.update({f"E{k}": ("E", k, False, 0.0, top) for k in range(3)})
        s.update({f"w{k}": ("w", k, False, 0.0, 1.0) for k in range(3)})
        s.update(tau1=("tau", 0, True, 1.0, 300.0), tau2=("tau", 1, True, 300.0, 3e4),
                 Cth=("Cth", None, True, 1.0, 5e3), hA=("hA", None, True, 1e-4, 5.0))
        return s

    def _pack(self, part):
        spec, x, lo, hi = self._spec(), [], [], []
        for nm in self._names(part):
            key, i, log, a, b = spec[nm]
            v = self.p[key] if i is None else self.p[key][i]
            if key == "E":
                v = min(v / 1e3, b)
            f = np.log if log else float
            x.append(f(v)); lo.append(f(a)); hi.append(f(b))
        lo, hi = np.array(lo), np.array(hi)
        return np.clip(x, lo + 1e-9, hi - 1e-9), (lo, hi)

    def _unpack(self, x, part):
        spec = self._spec()
        p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.p.items()}
        for nm, val in zip(self._names(part), x):
            key, i, log, *_ = spec[nm]
            val = np.exp(val) if log else (val * 1e3 if key == "E" else val)
            if i is None:
                p[key] = val
            else:
                p[key][i] = val
        return p

    def _args(self, p):
        R25 = p["R25"]
        return R25, p["E"], p["w"], self.E2, p["tau"][0] / R25[1], p["tau"][1] / R25[2], p["Cth"], p["hA"]

    @staticmethod
    def _arrays(g):
        return (g.I.values.astype(float), g.T_amb.values.astype(float), g.V.values.astype(float),
                g["T"].values.astype(float), g.measured.values.astype(bool))

    @staticmethod
    def _first(Tm, meas, Ta):
        # First measured temperature, ambient if the piece has none
        k = np.flatnonzero(meas & np.isfinite(Tm))
        return float(Tm[k[0]]) if len(k) else float(Ta[0])

    @staticmethod
    def _fill(Tm, Ta):
        # Measured temperature on every row (interpolated over gaps), ambient if the piece has none
        if not np.isfinite(Tm).any():
            return Ta.copy()
        return pd.Series(Tm).interpolate(limit_direction="both").values

    # Prediction

    def free_run(self, g):
        I, Ta, Vm, Tm, meas = self._arrays(g)
        V, T, _, _ = _simulate(I, self.ocv(g.soc.values), Ta, Tm, *self._args(self.p), 0.0, 0.0,
                               self._first(Tm, meas, Ta), False, self.dt)
        return V, T

    def predict_windows(self, g, starts, horizon):
        I, Ta, Vm, Tm, meas = self._arrays(g)
        return _windows(np.asarray(starts, dtype=np.int64), int(horizon), WARMUP, I, self.ocv(g.soc.values), Ta,
                        Vm, self._fill(Tm, Ta), meas, *self._args(self.p), self.dt)

    # Fitting

    def _prepare(self, df):
        pieces, n_file = [], {}
        for _, g in df.groupby("piece", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < 10:
                continue
            I, Ta, Vm, Tm, meas = self._arrays(g)
            f, mv = g.file.iat[0], meas & np.isfinite(Vm)
            pieces.append(dict(file=f, I=I, Ta=Ta, Vm=Vm, Tm=self._fill(Tm, Ta), U=self.ocv(g.soc.values),
                               mv=mv, mt=meas & np.isfinite(Tm), T0=self._first(Tm, meas, Ta)))
            n_file[f] = n_file.get(f, 0) + int(mv.sum())
        # Score every stride-th row to stay under MAX_RESIDUALS, each file weighted once
        stride = max(1, int(np.ceil(sum(n_file.values()) / MAX_RESIDUALS)))
        for p in pieces:
            thin = np.zeros(len(p["I"]), dtype=bool)
            thin[::stride] = True
            p["mv"] &= thin
            p["mt"] &= thin
            p["w"] = np.sqrt(stride / max(n_file[p["file"]], 1))
        return pieces

    def _residual_fn(self, pieces, part):
        # Weighted residual vector of one stage ("th", "el" or "all") as a function of the packed x
        dt = self.dt

        def res_thermal(x):
            p = self._unpack(x, "th")
            return np.concatenate([
                (_thermal(q["I"], q["Vm"], q["U"], q["Ta"], p["Cth"], p["hA"], q["T0"], dt) - q["Tm"])[q["mt"]]
                / T_SCALE * q["w"] for q in pieces])

        def res_electric(x):
            args = self._args(self._unpack(x, "el"))
            out = []
            for q in pieces:
                V = _simulate(q["I"], q["U"], q["Ta"], q["Tm"], *args, 0.0, 0.0, q["T0"], True, dt)[0]
                out.append((V - q["Vm"])[q["mv"]] / V_SCALE * q["w"])
            return np.concatenate(out)

        def res_all(x):
            args = self._args(self._unpack(x, "all"))
            out = []
            for q in pieces:
                V, T, _, _ = _simulate(q["I"], q["U"], q["Ta"], q["Tm"], *args, 0.0, 0.0, q["T0"], False, dt)
                out.append((V - q["Vm"])[q["mv"]] / V_SCALE * q["w"])
                out.append((T - q["Tm"])[q["mt"]] / T_SCALE * q["w"])
            return np.concatenate(out)

        return {"th": res_thermal, "el": res_electric, "all": res_all}[part]

    def fit(self, df, verbose=0):
        pieces = self._prepare(df)
        for part in ("th", "el", "all"):
            x0, bounds = self._pack(part)
            sol = least_squares(self._residual_fn(pieces, part), x0, bounds=bounds, method="trf", x_scale="jac",
                                max_nfev=300, verbose=verbose)
            self.p = self._unpack(sol.x, part)
            self.cost_ = {**getattr(self, "cost_", {}), part: float(sol.cost)}
        return self

    # Saving and loading

    def params(self):
        p = self.p
        return {"law": self.law, "E2_J_per_mol": self.E2 if self.law == "two_process" else None,
                "R25_ohm": p["R25"].tolist(), "tau_s": p["tau"].tolist(), "E_J_per_mol": p["E"].tolist(),
                "w": p["w"].tolist(), "Cth_J_per_K": float(p["Cth"]), "hA_W_per_K": float(p["hA"]),
                "fit_cost": getattr(self, "cost_", {})}

    def save(self, path):
        Path(path).write_text(json.dumps(self.params(), indent=2))

    @classmethod
    def load(cls, path, ocv):
        d = json.loads(Path(path).read_text())
        m = cls(ocv, d["law"], d["E2_J_per_mol"] or 54.8e3)
        m.p = dict(R25=np.array(d["R25_ohm"]), tau=np.array(d["tau_s"]), E=np.array(d["E_J_per_mol"]),
                   w=np.array(d["w"]), Cth=d["Cth_J_per_K"], hA=d["hA_W_per_K"])
        return m