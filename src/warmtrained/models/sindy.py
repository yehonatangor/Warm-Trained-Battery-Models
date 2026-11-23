"""
Sparse identification (SINDy) with an Arrhenius library, started from the fitted single-E ECM

States: polarisation voltages v1, v2 and lumped temperature T, SOC an input
f_k(T) = exp(-E_k theta / R_g), theta = 1/T_ref - 1/T, the Arrhenius factors of the ECM's three resistances,
E_k fixed at the single-E ECM fit (never refitted), zc = SOC - 0.5 centres the SOC terms

Output, coefficients xi:
    V = U_oc(SOC) + v1 + v2 + sum_j xi_j phi_j
    phi: 0  f0 I                                  series resistance (ECM term)
         1  f0 I zc,  2  f0 I zc^2                SOC-dependent resistance
         3-6  (2 R_g T / F) asinh(I / (2 I0(T)))  Butler-Volmer loss, I0(T) = I0_ref exp(E0 theta / R_g),
                                                  one column per I0_ref in BV_GRID x nominal capacity [A]
         7  1,   8  T_ref theta                   offset and temperature-linear shift
Branch i (i = 1, 2), linear in v_i:
    dv_i/dt = -(a_i / f_i + c_i |I| / Q + d_i zc / f_i) v_i + (b_i + e_i zc) I
    a_i = 1/tau_i, b_i = 1/C_i (ECM terms), c_i, d_i, e_i candidates
Temperature:
    dT/dt = h0 I (V - U_oc) - h1 (T - T_amb) + h2 I^2 + h3 |I|
    h0 = 1/C_th, h1 = hA/C_th (ECM terms), h2, h3 candidates
Candidates at zero with the ECM core values gives exactly the single-E ECM

Output-error fit (10 mV and 1 K residual units, each file weighted once): all active coefficients by bounded least
squares, then terms with small contribution are zeroed and the rest refitted until the set stops changing
Contribution: RMS over training rows of a term's share of its equation, in volts for the output (dropped below
OUT_TOL), relative to its equation's largest term for the states (dropped below REL_TOL)
Core state terms (a_i, b_i, h0, h1) are never dropped, f0 I can be, resistances, rates and heat terms bounded >= 0

1 s grid: exact zero-order-hold branch step with rate and forcing frozen over the step, explicit Euler for temperature
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares

from warmtrained import config as cfg

R_GAS = cfg.R_GAS
T_REF = cfg.T_REF_K
FARADAY = 96485.33
V_SCALE, T_SCALE = 0.010, 1.0  # residual units, 10 mV and 1 K
MAX_RESIDUALS = 400_000
WARMUP = 300  # window warm-up [s]
BV_GRID = np.array([0.5, 1.0, 2.0, 4.0])  # I0_ref per Ah of nominal capacity [A/Ah]
OUT_TOL = 1e-3  # output terms under 1 mV RMS are dropped
REL_TOL = 0.05  # state terms under 5 % of their equation's largest term are dropped
MAX_ROUNDS = 6
T_GUARD = (-60.0, 120.0)  # degC, only reached by a diverging trial step

OUT_NAMES = ["R0_f0I", "R0_f0I_z", "R0_f0I_z2", "BV_0.5", "BV_1", "BV_2", "BV_4", "offset", "theta_shift"]
DYN_NAMES = ["a_rate", "c_absI", "d_z_rate", "b_input", "e_z_input"]
TH_NAMES = ["h0_heat", "h1_loss", "h2_I2", "h3_absI"]
N_OUT, N_DYN, N_TH = len(OUT_NAMES), len(DYN_NAMES), len(TH_NAMES)
# Lower bounds: 0 for non-negative physical terms, -inf for free candidates
LO_OUT = np.array([0, -np.inf, -np.inf, 0, 0, 0, 0, -np.inf, -np.inf], dtype=float)
LO_DYN = np.array([1e-7, -np.inf, -np.inf, 0, -np.inf], dtype=float)
LO_TH = np.array([1e-6, 0, 0, -np.inf], dtype=float)
CORE_DYN = (0, 3)
CORE_TH = (0, 1)
BV_IDX = (3, 4, 5, 6)


# Numba kernels

@njit(cache=True, error_model="numpy")
def _out_terms(I, z, T_c, E0, I0ref, Q, phi):
    # Output library columns into phi
    Tg = min(max(T_c, T_GUARD[0]), T_GUARD[1])
    TK = Tg + 273.15
    th = 1.0 / T_REF - 1.0 / TK
    f0 = np.exp(-E0 * th / R_GAS)
    zc = z - 0.5
    phi[0] = f0 * I
    phi[1] = f0 * I * zc
    phi[2] = f0 * I * zc * zc
    for g in range(4):
        i0 = I0ref[g] * np.exp(E0 * th / R_GAS)
        phi[3 + g] = 2.0 * R_GAS * TK / FARADAY * np.arcsinh(I / (2.0 * i0))
    phi[7] = 1.0
    phi[8] = T_REF * th


@njit(cache=True, error_model="numpy")
def _branch_step(v, I, z, T_c, E, cd, Q, dt):
    # One exact ZOH step of dv/dt = -lam v + g, returns (alpha, beta) with v_next = alpha v + beta
    Tg = min(max(T_c, T_GUARD[0]), T_GUARD[1])
    th = 1.0 / T_REF - 1.0 / (Tg + 273.15)
    f = np.exp(-E * th / R_GAS)
    zc = z - 0.5
    lam = max(cd[0] / f + cd[1] * abs(I) / Q + cd[2] * zc / f, 1e-7)
    g = (cd[3] + cd[4] * zc) * I
    a = np.exp(-lam * dt)
    return a, g * (1.0 - a) / lam


@njit(cache=True, error_model="numpy")
def _simulate(I, z, U, Ta, Tm, use_meas, co, c1, c2, ct, E, I0ref, Q, v10, v20, T0, dt):
    # Simulate one piece, returns V, T and the two branch voltages
    n = len(I)
    V, T, v1, v2 = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    phi = np.empty(9)
    v1[0], v2[0], T[0] = v10, v20, T0
    for k in range(n):
        Tk = Tm[k] if use_meas else T[k]
        _out_terms(I[k], z[k], Tk, E[0], I0ref, Q, phi)
        out = 0.0
        for j in range(9):
            out += co[j] * phi[j]
        V[k] = U[k] + v1[k] + v2[k] + out
        if k + 1 < n:
            a, b = _branch_step(v1[k], I[k], z[k], Tk, E[1], c1, Q, dt)
            v1[k + 1] = a * v1[k] + b
            a, b = _branch_step(v2[k], I[k], z[k], Tk, E[2], c2, Q, dt)
            v2[k + 1] = a * v2[k] + b
            dT = ct[0] * I[k] * (V[k] - U[k]) - ct[1] * (T[k] - Ta[k]) + ct[2] * I[k] * I[k] + ct[3] * abs(I[k])
            T[k + 1] = T[k] + dt * dT
    return V, T, v1, v2


@njit(cache=True, error_model="numpy")
def _windows(starts, horizon, warm, I, z, U, Ta, Vm, Tm, meas, co, c1, c2, ct, E, I0ref, Q, dt):
    # Open-loop predictions per window start, branch states fitted on the warm-up as in the ECM
    nw = len(starts)
    Vout, Tout = np.empty((nw, horizon)), np.empty((nw, horizon))
    phi = np.empty(9)
    for j in range(nw):
        s = starts[j]
        a0 = max(s - warm, 0)
        n = s - a0 + 1
        # Warm-up: forced response from zero states and decay of a unit initial state, measured temperature
        F1, F2, D1, D2 = np.zeros(n), np.zeros(n), np.ones(n), np.ones(n)
        for k in range(n - 1):
            i = a0 + k
            al, be = _branch_step(0.0, I[i], z[i], Tm[i], E[1], c1, Q, dt)
            F1[k + 1] = al * F1[k] + be
            D1[k + 1] = al * D1[k]
            al, be = _branch_step(0.0, I[i], z[i], Tm[i], E[2], c2, Q, dt)
            F2[k + 1] = al * F2[k] + be
            D2[k + 1] = al * D2[k]
        # Least squares for initial states on measured warm-up rows (small ridge for rests)
        s11, s12, s22, b1, b2 = 1e-9, 0.0, 1e-9, 0.0, 0.0
        for k in range(n):
            i = a0 + k
            if meas[i]:
                _out_terms(I[i], z[i], Tm[i], E[0], I0ref, Q, phi)
                out = 0.0
                for q in range(9):
                    out += co[q] * phi[q]
                r = Vm[i] - U[i] - out - F1[k] - F2[k]
                s11 += D1[k] * D1[k]
                s12 += D1[k] * D2[k]
                s22 += D2[k] * D2[k]
                b1 += D1[k] * r
                b2 += D2[k] * r
        det = s11 * s22 - s12 * s12
        x1 = (s22 * b1 - s12 * b2) / det
        x2 = (s11 * b2 - s12 * b1) / det
        v1 = x1 * D1[n - 1] + F1[n - 1]
        v2 = x2 * D2[n - 1] + F2[n - 1]
        T = Tm[s]
        # Open loop over the horizon, predicted temperature
        for h in range(horizon + 1):
            i = s + h
            _out_terms(I[i], z[i], T, E[0], I0ref, Q, phi)
            out = 0.0
            for q in range(9):
                out += co[q] * phi[q]
            Vk = U[i] + v1 + v2 + out
            if h > 0:
                Vout[j, h - 1] = Vk
                Tout[j, h - 1] = T
            al, be = _branch_step(v1, I[i], z[i], T, E[1], c1, Q, dt)
            v1 = al * v1 + be
            al, be = _branch_step(v2, I[i], z[i], T, E[2], c2, Q, dt)
            v2 = al * v2 + be
            T = T + dt * (ct[0] * I[i] * (Vk - U[i]) - ct[1] * (T - Ta[i]) + ct[2] * I[i] * I[i] + ct[3] * abs(I[i]))
    return Vout, Tout


# Model

class SINDyModel:
    # library: "full" or "no_bv" (Butler-Volmer columns removed from the start)

    def __init__(self, ocv, ecm_params, nominal_ah, library="full"):
        if ecm_params["law"] != "single":
            raise ValueError("SINDy starts from the single-E ECM")
        self.ocv, self.Q, self.library = ocv, float(nominal_ah), library
        self.E = np.array(ecm_params["E_J_per_mol"], dtype=float)  # fixed, never refitted
        self.I0ref = BV_GRID * self.Q
        R25, tau = np.array(ecm_params["R25_ohm"]), np.array(ecm_params["tau_s"])
        Cth, hA = ecm_params["Cth_J_per_K"], ecm_params["hA_W_per_K"]
        self.co = np.zeros(N_OUT)
        self.co[0] = R25[0]
        self.c1 = np.array([1 / tau[0], 0, 0, R25[1] / tau[0], 0])  # b = 1/C = R/tau
        self.c2 = np.array([1 / tau[1], 0, 0, R25[2] / tau[1], 0])
        self.ct = np.array([1 / Cth, hA / Cth, 0, 0])
        self.active = {"out": np.ones(N_OUT, bool), "d1": np.ones(N_DYN, bool), "d2": np.ones(N_DYN, bool),
                       "th": np.ones(N_TH, bool)}
        if library == "no_bv":
            self.active["out"][list(BV_IDX)] = False
        self.dt = cfg.RESAMPLE_S
        self.history = []

    # Coefficient vector <-> arrays, active entries only

    def _blocks(self):
        return [("out", self.co, LO_OUT), ("d1", self.c1, LO_DYN), ("d2", self.c2, LO_DYN), ("th", self.ct, LO_TH)]

    def _pack(self):
        x = np.concatenate([arr[self.active[k]] for k, arr, _ in self._blocks()])
        lo = np.concatenate([low[self.active[k]] for k, _, low in self._blocks()])
        return np.maximum(x, lo + 1e-12), lo

    def _unpack(self, x):
        out, k = [], 0
        for key, arr, _ in self._blocks():
            m = self.active[key]
            a = np.zeros_like(arr)
            a[m] = x[k:k + m.sum()]
            k += m.sum()
            out.append(a)
        return tuple(out)

    def _coefs(self):
        return self.co, self.c1, self.c2, self.ct

    @staticmethod
    def _arrays(g):
        return (g.I.values.astype(float), g.soc.values.astype(float), g.T_amb.values.astype(float),
                g.V.values.astype(float), g["T"].values.astype(float), g.measured.values.astype(bool))

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
        I, z, Ta, Vm, Tm, meas = self._arrays(g)
        V, T, _, _ = _simulate(I, z, self.ocv(z), Ta, Tm, False, *self._coefs(), self.E, self.I0ref, self.Q,
                               0.0, 0.0, self._first(Tm, meas, Ta), self.dt)
        return V, T

    def predict_windows(self, g, starts, horizon):
        I, z, Ta, Vm, Tm, meas = self._arrays(g)
        return _windows(np.asarray(starts, dtype=np.int64), int(horizon), WARMUP, I, z, self.ocv(z), Ta, Vm,
                        self._fill(Tm, Ta), meas, *self._coefs(), self.E, self.I0ref, self.Q, self.dt)

    # Fitting

    def _prepare(self, df):
        pieces, n_file = [], {}
        for _, g in df.groupby("piece", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < 10:
                continue
            I, z, Ta, Vm, Tm, meas = self._arrays(g)
            f, mv = g.file.iat[0], meas & np.isfinite(Vm)
            pieces.append(dict(file=f, I=I, z=z, U=self.ocv(z), Ta=Ta, Vm=Vm, Tm=self._fill(Tm, Ta),
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

    def _residuals(self, x, pieces):
        co, c1, c2, ct = self._unpack(x)
        out = []
        for q in pieces:
            V, T, _, _ = _simulate(q["I"], q["z"], q["U"], q["Ta"], q["Tm"], False, co, c1, c2, ct, self.E,
                                   self.I0ref, self.Q, 0.0, 0.0, q["T0"], self.dt)
            out.append((V - q["Vm"])[q["mv"]] / V_SCALE * q["w"])
            out.append((T - q["Tm"])[q["mt"]] / T_SCALE * q["w"])
        r = np.concatenate(out)
        return np.where(np.isfinite(r), r, 1e3)  # a diverging trial step is penalised, not a crash

    def _contributions(self, pieces):
        # RMS contribution of every term along the simulated training trajectories
        acc = {k: np.zeros(n) for k, n in (("out", N_OUT), ("d1", N_DYN), ("d2", N_DYN), ("th", N_TH))}
        cnt = 0
        phi = np.empty(N_OUT)
        for q in pieces:
            V, T, v1, v2 = _simulate(q["I"], q["z"], q["U"], q["Ta"], q["Tm"], False, *self._coefs(), self.E,
                                     self.I0ref, self.Q, 0.0, 0.0, q["T0"], self.dt)
            idx = np.flatnonzero(q["mv"])
            I, zc = q["I"][idx], q["z"][idx] - 0.5
            th = 1.0 / T_REF - 1.0 / (T[idx] + 273.15)
            P = np.empty((len(idx), N_OUT))
            for r, k in enumerate(idx):
                _out_terms(q["I"][k], q["z"][k], T[k], self.E[0], self.I0ref, self.Q, phi)
                P[r] = phi
            acc["out"] += ((P * self.co) ** 2).sum(axis=0)
            for key, v, E, c in (("d1", v1[idx], self.E[1], self.c1), ("d2", v2[idx], self.E[2], self.c2)):
                f = np.exp(-E * th / R_GAS)
                terms = np.column_stack([-v / f, -np.abs(I) / self.Q * v, -zc / f * v, I, zc * I]) * c
                acc[key] += (terms ** 2).sum(axis=0)
            terms = np.column_stack([I * (V[idx] - q["U"][idx]), -(T[idx] - q["Ta"][idx]), I * I, np.abs(I)]) * self.ct
            acc["th"] += (terms ** 2).sum(axis=0)
            cnt += len(idx)
        return {k: np.sqrt(v / max(cnt, 1)) for k, v in acc.items()}

    def _prune(self, contrib):
        # Drop small non-core terms, returns whether the active set changed
        changed = False
        rules = (("out", OUT_TOL, False, ()), ("d1", REL_TOL, True, CORE_DYN), ("d2", REL_TOL, True, CORE_DYN),
                 ("th", REL_TOL, True, CORE_TH))
        for key, tol, relative, core in rules:
            c = contrib[key]
            limit = tol * c.max() if relative else tol
            for j in range(len(c)):
                if self.active[key][j] and j not in core and c[j] < limit:
                    self.active[key][j] = False
                    changed = True
        for key, arr, _ in self._blocks():
            arr[~self.active[key]] = 0.0
        return changed

    def _solve(self, pieces, verbose):
        x0, lo = self._pack()
        sol = least_squares(self._residuals, x0, args=(pieces,), bounds=(lo, np.full_like(lo, np.inf)),
                            method="trf", x_scale="jac", max_nfev=200, verbose=verbose)
        self.co, self.c1, self.c2, self.ct = self._unpack(sol.x)
        return float(sol.cost)

    def fit(self, df, verbose=0):
        t0 = time.time()
        pieces = self._prepare(df)
        self.cost_ecm_start = float(0.5 * np.sum(self._residuals(self._pack()[0], pieces) ** 2))
        for rnd in range(MAX_ROUNDS):
            cost = self._solve(pieces, verbose)
            contrib = self._contributions(pieces)
            self.history.append(dict(round=rnd, cost=cost, n_active=int(sum(a.sum() for a in self.active.values())),
                                     seconds=round(time.time() - t0, 1)))
            if not self._prune(contrib):
                break
        else:
            self._solve(pieces, verbose)  # refit the final set if the round limit stopped the loop
        self.contrib_ = self._contributions(pieces)
        self.fit_seconds = time.time() - t0
        return self

    # Diagnostics, saving and loading

    def bv_correlation(self, df):
        # Correlation of each Butler-Volmer column with f0 I on the given rows (measured temperature)
        d = df[df.measured & df["T"].notna()]
        if len(d) > 200_000:
            d = d.iloc[:: len(d) // 200_000]
        phi = np.empty(N_OUT)
        P = np.empty((len(d), N_OUT))
        for r, (i, z, t) in enumerate(zip(d.I.values, d.soc.values, d["T"].values)):
            _out_terms(i, z, t, self.E[0], self.I0ref, self.Q, phi)
            P[r] = phi
        return {OUT_NAMES[j]: float(np.corrcoef(P[:, 0], P[:, j])[0, 1]) for j in BV_IDX}

    def params(self):
        def block(names, arr, key):
            return {n: (float(v) if a else 0.0) for n, v, a in zip(names, arr, self.active[key])}

        sel = {k: [n for n, a in zip(names, self.active[k]) if a] for k, names in
               (("out", OUT_NAMES), ("d1", DYN_NAMES), ("d2", DYN_NAMES), ("th", TH_NAMES))}
        return {"library": self.library, "Q_Ah": self.Q, "E_fixed_J_per_mol": self.E.tolist(),
                "I0_ref_A": self.I0ref.tolist(),
                "output": block(OUT_NAMES, self.co, "out"), "branch1": block(DYN_NAMES, self.c1, "d1"),
                "branch2": block(DYN_NAMES, self.c2, "d2"), "thermal": block(TH_NAMES, self.ct, "th"),
                "selected": sel, "bv_selected": any(self.active["out"][list(BV_IDX)]),
                "contribution_rms": {k: v.tolist() for k, v in getattr(self, "contrib_", {}).items()},
                "fit_history": self.history, "fit_seconds": round(getattr(self, "fit_seconds", 0.0), 1),
                "cost_at_ecm_start": getattr(self, "cost_ecm_start", None)}

    def save(self, path, extra=None):
        Path(path).write_text(json.dumps({**self.params(), **(extra or {})}, indent=2))

    @classmethod
    def load(cls, path, ocv):
        d = json.loads(Path(path).read_text())
        m = cls.__new__(cls)
        m.ocv, m.Q, m.library = ocv, d["Q_Ah"], d["library"]
        m.E, m.I0ref, m.dt, m.history = np.array(d["E_fixed_J_per_mol"]), np.array(d["I0_ref_A"]), cfg.RESAMPLE_S, []
        m.co = np.array([d["output"][n] for n in OUT_NAMES])
        m.c1 = np.array([d["branch1"][n] for n in DYN_NAMES])
        m.c2 = np.array([d["branch2"][n] for n in DYN_NAMES])
        m.ct = np.array([d["thermal"][n] for n in TH_NAMES])
        m.active = {"out": m.co != 0, "d1": m.c1 != 0, "d2": m.c2 != 0, "th": m.ct != 0}
        return m