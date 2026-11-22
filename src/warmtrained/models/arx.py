"""
ARX floor model: the simplest dynamic model every other model must beat

Voltage, y = V - U_oc(SOC):
    y[k] = a y[k-1] + (b0 + b0t theta) I[k] + (b1 + b1t theta) I[k-1] + c0 + c1 theta
One resistor plus one RC branch, resistances and offset linear in theta (theta_dimless, 0 at 25 degC)

Surface temperature, z = T - T_amb:
    z[k] = aT z[k-1] + d0 I[k]^2 + d1 |I[k]|
Lumped thermal mass, ohmic heating, cooled to ambient

Fitted by output error (simulation error on measured rows, files weighted equally), since scoring is in simulation
Windows: free run plus a decaying correction from the last measured warm-up value, y_free + a^j (y0 - y_free[s])
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

V_NAMES = ["a", "b0", "b0t", "b1", "b1t", "c0", "c1"]
T_NAMES = ["aT", "d0", "d1"]
MAX_RESIDUALS = 400_000  # largest residual vector in the output-error fit
MIN_PIECE_ROWS = 10  # shorter pieces are skipped in the fit


# Helpers

def logit(a):
    return np.log(a / (1 - a))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(a, u, y0):
    # y[0] = y0, y[k] = a y[k-1] + u[k] for k >= 1
    y = np.empty(len(u))
    y[0] = y0
    if len(u) > 1:
        y[1:], _ = lfilter([1.0], [1.0, -a], u[1:], zi=[a * y0])
    return y


def first_value(x, meas):
    # Initial state: first measured value of the piece (first value if none is measured)
    k = np.flatnonzero(meas & np.isfinite(x))
    return float(x[k[0]]) if len(k) else float(np.nan_to_num(x[0]))


def v_inputs(g):
    # Voltage equation inputs per step: I, theta I, I[k-1], theta I[k-1], 1, theta
    # I[k-1] at the first row is the first row's current
    I = g.I.values
    th = g.theta_dimless.values
    Ip = np.r_[I[0], I[:-1]]
    return np.column_stack([I, th * I, Ip, th * Ip, np.ones_like(I), th])


def t_inputs(g):
    # Temperature equation inputs per step: I^2 (ohmic heat), |I|
    I = g.I.values
    return np.column_stack([I ** 2, np.abs(I)])


def fit_one(pieces, verbose):
    # Fit one first-order equation y[k] = a y[k-1] + u[k] . b, returns [a, b...]
    # Each piece: u (inputs), y (target), m (measured rows), m_oe (rows scored), y0, w, w_oe (weights)

    # 1. One-step least squares on rows where y[k] and y[k-1] are both measured
    X, Y, W = [], [], []
    for p in pieces:
        y, m = p["y"], p["m"]
        ok = m[1:] & m[:-1]
        X.append(np.column_stack([y[:-1][ok], p["u"][1:][ok]]))
        Y.append(y[1:][ok])
        W.append(np.full(ok.sum(), p["w"]))
    X, Y, W = np.vstack(X), np.concatenate(Y), np.concatenate(W)
    x_ls, *_ = np.linalg.lstsq(X * W[:, None], Y * W, rcond=None)
    x_ls[0] = float(np.clip(x_ls[0], 0.5, 0.9999))

    # 2. Output error: simulate every piece, residual on scored rows, a kept in (0, 1) through a sigmoid
    def residuals(x):
        a = sigmoid(x[0])
        out = []
        for p in pieces:
            ys = simulate(a, p["u"] @ x[1:], p["y0"])
            out.append((ys - p["y"])[p["m_oe"]] * p["w_oe"])
        return np.concatenate(out)

    x0 = np.r_[logit(x_ls[0]), x_ls[1:]]
    sol = least_squares(residuals, x0, method="trf", x_scale="jac", max_nfev=200, verbose=verbose)
    return np.r_[sigmoid(sol.x[0]), sol.x[1:]]


# Model

class ARXModel:

    def __init__(self, ocv):
        self.ocv = ocv
        self.pv = None  # voltage parameters, V_NAMES order
        self.pt = None  # temperature parameters, T_NAMES order

    def targets(self, g):
        # Overpotential y = V - U_oc and temperature rise z = T - T_amb
        y = g.V.values - self.ocv(g.soc.values)
        z = g["T"].values - g.T_amb.values
        return y, z

    def free_run(self, g):
        # Simulate the whole piece from its first measured values
        y, z = self.targets(g)
        meas = g.measured.values
        ys = simulate(self.pv[0], v_inputs(g) @ self.pv[1:], first_value(y, meas))
        zs = simulate(self.pt[0], t_inputs(g) @ self.pt[1:], first_value(z, meas))
        return ys + self.ocv(g.soc.values), zs + g.T_amb.values

    def predict_windows(self, g, starts, horizon):
        # Free run corrected by the measured value at each window start, decaying as a^j
        y, z = self.targets(g)
        Vf, Tf = self.free_run(g)
        yf, zf = Vf - self.ocv(g.soc.values), Tf - g.T_amb.values

        j = np.arange(1, horizon + 1)
        idx = starts[:, None] + j[None, :]
        a, aT = self.pv[0], self.pt[0]

        y0 = np.where(np.isfinite(y[starts]), y[starts], yf[starts])  # measured value at the start, else free run
        z0 = np.where(np.isfinite(z[starts]), z[starts], zf[starts])
        yw = yf[idx] + (a ** j)[None, :] * (y0 - yf[starts])[:, None]
        zw = zf[idx] + (aT ** j)[None, :] * (z0 - zf[starts])[:, None]
        return yw + self.ocv(g.soc.values)[idx], zw + g.T_amb.values[idx]

    def fit(self, df, verbose=0):
        # Fit both equations on the training table
        files, v_pieces, t_pieces = [], [], []
        for piece, g in df.groupby("piece", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < MIN_PIECE_ROWS:
                continue
            y, z = self.targets(g)
            meas = g.measured.values
            files.append(g.file.iat[0])
            v_pieces.append(dict(u=v_inputs(g), y=y, m=meas & np.isfinite(y), y0=first_value(y, meas)))
            t_pieces.append(dict(u=t_inputs(g), y=z, m=meas & np.isfinite(z), y0=first_value(z, meas)))

        # Each file weighted once: weight = 1 / sqrt(measured voltage rows of its file)
        n_file = {}
        for f, p in zip(files, v_pieces):
            n_file[f] = n_file.get(f, 0) + int(p["m"].sum())

        # Score every stride-th measured row so the residuals stay under MAX_RESIDUALS
        # (the simulation still runs over every row), thinning a 1 s series this way does not bias the fit
        stride = max(1, int(np.ceil(sum(n_file.values()) / MAX_RESIDUALS)))

        for f, pv, pt in zip(files, v_pieces, t_pieces):
            w = 1.0 / np.sqrt(max(n_file[f], 1))
            for p in (pv, pt):
                thin = np.zeros_like(p["m"])
                thin[::stride] = True
                p["m_oe"] = p["m"] & thin
                p["w"] = w
                p["w_oe"] = w * np.sqrt(stride)

        self.pv = fit_one(v_pieces, verbose)
        self.pt = fit_one(t_pieces, verbose)
        return self

    def params(self):
        return {"voltage": dict(zip(V_NAMES, map(float, self.pv))),
                "temperature": dict(zip(T_NAMES, map(float, self.pt)))}

    def save(self, path):
        Path(path).write_text(json.dumps(self.params(), indent=2))

    @classmethod
    def load(cls, path, ocv):
        d = json.loads(Path(path).read_text())
        m = cls(ocv)
        m.pv = np.array([d["voltage"][k] for k in V_NAMES])
        m.pt = np.array([d["temperature"][k] for k in T_NAMES])
        return m