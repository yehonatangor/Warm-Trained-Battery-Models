"""
LSTM black-box model with the same inputs and window protocol as the other models

Inputs per 1 s step:
    C-rate I/Q, SOC, ambient temperature, T_ref theta_amb, Arrhenius-scaled C-rate (I/Q) exp(-E0 theta_amb / R_g)
    (E0 from the single-E ECM, the same temperature prior SINDy gets), and three warm-up channels:
    measured V - U_oc(SOC), measured T - T_amb, and a flag, 1 where those are given, 0 elsewhere
Outputs: overpotential and temperature rise, V = U_oc(SOC) + eta, T = T_amb + dT, shared 25 degC OCV curve
Surface temperature is predicted, never an input

Windows: 5 min warm-up with the warm-up channels on (measured rows only), then 20 min with them off
Free run: only the first measured row is given

Training: random 25 min windows from the training pieces, 30 % with a warm-up of 1 to 300 rows to learn free running
Loss on measured horizon rows: (voltage error / 10 mV)^2 + (temperature error / 1 K)^2, as for the ECM and SINDy
Adam, early stopping on fixed windows of the early-stopping set (selection temperature, B2: one held-out training
cell), up to 100 epochs of 200 batches, lr halved after 3 epochs without improvement, stopped after 10
Inputs standardised with training statistics
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from warmtrained import config as cfg

WARM, HOR = 300, 1200
SEQ = WARM + HOR + 1
ETA_SCALE, DT_SCALE = 0.1, 5.0  # network output units, 100 mV and 5 K
V_UNIT, T_UNIT = 0.010, 1.0  # loss units, as for the ECM and SINDy
N_IN = 8


class Net(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(N_IN, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, state=None):
        h, state = self.lstm(x.contiguous(), state)
        return self.head(h), state

    def run(self, x, chunk=None):
        # Forward pass within cuDNN limits: long sequences in chunks with the hidden state carried over
        # (same result), repeated without cuDNN if it still refuses the input
        def go():
            if chunk is None or x.shape[1] <= chunk:
                return self(x)[0]
            outs, state = [], None
            for a in range(0, x.shape[1], chunk):
                o, state = self(x[:, a:a + chunk], state)
                outs.append(o)
            return torch.cat(outs, dim=1)

        try:
            return go()
        except RuntimeError as e:
            if "cuDNN" not in str(e) and "CUDNN" not in str(e):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                return go()


class LSTMModel:

    def __init__(self, ocv, nominal_ah, E0, hidden=64, layers=1, lr=1e-3, seed=0, device=None):
        self.ocv, self.Q, self.E0 = ocv, float(nominal_ah), float(E0)
        self.hidden, self.layers, self.lr, self.seed = hidden, layers, lr, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.net = Net(hidden, layers).to(self.device)
        self.mu, self.sd = np.zeros(5), np.ones(5)
        self.history = []

    # Features

    def _base(self, g):
        I, soc, Ta = g.I.values / self.Q, g.soc.values, g.T_amb.values
        th = 1.0 / cfg.T_REF_K - 1.0 / (Ta + 273.15)
        return np.column_stack([I, soc, Ta, cfg.T_REF_K * th, I * np.exp(-self.E0 * th / cfg.R_GAS)])

    def _piece(self, g):
        # Standardised base inputs, scaled measured targets and masks of one piece
        U = self.ocv(g.soc.values)
        meas = g.measured.values
        eta = (g.V.values - U) / ETA_SCALE
        dT = (g["T"].values - g.T_amb.values) / DT_SCALE
        return dict(x=((self._base(g) - self.mu) / self.sd).astype(np.float32), eta=eta.astype(np.float32),
                    dT=dT.astype(np.float32), mv=meas & np.isfinite(eta), mt=meas & np.isfinite(dT), U=U,
                    Ta=g.T_amb.values)

    @staticmethod
    def _window(p, a, n, warm_len):
        # Network input for rows a .. a+n-1, warm-up channels on for the first warm_len rows
        x = np.zeros((n, N_IN), np.float32)
        x[:, :5] = p["x"][a:a + n]
        on_v, on_t = np.zeros(n, bool), np.zeros(n, bool)
        on_v[:warm_len] = p["mv"][a:a + warm_len]
        on_t[:warm_len] = p["mt"][a:a + warm_len]
        x[on_v, 5] = p["eta"][a:a + n][on_v]
        x[on_t, 6] = p["dT"][a:a + n][on_t]
        x[:warm_len, 7] = (on_v | on_t)[:warm_len]
        return x

    def _batch(self, pieces, rng, size, fixed=False):
        # Random windows, fixed=True always uses the full warm-up
        X, Y, M = [], [], []
        for _ in range(size):
            p = pieces[rng.integers(len(pieces))]
            a = rng.integers(0, len(p["x"]) - SEQ + 1)
            w = WARM + 1 if fixed or rng.random() > 0.3 else int(rng.integers(1, WARM + 2))
            X.append(self._window(p, a, SEQ, w))
            y = np.column_stack([p["eta"][a:a + SEQ], p["dT"][a:a + SEQ]])
            m = np.column_stack([p["mv"][a:a + SEQ], p["mt"][a:a + SEQ]])
            m[:w] = False  # loss on the horizon only
            Y.append(np.nan_to_num(y))
            M.append(m)
        t = lambda a, d: torch.tensor(np.array(a), dtype=d, device=self.device)
        return t(X, torch.float32), t(Y, torch.float32), t(M, torch.bool)

    @staticmethod
    def _loss(pred, y, m):
        w = torch.tensor([ETA_SCALE / V_UNIT, DT_SCALE / T_UNIT], device=pred.device)
        return (((pred - y) * w) ** 2 * m).sum() / m.sum().clamp(min=1)

    def _pieces(self, df):
        return [self._piece(g.reset_index(drop=True)) for _, g in df.groupby("piece", sort=False) if len(g) >= SEQ]

    # Training

    def fit(self, train, stop, epochs=100, steps=200, batch=32, patience=10, n_stop=256, verbose=True):
        b = self._base(train)
        self.mu, self.sd = b.mean(axis=0), b.std(axis=0) + 1e-9
        tr, st = self._pieces(train), self._pieces(stop)
        if not tr or not st:
            raise ValueError("no piece long enough for a 25 min window")
        rng = np.random.default_rng(self.seed)
        Xs, Ys, Ms = self._batch(st, np.random.default_rng(12345), n_stop, fixed=True)  # fixed stopping windows
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            tot = 0.0
            for _ in range(steps):
                X, Y, M = self._batch(tr, rng, batch)
                loss = self._loss(self.net.run(X), Y, M)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            self.net.eval()
            with torch.no_grad():
                vl = float(np.mean([self._loss(self.net.run(Xs[i:i + 64]), Ys[i:i + 64], Ms[i:i + 64]).item()
                                    for i in range(0, n_stop, 64)]))
            sched.step(vl)
            self.history.append(dict(epoch=ep, train_loss=tot / steps, stop_loss=vl, lr=opt.param_groups[0]["lr"]))
            if verbose:
                print(f"  epoch {ep:2d}  train {tot / steps:8.3f}  stop {vl:8.3f}", flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_stop_loss = best
        return self

    # Prediction

    @torch.no_grad()
    def predict_windows(self, g, starts, horizon):
        self.net.eval()
        p = self._piece(g.reset_index(drop=True))
        Vout, Tout = np.empty((len(starts), horizon)), np.empty((len(starts), horizon))
        n = WARM + horizon + 1
        for c in range(0, len(starts), 128):
            ss = starts[c:c + 128]
            X = np.array([self._window(p, s - WARM, n, WARM + 1) for s in ss])
            out = self.net.run(torch.tensor(X, device=self.device)).cpu().numpy()[:, WARM + 1:]
            for k, s in enumerate(ss):
                idx = slice(s + 1, s + 1 + horizon)
                Vout[c + k] = p["U"][idx] + ETA_SCALE * out[k, :, 0]
                Tout[c + k] = p["Ta"][idx] + DT_SCALE * out[k, :, 1]
        return Vout, Tout

    @torch.no_grad()
    def free_run(self, g):
        self.net.eval()
        p = self._piece(g.reset_index(drop=True))
        first = int(np.argmax(p["mv"])) if p["mv"].any() else 0
        X = self._window(p, 0, len(p["x"]), first + 1)
        X[:first, 5:] = 0.0
        out = self.net.run(torch.tensor(X[None], device=self.device), chunk=5000).cpu().numpy()[0]
        return p["U"] + ETA_SCALE * out[:, 0], p["Ta"] + DT_SCALE * out[:, 1]

    # Saving and loading

    def save(self, path):
        path = Path(path)
        torch.save(self.net.state_dict(), path.with_suffix(".pt"))
        path.write_text(json.dumps(dict(Q=self.Q, E0=self.E0, hidden=self.hidden, layers=self.layers, lr=self.lr,
                                        seed=self.seed, mu=self.mu.tolist(), sd=self.sd.tolist(),
                                        best_stop_loss=getattr(self, "best_stop_loss", None),
                                        train_seconds=getattr(self, "train_seconds", None), device=self.device,
                                        history=self.history), indent=1))

    @classmethod
    def load(cls, path, ocv):
        path = Path(path)
        d = json.loads(path.read_text())
        m = cls(ocv, d["Q"], d["E0"], d["hidden"], d["layers"], d["lr"], d["seed"])
        m.net.load_state_dict(torch.load(path.with_suffix(".pt"), map_location=m.device))
        m.mu, m.sd, m.history = np.array(d["mu"]), np.array(d["sd"]), d["history"]
        m.best_stop_loss, m.train_seconds = d.get("best_stop_loss"), d.get("train_seconds")
        return m