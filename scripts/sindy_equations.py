"""
Write the fitted SINDy equations of every cell as Markdown with LaTeX (results/sindy/equations_<variant>.md)
Per cell: selected terms with coefficients (zero terms left out), fixed activation energies,
and each term's inclusion probability if the ensemble was aggregated
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # runs without installing the package

import pandas as pd

from warmtrained import config as cfg

CELLS = ["Samsung30T", "LGHG2", "Tesla2170", "TUBerlinM50LT", "LGM50", "SimLGM50"]
LABEL = {"Samsung30T": "B1 Samsung 30T", "LGM50": "B2 LG M50", "LGHG2": "B3 LG HG2",
         "TUBerlinM50LT": "B4 LG M50LT", "Tesla2170": "B5 Tesla 2170", "SimLGM50": "S1 simulated LG M50"}
OUT_TEX = {"R0_f0I": r"f_0 I", "R0_f0I_z": r"f_0 I z_c", "R0_f0I_z2": r"f_0 I z_c^{2}",
           "BV_0.5": r"\eta_{BV}(0.5 Q)", "BV_1": r"\eta_{BV}(Q)", "BV_2": r"\eta_{BV}(2 Q)", "BV_4": r"\eta_{BV}(4 Q)",
           "offset": "1", "theta_shift": r"T_{ref} \theta"}
RATE_TEX = {"a_rate": r"\frac{1}{f_{i}}", "c_absI": r"\frac{|I|}{Q}", "d_z_rate": r"\frac{z_c}{f_{i}}"}
INPUT_TEX = {"b_input": "", "e_z_input": "z_c"}
TH_TEX = {"h0_heat": r"I (V - U_{oc})", "h1_loss": r"(T - T_{amb})", "h2_I2": r"I^{2}", "h3_absI": r"|I|"}
HEAD = ["# Fitted SINDy equations", "",
        "Notation: $z_c$ = SOC - 0.5; $f_k = e^{-E_k \\theta / R_g}$ with $\\theta = 1/T_{ref} - 1/T$; "
        "$\\eta_{BV}(I_0) = \\frac{2 R_g T}{F} \\sinh^{-1}\\left(\\frac{I}{2 I_0 e^{E_0 \\theta / R_g}}\\right)$; "
        "$I$ in A (positive for charge), $V$ in V, $T$ in °C, time in s.", ""]


def num(x):
    return f"{x:.3g}".replace("e-0", "e-").replace("e+0", "e")


def tex_num(x):
    s = f"{abs(x):.3g}"
    if "e" in s:
        m, e = s.split("e")
        s = rf"{m} \times 10^{{{int(e)}}}"
    return s


def sum_terms(pairs):
    # (coefficient, term tex) pairs -> "a x - b y" with signs folded in
    out = []
    for k, (c, t) in enumerate(pairs):
        body = tex_num(c) + (" " + t if t not in ("", "1") else "")
        out.append(("- " if c < 0 else ("" if k == 0 else "+ ")) + body)
    return " ".join(out) if out else "0"


def inclusion(cell):
    # {(equation, term): inclusion} and number of refits, empty if not aggregated
    f = cfg.RESULTS / "sindy_ensemble" / cell / "inclusion.csv"
    if not f.exists():
        return {}, None
    d = pd.read_csv(f)
    return {(r.equation, r.term): r.inclusion for r in d.itertuples()}, int(d.n_fits.iat[0])


def cell_section(cell, variant):
    f = cfg.RESULTS / "sindy" / cell / variant / "sindy_params.json"
    if not f.exists():
        return None
    p = json.loads(f.read_text())
    inc, n = inclusion(cell)
    E = [e / 1e3 for e in p["E_fixed_J_per_mol"]]
    lines = [f"## {LABEL[cell]}", "",
             f"Fixed activation energies (from the single-E ECM): E0 = {E[0]:.1f}, E1 = {E[1]:.1f}, "
             f"E2 = {E[2]:.1f} kJ/mol. Q = {p['Q_Ah']} Ah. Terms kept: "
             f"{sum(len(v) for v in p['selected'].values())} of 23.", ""]

    out = [(c, OUT_TEX[k]) for k, c in p["output"].items() if c != 0]
    lines += ["$$", r"V = U_{oc}(z) + v_{1} + v_{2} " + ("+ " if out and out[0][0] > 0 else "") + sum_terms(out),
              "$$", ""]
    for i, key in ((1, "branch1"), (2, "branch2")):
        b = p[key]
        rate = [(c, RATE_TEX[k].replace("{i}", "{" + str(i) + "}")) for k, c in b.items() if k in RATE_TEX and c != 0]
        inp = [(c, INPUT_TEX[k]) for k, c in b.items() if k in INPUT_TEX and c != 0]
        lines += ["$$", rf"\frac{{d v_{{{i}}}}}{{dt}} = -\left({sum_terms(rate)}\right) v_{{{i}}} + "
                        rf"\left({sum_terms(inp)}\right) I", "$$", ""]
    th = [(-c if k == "h1_loss" else c, TH_TEX[k]) for k, c in p["thermal"].items() if c != 0]
    lines += ["$$", r"\frac{dT}{dt} = " + sum_terms(th), "$$", ""]

    tau = [1 / p["branch1"]["a_rate"], 1 / p["branch2"]["a_rate"]]
    r0 = p["output"]["R0_f0I"]
    r0_txt = (f"{num(r0 * 1e3)} mOhm" if r0 else
              "dropped (the SOC-dependent and Butler-Volmer terms carry the instantaneous loss)")
    lines += [f"Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): "
              f"{num(tau[0])} s and {num(tau[1])} s. Linear series resistance at 25 °C: {r0_txt}. "
              f"Effective heat capacity 1/h0 = {num(1 / p['thermal']['h0_heat'])} J/K.", ""]

    if inc:
        rows = ["| Equation | Term | Coefficient | Kept in refits |", "| --- | --- | --- | --- |"]
        for eq in ("output", "branch1", "branch2", "thermal"):
            for k, c in p[eq].items():
                rows.append(f"| {eq} | {k} | {num(c) if c else 'dropped'} | {inc.get((eq, k), float('nan')):.2f} |")
        rows.append(f"| output | Butler-Volmer group | | {inc.get(('output', 'BV group'), float('nan')):.2f} |")
        lines += [f"Ensemble: {n} bootstrap refits.", ""] + rows + [""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "no_bv"])
    a = ap.parse_args()
    parts = [s for s in (cell_section(c, a.variant) for c in CELLS) if s]
    out = cfg.RESULTS / "sindy" / f"equations_{a.variant}.md"
    out.write_text("\n".join(HEAD + parts))
    print(f"wrote {out} ({len(parts)} cells)")


if __name__ == "__main__":
    main()