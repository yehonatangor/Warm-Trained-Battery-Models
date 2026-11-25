# Fitted SINDy equations

Notation: $z_c$ = SOC - 0.5; $f_k = e^{-E_k \theta / R_g}$ with $\theta = 1/T_{ref} - 1/T$; $\eta_{BV}(I_0) = \frac{2 R_g T}{F} \sinh^{-1}\left(\frac{I}{2 I_0 e^{E_0 \theta / R_g}}\right)$; $I$ in A (positive for charge), $V$ in V, $T$ in °C, time in s.

## B1 Samsung 30T

Fixed activation energies (from the single-E ECM): E0 = 13.6, E1 = 114.1, E2 = 0.0 kJ/mol. Q = 3.0 Ah. Terms kept: 15 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0106 f_0 I + 0.00576 f_0 I z_c + 0.00828 f_0 I z_c^{2} + 0.0101 + 0.113 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0711 \frac{1}{f_{1}} + 0.145 \frac{z_c}{f_{1}}\right) v_{1} + \left(9.68 \times 10^{-5} - 0.000625 z_c\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.000222 \frac{1}{f_{2}} + 0.000767 \frac{|I|}{Q}\right) v_{2} + \left(1.73 \times 10^{-5}\right) I
$$

$$
\frac{dT}{dt} = 0.00944 I (V - U_{oc}) - 0.0028 (T - T_{amb}) + 1.86 \times 10^{-5} I^{2}
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 14.1 s and 4.51e3 s. Linear series resistance at 25 °C: 10.6 mOhm. Effective heat capacity 1/h0 = 106 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0106 | 0.77 |
| output | R0_f0I_z | 0.00576 | 0.92 |
| output | R0_f0I_z2 | 0.00828 | 0.83 |
| output | BV_0.5 | dropped | 0.00 |
| output | BV_1 | dropped | 0.00 |
| output | BV_2 | dropped | 0.01 |
| output | BV_4 | dropped | 0.39 |
| output | offset | 0.0101 | 0.99 |
| output | theta_shift | 0.113 | 0.94 |
| branch1 | a_rate | 0.0711 | 1.00 |
| branch1 | c_absI | dropped | 0.13 |
| branch1 | d_z_rate | 0.145 | 0.98 |
| branch1 | b_input | 9.68e-5 | 1.00 |
| branch1 | e_z_input | -0.000625 | 0.48 |
| branch2 | a_rate | 0.000222 | 1.00 |
| branch2 | c_absI | 0.000767 | 0.98 |
| branch2 | d_z_rate | dropped | 0.06 |
| branch2 | b_input | 1.73e-5 | 1.00 |
| branch2 | e_z_input | dropped | 0.53 |
| thermal | h0_heat | 0.00944 | 1.00 |
| thermal | h1_loss | 0.0028 | 1.00 |
| thermal | h2_I2 | 1.86e-5 | 0.50 |
| thermal | h3_absI | dropped | 0.73 |
| output | Butler-Volmer group | | 0.39 |

## B3 LG HG2

Fixed activation energies (from the single-E ECM): E0 = 5.6, E1 = 56.6, E2 = 0.0 kJ/mol. Q = 3.0 Ah. Terms kept: 18 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0171 f_0 I - 0.00314 f_0 I z_c + 0.0627 f_0 I z_c^{2} + 0.0175 - 0.0474 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.00464 \frac{1}{f_{1}} + 0.0279 \frac{|I|}{Q} + 0.0103 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000792 - 0.000348 z_c\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00875 \frac{1}{f_{2}} - 0.00841 \frac{|I|}{Q} - 0.00312 \frac{z_c}{f_{2}}\right) v_{2} + \left(6.23 \times 10^{-6} - 4.45 \times 10^{-6} z_c\right) I
$$

$$
\frac{dT}{dt} = 0.00419 I (V - U_{oc}) - 0.00235 (T - T_{amb}) + 0.000127 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 215 s and 114 s. Linear series resistance at 25 °C: 17.1 mOhm. Effective heat capacity 1/h0 = 239 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0171 | 0.99 |
| output | R0_f0I_z | -0.00314 | 0.72 |
| output | R0_f0I_z2 | 0.0627 | 1.00 |
| output | BV_0.5 | dropped | 0.24 |
| output | BV_1 | dropped | 0.10 |
| output | BV_2 | dropped | 0.07 |
| output | BV_4 | dropped | 0.02 |
| output | offset | 0.0175 | 1.00 |
| output | theta_shift | -0.0474 | 0.68 |
| branch1 | a_rate | 0.00464 | 1.00 |
| branch1 | c_absI | 0.0279 | 1.00 |
| branch1 | d_z_rate | 0.0103 | 1.00 |
| branch1 | b_input | 0.000792 | 1.00 |
| branch1 | e_z_input | -0.000348 | 0.89 |
| branch2 | a_rate | 0.00875 | 1.00 |
| branch2 | c_absI | -0.00841 | 0.99 |
| branch2 | d_z_rate | -0.00312 | 0.71 |
| branch2 | b_input | 6.23e-6 | 1.00 |
| branch2 | e_z_input | -4.45e-6 | 0.48 |
| thermal | h0_heat | 0.00419 | 1.00 |
| thermal | h1_loss | 0.00235 | 1.00 |
| thermal | h2_I2 | dropped | 0.43 |
| thermal | h3_absI | 0.000127 | 0.78 |
| output | Butler-Volmer group | | 0.40 |

## B5 Tesla 2170

Fixed activation energies (from the single-E ECM): E0 = 15.9, E1 = 26.1, E2 = 0.0 kJ/mol. Q = 4.5 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.00302 f_0 I z_c + 0.0895 f_0 I z_c^{2} + 14.6 \eta_{BV}(4 Q) - 0.0111 + 0.394 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0112 \frac{1}{f_{1}} + 0.0201 \frac{|I|}{Q} + 0.0232 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000516\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.000357 \frac{1}{f_{2}} - 0.0599 \frac{|I|}{Q}\right) v_{2} + \left(2.47 \times 10^{-6}\right) I
$$

$$
\frac{dT}{dt} = 0.003 I (V - U_{oc}) - 0.00168 (T - T_{amb}) + 0.000119 I^{2} + 0.000304 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 89.6 s and 2.8e3 s. Linear series resistance at 25 °C: dropped (the SOC-dependent and Butler-Volmer terms carry the instantaneous loss). Effective heat capacity 1/h0 = 334 J/K.

Ensemble: 30 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | dropped | 0.57 |
| output | R0_f0I_z | 0.00302 | 0.63 |
| output | R0_f0I_z2 | 0.0895 | 0.90 |
| output | BV_0.5 | dropped | 0.03 |
| output | BV_1 | dropped | 0.03 |
| output | BV_2 | dropped | 0.10 |
| output | BV_4 | 14.6 | 0.27 |
| output | offset | -0.0111 | 1.00 |
| output | theta_shift | 0.394 | 0.83 |
| branch1 | a_rate | 0.0112 | 1.00 |
| branch1 | c_absI | 0.0201 | 0.90 |
| branch1 | d_z_rate | 0.0232 | 1.00 |
| branch1 | b_input | 0.000516 | 1.00 |
| branch1 | e_z_input | dropped | 0.70 |
| branch2 | a_rate | 0.000357 | 1.00 |
| branch2 | c_absI | -0.0599 | 1.00 |
| branch2 | d_z_rate | dropped | 0.07 |
| branch2 | b_input | 2.47e-6 | 1.00 |
| branch2 | e_z_input | dropped | 0.10 |
| thermal | h0_heat | 0.003 | 1.00 |
| thermal | h1_loss | 0.00168 | 1.00 |
| thermal | h2_I2 | 0.000119 | 0.57 |
| thermal | h3_absI | 0.000304 | 0.77 |
| output | Butler-Volmer group | | 0.43 |

## B4 LG M50LT

Fixed activation energies (from the single-E ECM): E0 = 6.4, E1 = 9.5, E2 = 7.6 kJ/mol. Q = 4.93 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0256 f_0 I + 0.034 f_0 I z_c^{2} + 0.558 \eta_{BV}(0.5 Q) + 0.00739 + 0.0683 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0353 \frac{1}{f_{1}} - 0.0149 \frac{|I|}{Q}\right) v_{1} + \left(0.000309\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00185 \frac{1}{f_{2}} - 0.024 \frac{|I|}{Q} + 0.00612 \frac{z_c}{f_{2}}\right) v_{2} + \left(2.76 \times 10^{-6}\right) I
$$

$$
\frac{dT}{dt} = 0.00622 I (V - U_{oc}) - 0.0012 (T - T_{amb}) + 5.73 \times 10^{-5} I^{2} - 0.000368 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 28.3 s and 541 s. Linear series resistance at 25 °C: 25.6 mOhm. Effective heat capacity 1/h0 = 161 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0256 | 1.00 |
| output | R0_f0I_z | dropped | 0.21 |
| output | R0_f0I_z2 | 0.034 | 0.87 |
| output | BV_0.5 | 0.558 | 0.61 |
| output | BV_1 | dropped | 0.00 |
| output | BV_2 | dropped | 0.00 |
| output | BV_4 | dropped | 0.00 |
| output | offset | 0.00739 | 1.00 |
| output | theta_shift | 0.0683 | 1.00 |
| branch1 | a_rate | 0.0353 | 1.00 |
| branch1 | c_absI | -0.0149 | 0.60 |
| branch1 | d_z_rate | dropped | 0.62 |
| branch1 | b_input | 0.000309 | 1.00 |
| branch1 | e_z_input | dropped | 0.60 |
| branch2 | a_rate | 0.00185 | 1.00 |
| branch2 | c_absI | -0.024 | 0.99 |
| branch2 | d_z_rate | 0.00612 | 0.98 |
| branch2 | b_input | 2.76e-6 | 1.00 |
| branch2 | e_z_input | dropped | 0.07 |
| thermal | h0_heat | 0.00622 | 1.00 |
| thermal | h1_loss | 0.0012 | 1.00 |
| thermal | h2_I2 | 5.73e-5 | 0.74 |
| thermal | h3_absI | -0.000368 | 1.00 |
| output | Butler-Volmer group | | 0.61 |

## B2 LG M50

Fixed activation energies (from the single-E ECM): E0 = 52.3, E1 = 0.0, E2 = 99.5 kJ/mol. Q = 5.0 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.023 f_0 I z_c - 0.0232 f_0 I z_c^{2} + 0.327 \eta_{BV}(0.5 Q) + 0.0211 - 0.295 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(2.04 \frac{1}{f_{1}} + 0.508 \frac{|I|}{Q} + 4.09 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.11 + 0.13 z_c\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.135 \frac{1}{f_{2}} + 0.285 \frac{z_c}{f_{2}}\right) v_{2} + \left(0.000433\right) I
$$

$$
\frac{dT}{dt} = 0.0165 I (V - U_{oc}) - 0.00152 (T - T_{amb}) - 0.00132 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 0.49 s and 7.39 s. Linear series resistance at 25 °C: dropped (the SOC-dependent and Butler-Volmer terms carry the instantaneous loss). Effective heat capacity 1/h0 = 60.7 J/K.

Ensemble: 30 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | dropped | 0.00 |
| output | R0_f0I_z | 0.023 | 0.97 |
| output | R0_f0I_z2 | -0.0232 | 0.90 |
| output | BV_0.5 | 0.327 | 0.97 |
| output | BV_1 | dropped | 0.03 |
| output | BV_2 | dropped | 0.00 |
| output | BV_4 | dropped | 0.00 |
| output | offset | 0.0211 | 1.00 |
| output | theta_shift | -0.295 | 1.00 |
| branch1 | a_rate | 2.04 | 1.00 |
| branch1 | c_absI | 0.508 | 1.00 |
| branch1 | d_z_rate | 4.09 | 1.00 |
| branch1 | b_input | 0.11 | 1.00 |
| branch1 | e_z_input | 0.13 | 1.00 |
| branch2 | a_rate | 0.135 | 1.00 |
| branch2 | c_absI | dropped | 0.00 |
| branch2 | d_z_rate | 0.285 | 1.00 |
| branch2 | b_input | 0.000433 | 1.00 |
| branch2 | e_z_input | dropped | 0.03 |
| thermal | h0_heat | 0.0165 | 1.00 |
| thermal | h1_loss | 0.00152 | 1.00 |
| thermal | h2_I2 | dropped | 0.03 |
| thermal | h3_absI | -0.00132 | 1.00 |
| output | Butler-Volmer group | | 1.00 |

## S1 simulated LG M50

Fixed activation energies (from the single-E ECM): E0 = 20.1, E1 = 21.4, E2 = 16.4 kJ/mol. Q = 5.0 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} - 0.0163 f_0 I z_c + 0.0255 f_0 I z_c^{2} + 1.5 \eta_{BV}(0.5 Q) + 0.00193 + 0.0646 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.088 \frac{1}{f_{1}} + 0.178 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000815\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00383 \frac{1}{f_{2}} - 0.00244 \frac{|I|}{Q} + 0.00739 \frac{z_c}{f_{2}}\right) v_{2} + \left(6.06 \times 10^{-5} + 0.000216 z_c\right) I
$$

$$
\frac{dT}{dt} = 0.0146 I (V - U_{oc}) - 0.00236 (T - T_{amb}) - 0.0004 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 11.4 s and 261 s. Linear series resistance at 25 °C: dropped (the SOC-dependent and Butler-Volmer terms carry the instantaneous loss). Effective heat capacity 1/h0 = 68.5 J/K.
