# Fitted SINDy equations

Notation: $z_c$ = SOC - 0.5; $f_k = e^{-E_k \theta / R_g}$ with $\theta = 1/T_{ref} - 1/T$; $\eta_{BV}(I_0) = \frac{2 R_g T}{F} \sinh^{-1}\left(\frac{I}{2 I_0 e^{E_0 \theta / R_g}}\right)$; $I$ in A (positive for charge), $V$ in V, $T$ in °C, time in s.

## B1 Samsung 30T

Fixed activation energies (from the single-E ECM): E0 = 13.6, E1 = 114.1, E2 = 0.0 kJ/mol. Q = 3.0 Ah. Terms kept: 14 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.00946 f_0 I + 0.00489 f_0 I z_c + 0.0092 - 0.0513 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.217 \frac{1}{f_{1}} + 0.439 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000885\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.000249 \frac{1}{f_{2}} + 0.00061 \frac{|I|}{Q}\right) v_{2} + \left(1.51 \times 10^{-5}\right) I
$$

$$
\frac{dT}{dt} = 2.3 \times 10^{-5} I (V - U_{oc}) - 0.00255 (T - T_{amb}) + 4.08 \times 10^{-5} I^{2} + 0.000417 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 4.6 s and 4.02e3 s. Linear series resistance at 25 °C: 9.46 mOhm. Effective heat capacity 1/h0 = 4.35e4 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.00946 | 0.77 |
| output | R0_f0I_z | 0.00489 | 0.92 |
| output | R0_f0I_z2 | dropped | 0.83 |
| output | BV_0.5 | dropped | 0.00 |
| output | BV_1 | dropped | 0.00 |
| output | BV_2 | dropped | 0.01 |
| output | BV_4 | dropped | 0.39 |
| output | offset | 0.0092 | 0.99 |
| output | theta_shift | -0.0513 | 0.94 |
| branch1 | a_rate | 0.217 | 1.00 |
| branch1 | c_absI | dropped | 0.13 |
| branch1 | d_z_rate | 0.439 | 0.98 |
| branch1 | b_input | 0.000885 | 1.00 |
| branch1 | e_z_input | dropped | 0.48 |
| branch2 | a_rate | 0.000249 | 1.00 |
| branch2 | c_absI | 0.00061 | 0.98 |
| branch2 | d_z_rate | dropped | 0.06 |
| branch2 | b_input | 1.51e-5 | 1.00 |
| branch2 | e_z_input | dropped | 0.53 |
| thermal | h0_heat | 2.3e-5 | 1.00 |
| thermal | h1_loss | 0.00255 | 1.00 |
| thermal | h2_I2 | 4.08e-5 | 0.50 |
| thermal | h3_absI | 0.000417 | 0.73 |
| output | Butler-Volmer group | | 0.39 |

## B3 LG HG2

Fixed activation energies (from the single-E ECM): E0 = 5.6, E1 = 56.6, E2 = 0.0 kJ/mol. Q = 3.0 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0171 f_0 I - 0.00327 f_0 I z_c + 0.0623 f_0 I z_c^{2} + 0.0179 - 0.0545 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.00463 \frac{1}{f_{1}} + 0.028 \frac{|I|}{Q} + 0.0103 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000806 - 0.000283 z_c\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00847 \frac{1}{f_{2}} - 0.00822 \frac{|I|}{Q}\right) v_{2} + \left(5.22 \times 10^{-6}\right) I
$$

$$
\frac{dT}{dt} = 0.00411 I (V - U_{oc}) - 0.0024 (T - T_{amb}) + 0.000146 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 216 s and 118 s. Linear series resistance at 25 °C: 17.1 mOhm. Effective heat capacity 1/h0 = 243 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0171 | 0.99 |
| output | R0_f0I_z | -0.00327 | 0.72 |
| output | R0_f0I_z2 | 0.0623 | 1.00 |
| output | BV_0.5 | dropped | 0.24 |
| output | BV_1 | dropped | 0.10 |
| output | BV_2 | dropped | 0.07 |
| output | BV_4 | dropped | 0.02 |
| output | offset | 0.0179 | 1.00 |
| output | theta_shift | -0.0545 | 0.68 |
| branch1 | a_rate | 0.00463 | 1.00 |
| branch1 | c_absI | 0.028 | 1.00 |
| branch1 | d_z_rate | 0.0103 | 1.00 |
| branch1 | b_input | 0.000806 | 1.00 |
| branch1 | e_z_input | -0.000283 | 0.89 |
| branch2 | a_rate | 0.00847 | 1.00 |
| branch2 | c_absI | -0.00822 | 0.99 |
| branch2 | d_z_rate | dropped | 0.71 |
| branch2 | b_input | 5.22e-6 | 1.00 |
| branch2 | e_z_input | dropped | 0.48 |
| thermal | h0_heat | 0.00411 | 1.00 |
| thermal | h1_loss | 0.0024 | 1.00 |
| thermal | h2_I2 | dropped | 0.43 |
| thermal | h3_absI | 0.000146 | 0.78 |
| output | Butler-Volmer group | | 0.40 |

## B5 Tesla 2170

Fixed activation energies (from the single-E ECM): E0 = 15.9, E1 = 26.1, E2 = 0.0 kJ/mol. Q = 4.5 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0212 f_0 I + 0.00303 f_0 I z_c + 0.0893 f_0 I z_c^{2} - 0.011 + 0.38 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0112 \frac{1}{f_{1}} + 0.0202 \frac{|I|}{Q} + 0.0233 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000519\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.000361 \frac{1}{f_{2}} - 0.0605 \frac{|I|}{Q}\right) v_{2} + \left(2.46 \times 10^{-6}\right) I
$$

$$
\frac{dT}{dt} = 0.00325 I (V - U_{oc}) - 0.0017 (T - T_{amb}) + 0.000104 I^{2} + 0.000332 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 89.4 s and 2.77e3 s. Linear series resistance at 25 °C: 21.2 mOhm. Effective heat capacity 1/h0 = 308 J/K.

Ensemble: 30 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0212 | 0.57 |
| output | R0_f0I_z | 0.00303 | 0.63 |
| output | R0_f0I_z2 | 0.0893 | 0.90 |
| output | BV_0.5 | dropped | 0.03 |
| output | BV_1 | dropped | 0.03 |
| output | BV_2 | dropped | 0.10 |
| output | BV_4 | dropped | 0.27 |
| output | offset | -0.011 | 1.00 |
| output | theta_shift | 0.38 | 0.83 |
| branch1 | a_rate | 0.0112 | 1.00 |
| branch1 | c_absI | 0.0202 | 0.90 |
| branch1 | d_z_rate | 0.0233 | 1.00 |
| branch1 | b_input | 0.000519 | 1.00 |
| branch1 | e_z_input | dropped | 0.70 |
| branch2 | a_rate | 0.000361 | 1.00 |
| branch2 | c_absI | -0.0605 | 1.00 |
| branch2 | d_z_rate | dropped | 0.07 |
| branch2 | b_input | 2.46e-6 | 1.00 |
| branch2 | e_z_input | dropped | 0.10 |
| thermal | h0_heat | 0.00325 | 1.00 |
| thermal | h1_loss | 0.0017 | 1.00 |
| thermal | h2_I2 | 0.000104 | 0.57 |
| thermal | h3_absI | 0.000332 | 0.77 |
| output | Butler-Volmer group | | 0.43 |

## B4 LG M50LT

Fixed activation energies (from the single-E ECM): E0 = 6.4, E1 = 9.5, E2 = 7.6 kJ/mol. Q = 4.93 Ah. Terms kept: 15 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.0311 f_0 I + 0.0337 f_0 I z_c^{2} + 0.00748 + 0.0669 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0332 \frac{1}{f_{1}} - 0.0123 \frac{|I|}{Q}\right) v_{1} + \left(0.000309\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00205 \frac{1}{f_{2}} - 0.0276 \frac{|I|}{Q} + 0.00687 \frac{z_c}{f_{2}}\right) v_{2} + \left(2.77 \times 10^{-6}\right) I
$$

$$
\frac{dT}{dt} = 0.00621 I (V - U_{oc}) - 0.0012 (T - T_{amb}) + 5.76 \times 10^{-5} I^{2} - 0.000368 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 30.2 s and 489 s. Linear series resistance at 25 °C: 31.1 mOhm. Effective heat capacity 1/h0 = 161 J/K.

Ensemble: 100 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.0311 | 1.00 |
| output | R0_f0I_z | dropped | 0.21 |
| output | R0_f0I_z2 | 0.0337 | 0.87 |
| output | BV_0.5 | dropped | 0.61 |
| output | BV_1 | dropped | 0.00 |
| output | BV_2 | dropped | 0.00 |
| output | BV_4 | dropped | 0.00 |
| output | offset | 0.00748 | 1.00 |
| output | theta_shift | 0.0669 | 1.00 |
| branch1 | a_rate | 0.0332 | 1.00 |
| branch1 | c_absI | -0.0123 | 0.60 |
| branch1 | d_z_rate | dropped | 0.62 |
| branch1 | b_input | 0.000309 | 1.00 |
| branch1 | e_z_input | dropped | 0.60 |
| branch2 | a_rate | 0.00205 | 1.00 |
| branch2 | c_absI | -0.0276 | 0.99 |
| branch2 | d_z_rate | 0.00687 | 0.98 |
| branch2 | b_input | 2.77e-6 | 1.00 |
| branch2 | e_z_input | dropped | 0.07 |
| thermal | h0_heat | 0.00621 | 1.00 |
| thermal | h1_loss | 0.0012 | 1.00 |
| thermal | h2_I2 | 5.76e-5 | 0.74 |
| thermal | h3_absI | -0.000368 | 1.00 |
| output | Butler-Volmer group | | 0.61 |

## B2 LG M50

Fixed activation energies (from the single-E ECM): E0 = 52.3, E1 = 0.0, E2 = 99.5 kJ/mol. Q = 5.0 Ah. Terms kept: 16 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.00113 f_0 I + 0.0234 f_0 I z_c - 0.0211 f_0 I z_c^{2} + 0.0214 - 0.296 T_{ref} \theta
$$

$$
\frac{d v_{1}}{dt} = -\left(2.15 \frac{1}{f_{1}} + 0.565 \frac{|I|}{Q} + 4.31 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.122 + 0.144 z_c\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.132 \frac{1}{f_{2}} + 0.278 \frac{z_c}{f_{2}}\right) v_{2} + \left(0.000434\right) I
$$

$$
\frac{dT}{dt} = 0.0166 I (V - U_{oc}) - 0.00154 (T - T_{amb}) - 0.00135 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 0.465 s and 7.58 s. Linear series resistance at 25 °C: 1.13 mOhm. Effective heat capacity 1/h0 = 60.1 J/K.

Ensemble: 30 bootstrap refits.

| Equation | Term | Coefficient | Kept in refits |
| --- | --- | --- | --- |
| output | R0_f0I | 0.00113 | 0.00 |
| output | R0_f0I_z | 0.0234 | 0.97 |
| output | R0_f0I_z2 | -0.0211 | 0.90 |
| output | BV_0.5 | dropped | 0.97 |
| output | BV_1 | dropped | 0.03 |
| output | BV_2 | dropped | 0.00 |
| output | BV_4 | dropped | 0.00 |
| output | offset | 0.0214 | 1.00 |
| output | theta_shift | -0.296 | 1.00 |
| branch1 | a_rate | 2.15 | 1.00 |
| branch1 | c_absI | 0.565 | 1.00 |
| branch1 | d_z_rate | 4.31 | 1.00 |
| branch1 | b_input | 0.122 | 1.00 |
| branch1 | e_z_input | 0.144 | 1.00 |
| branch2 | a_rate | 0.132 | 1.00 |
| branch2 | c_absI | dropped | 0.00 |
| branch2 | d_z_rate | 0.278 | 1.00 |
| branch2 | b_input | 0.000434 | 1.00 |
| branch2 | e_z_input | dropped | 0.03 |
| thermal | h0_heat | 0.0166 | 1.00 |
| thermal | h1_loss | 0.00154 | 1.00 |
| thermal | h2_I2 | dropped | 0.03 |
| thermal | h3_absI | -0.00135 | 1.00 |
| output | Butler-Volmer group | | 1.00 |

## S1 simulated LG M50

Fixed activation energies (from the single-E ECM): E0 = 20.1, E1 = 21.4, E2 = 16.4 kJ/mol. Q = 5.0 Ah. Terms kept: 15 of 23.

$$
V = U_{oc}(z) + v_{1} + v_{2} + 0.014 f_0 I - 0.0163 f_0 I z_c + 0.0251 f_0 I z_c^{2} + 0.00268
$$

$$
\frac{d v_{1}}{dt} = -\left(0.0875 \frac{1}{f_{1}} + 0.177 \frac{z_c}{f_{1}}\right) v_{1} + \left(0.000832\right) I
$$

$$
\frac{d v_{2}}{dt} = -\left(0.00416 \frac{1}{f_{2}} - 0.00261 \frac{|I|}{Q} + 0.00803 \frac{z_c}{f_{2}}\right) v_{2} + \left(6.64 \times 10^{-5} + 0.000235 z_c\right) I
$$

$$
\frac{dT}{dt} = 0.0139 I (V - U_{oc}) - 0.00227 (T - T_{amb}) - 0.00037 |I|
$$

Time constants at 25 °C, SOC 0.5 and zero current (where the SOC and current terms vanish): 11.4 s and 240 s. Linear series resistance at 25 °C: 14 mOhm. Effective heat capacity 1/h0 = 71.8 J/K.
