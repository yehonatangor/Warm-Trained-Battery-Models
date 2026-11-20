# Warm-Trained Battery Models and Fast-Charging Limits in Canadian Winters

A battery management system limits fast charging using a model that predicts cell voltage from current temperature, and state of charge, with most models running at room temperature. This project aims to answer how wrong model fitted only at $25 \degree{C}$ and $40 \degree{C}$ acts at $0 \degree{C}$, $-10 \degree{C}$, and $-20 \degree{C}$, and which model holds up the best.

Four models are compared on the same data and the same score:

- ARX: linear autoregressive baseline
- ECM: second-order RC circuit with Arrhenius resistances and a lumped thermal model
- SINDy: sparse regression over a 23-term physics library (SOC, current, Butler-Volmer and heat terms), with a bootstrap ensemble
- LSTM: recurrent network with an Arrhenius-scaled C-rate input

Author: Yehonatan Gor.
Course: CBE 4427.
Institution: Western University.

## Main findings

- Every model's voltage error grows about tenfold from 25 to −20 °C. Accuracy at warm temperatures says nothing about accuracy in the cold.
- The failure is the temperature law, not the model class: the activation energies cannot be identified from a 15 K warm range.
- The ECM beats the LSTM in the cold (30 of 30 files on the Tesla 2170 cell, 5 of 5 temperatures on a pre-registered Panasonic cell).
- SINDy recovers the ECM structure from warm data but its extra terms do not extrapolate.
- One test at 10 °C plus online adaptation of the resistance brings the three main cells to about 50 mV at −20 °C and removes nearly all large-error winter hours in Toronto and London ON.

## Data

Raw data is not in this repository. 

Download each set into `data/raw/<folder>`. 

The folder names are in `src/fastcharge/config.py`.

| Type | Source | Licence |
|---|---|---|
| Samsung INR21700-30T, 3 Ah | McMaster University (Kollmeyer and Skells), Mendeley Data | CC BY 4.0 |
| LG INR18650-HG2, 3 Ah | McMaster University (Kollmeyer et al.), Mendeley Data | CC BY 4.0 |
| Tesla Model 3 2170, 4.5 Ah | McMaster University (Kollmeyer et al.), Borealis | CC BY 4.0 |
| LG INR21700-M50, 5 Ah | Brosa Planella et al., Zenodo 10.5281/zenodo.4864437 | BSD 3-Clause |
| LG INR21700-M50LT, 4.9 Ah | TU Berlin (Yao) | CC BY 4.0 |
| Panasonic 18650PF, 2.9 Ah | University of Wisconsin-Madison (Kollmeyer), Mendeley Data | CC BY 4.0 |
| Turnigy Graphene, 5 Ah | McMaster University (Kollmeyer), Mendeley Data 4fx8cjprxm | CC BY 4.0 |
| A123 26650 m1b, 2.5 Ah | UCCS (de Souza, Plett), Mendeley Data p8kf893yv3 | CC BY 4.0 |
| A123 18650, 1.1 Ah | CALCE, University of Maryland | see source |
| Simulated LG M50 | PyBaMM Doyle-Fuller-Newman model, O'Regan et al. 2022 parameters | open source |
| BMW i3: 72 real trips | TU Munich, IEEE DataPort 10.21227/6jr9-5235 | CC BY 4.0 |
| Weather: Edmonton, Toronto, London ON, hourly, 2015 to 2025 | Environment and Climate Change Canada | Open Government Licence Canada |

