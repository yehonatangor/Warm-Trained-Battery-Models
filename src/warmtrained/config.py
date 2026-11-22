"""
Configuration module for the project to hold required constants
Project paths, cell metadata and the fixed numbers that each module shares
"""

from pathlib import Path

# Making the paths portable
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

# Cleaned, resampled data with SOC and features
DATA_PROCESSED = ROOT / "data" / "processed"

RESULTS = ROOT / "results"

# Physical Constants
T_REF_K = 298.15  # The reference temperature for Arrhenius terms [K]
R_GAS = 8.314  # Universal Gas Constant [J/(mol*K)]

# CELLS lookup table to handle the large amount of batteries
CELLS = {
    # B1: Samsung INR21700 30T (McMaster, Digatron CSV)
    "Samsung30T": {
        "label": "B1",
        "root": "samsung30T/Samsung INR21700 30T 3Ah Li-ion Battery Data",
        "folders": {
            40: "40degC",
            25: "25degC",
            10: "10degC",
            0: "0degC",
            -10: "-10degC",
            -20: "-20degC",
        },
        "nominal_ah": 3.0,
        "format": "digatron",
    },
    # B2: LG M50 (Brosa Planella, BaSyTec)
    "LGM50": {
        "label": "B2",
        "root": "brosaplanella_LGM50/data",
        "folders": {25: "25degC", 10: "10degC", 0: "0degC"},
        "nominal_ah": 5.0,
        "format": "basytec",
    },
    # B3: LG HG2 (McMaster, Digatron CSV)
    "LGHG2": {
        "label": "B3",
        "root": "lghg2",
        "folders": {
            40: "40degC",
            25: "25degC",
            10: "10degC",
            0: "0degC",
            -10: "n10degC",
            -20: "n20degC",
        },
        "nominal_ah": 3.0,
        "format": "digatron",
    },
    # B4: TU Berlin, 3x LG M50LT (BaSyTec .txt)
    # Chamber stays at set point, so warmer readings = self-heating (no warm-up rule)
    # Ambient = cell temperature at the end of each log's 2 h rest
    # A CV end counts as full at every temperature
    "TUBerlinM50LT": {
        "label": "B4",
        "root": "tuberlin_m50lt/extracted/0_raw",
        "folders": {
            45: "JY_SOC_45deg",
            35: "JY_SOC_35deg",
            25: "JY_SOC_25deg",
            15: "JY_SOC_15deg",
            5: "JY_SOC_5deg",
        },
        "glob": "{folder}/*.txt",
        "nominal_ah": 4.93,
        "format": "basytec_txt",
        "warmup_rule": False,
        "ambient_mode": "log_rest_end",
        "anchor_min_T_c": -100.0,
        "capacity_glob": "capacity_test/*_capacity_25deg.txt",
    },
    # B5: McMaster, 3x Tesla Model 3 2170 (.mat, 1 Hz)
    # Rows with chamber set point off by > WARMUP_BAND_K are dropped; "soak" rule
    # replaces the warm-up rule. Capacity fades ~10 % hot->cold: per-stage Q_ref
    "Tesla2170": {
        "label": "B5",
        "root": "tesla2170/open_data",
        "folders": {40: "40C", 25: "25C", 10: "10C", 0: "0C", -10: "-10C", -20: "-20C"},
        "glob": "*/{folder}/*/*.mat",
        "nominal_ah": 4.5,
        "format": "tesla_mat",
        "warmup_rule": "soak",
        "grid_align": "round",  # 1 s grid with jittering time stamps
        "q_ref_mode": "per_stage",
        "exclude_files": {
            # full -10 degC log; same rows exist in the split files beside it
            "10-23-21_17.55 Tesla_Char_-10C_V3_Channel_3_Wb_1.mat": "duplicate_of_split_files",
            # owners' readme: this -10 degC C/20 run failed and was re-run
            "8_C_20_Discharge_Charge_10-27-21_11.40 Tesla_Char_-10C_V3_Channel_4_Wb_1.mat": "owner_reported_bad_run",
        },
    },
    # B6: UW-Madison, Panasonic 18650PF (.mat)
    # Trise tests: chamber warms during the drive, so ambient = measured chamber
    # temperature and no warm-up rule. Aged end-of-campaign tests excluded
    "Panasonic18650PF": {
        "label": "B6",
        "root": "panasonic18650pf/Panasonic 18650PF Data",
        "folders": {
            25: "Panasonic 18650PF Data/25degC",
            10: "10degC",
            0: "0degC",
            -10: "-10degC",
            -20: "-20degC",
        },
        "glob": "{folder}/*/*.mat",
        "trise_folders": {
            -20: ["-20degC Trise", "-20degC Trise with pause"],
            10: ["10degC Trise", "10degC Trise with pause"],
        },
        "trise_glob": "{folder}/*.mat",
        "exclude_dirs": {"1C discharge tests_end_of_tests": "end_of_campaign_aged"},
        "exclude_types": {"Combined drive cycles": "duplicate_of_split_files"},
        "nominal_ah": 2.9,
        "format": "panasonic_mat",
        "start_full_under_load": True,  # drive files start under load right after full charge
        "q_ref_glob": "Panasonic 18650PF Data/25degC/C20*/*C20*.mat",
    },
    # B7: McMaster, Turnigy Graphene 5 Ah (Digatron CSV)
    # No drive cycles published at -20 degC
    "Turnigy5Ah": {
        "label": "B7",
        "root": "turnigy",
        "folders": {
            40: "40 degC",
            25: "25 degC",
            10: "10 degC",
            0: "0 degC",
            -10: "-10 degC",
            -20: "-20 degC",
        },
        "nominal_ah": 5.0,
        "format": "digatron",
    },
    # B8: UCCS (Plett), A123 26650 LFP (Arbin .xlsx)
    # Dynamic tests = script 1 only
    # OCV C/30 discharge (S1) + charge (S3) = slow test (reference slow test)
    "A123_26650": {
        "label": "B8",
        "root": "plett_a123_26650",
        "folders": {
            45: "P45",
            35: "P35",
            25: "P25",
            15: "P15",
            5: "P05",
            -5: "N05",
            -15: "N15",
            -25: "N25",
        },
        "globs": [
            "DYNdata/A002_DYN/A002_DYN_*_{folder}_S1.xlsx",
            "OCVdata/A002_OCV_{folder}_S[13].xlsx",
        ],
        "nominal_ah": 2.5,
        "format": "arbin",
        "voltage_range": (1.9, 3.75),
        "anchor_volts": (3.40, 3.40, 3.59),
        "q_ref_glob": "OCVdata/A002_OCV_P25_S1.xlsx",
    },
    # B9: CALCE, 2x A123 18650 LFP (Arbin .xlsx)
    # Each log = 3 tests (charge, three drive tests: DST/US06/FUDS), split by io.split_calce_dynamic
    # Cold CV end counted as full (may be a few % short at -10 degC)
    "CALCE_A123": {
        "label": "B9",
        "root": "calce_a123",
        "folders": {
            50: "50",
            40: "40",
            25: "25",
            20: "20",
            10: "10",
            0: "0",
            -10: "N10",
        },
        "globs": ["dynamic/DST-US06-FUDS-{folder}/*.xlsx", "ocv/OCV{t}-2012*/*.xlsx"],
        "nominal_ah": 1.1,
        "format": "calce_dynamic",
        "voltage_range": (1.9, 3.75),
        "anchor_volts": (3.40, 3.40, 3.59),
        "anchor_min_T_c": -100.0,
        "q_ref_glob": "ocv/OCV25-*/*.xlsx",
    },
}

# Simulated Cells (known-truth check, PyBaMM DFN)
SIM_CELLS = {
    "SimLGM50": {"label": "S1", "nominal_ah": 5.0, "simulated": True},
}

# Every cell a model can be fitted on
MODEL_CELLS = {**CELLS, **SIM_CELLS}

# Temperature Splits (training / validation / testing)
SPLITS = {
    # B1
    "Samsung30T": {"train": [25, 40], "select": [10], "test": [0, -10, -20]},
    # B2: no spare temperature, so validate by leaving one cell out at a time
    "LGM50": {"train": [25, 10], "select": "leave-one-cell-out", "test": [0]},
    # B3
    "LGHG2": {"train": [25, 40], "select": [10], "test": [0, -10, -20]},
    # B4: 35 degC lies between the training temperatures (interpolation test)
    "TUBerlinM50LT": {"train": [25, 45], "select": [15], "test": [35, 5]},
    # B5
    "Tesla2170": {"train": [25, 40], "select": [10], "test": [0, -10, -20]},
    # B6: no 40 degC data, so no validation temperature;
    # trise = warming-chamber tests, an extra locked test split
    "Panasonic18650PF": {
        "train": [25, 10],
        "select": [],
        "test": [0, -10, -20],
        "trise": [-20, 10],
    },
    # B7
    "Turnigy5Ah": {"train": [25, 40], "select": [10], "test": [0, -10, -20]},
    # B8
    "A123_26650": {
        "train": [25, 45],
        "select": [15],
        "test": [5, -5, -15, -25],
        "unused": [35],
    },
    # B9
    "CALCE_A123": {
        "train": [25, 40],
        "select": [10],
        "test": [0, -10],
        "unused": [50, 20],
    },
    # S1 (simulated): fresh = added later, not seen by any of the models, scored once
    "SimLGM50": {
        "train": [25, 40],
        "select": [10],
        "test": [30, 35, 0, -10],
        "fresh": [-20, -5],
    },
}

# Held out Drive Cycles (not used for training at any temperatures)
HELD_OUT_CYCLES = ["US06", "LA92"]  # default (McMaster cells and B4)

HELD_OUT_BY_CELL = {
    "Tesla2170": ["HWCUST", "HWGRADE"],  # Reordered cycles reuse US06/LA92 portions
    "Panasonic18650PF": ["US06", "LA92", "NN"],  # NN cycle built from US06 + LA92
}

# B4 1C charges at a test temperature, kept out of training and never scored
USED_INFO_TYPES = {"TUBerlinM50LT": {"Charge 1C"}}

# Cleaning Thresholds

# Warm-up / Segmenting
WARMUP_BAND_K = 3.0  # cell must be within 3 K of the chamber temperature to count
WARMUP_MIN_SEGMENT_S = 120.0  # a kept stretch after warm-up must last at least 2 min
GAP_SPLIT_S = 600.0  # a logging gap longer than 10 min...
GAP_SPLIT_DV = 0.02  # ...with a voltage jump above 0.02 V splits the data in two
MIN_SEGMENT_S = 10.0  # drop pieces shorter than 10 s after resampling
KEPT_MIN_S = 600.0  # drop a whole file if under 10 min of usable data remains
MIN_LOAD_S = 60.0  # under 1 min of current flow = rest-only test, not a real test

# Charge and SOC
TOP_UP_START_V = 4.0  # B4 charge starting above 4.0 V = top-up, not a full charge

# Allowed SOC (-5 % to 108 %); outside this the file is dropped
SOC_RANGE = (-0.05, 1.08)

REST_CURRENT_A = 0.05  # current below 0.05 A counts as resting

# Sanity limits (outside these limits it is sensor or logging error)
LABEL_BAND_K = 3.0  # file's temperature label must match measured temp within 3 K
VOLTAGE_RANGE_V = (2.4, 4.25)  # believable voltage [V]
TEMPERATURE_RANGE_C = (-30.0, 60.0)  # believable temperature [degC]
MAX_ABS_CURRENT_A = 40.0  # believable current [A]

# Coulomb counting (adding up charge over time)
# Trust time steps up to 1 h; longer gaps are handled separately
MAX_DT_COUNT_S = 3600.0

# Resampling
RESAMPLE_S = 1.0  # convert all data to one point per second

# Dataset specific extras

# B2 (LG M50): ignore the faulty temperature sensor on this cell
LGM50_EXCLUDE_TEMPERATURE_CELLS = [791]

# B4 (TU Berlin): drive cycles each cell ran, in test order (from the dataset readme)
TUBERLIN_CYCLES = {
    "NMC063": ["BCDC", "LA92", "HHDDT", "IM"],
    "NMC068": ["CSHVC", "FTP-72", "FTP-75", "HWFET"],
    "NMC069": ["US06", "PDTCB", "PDMHC", "OCTBC"],
}
