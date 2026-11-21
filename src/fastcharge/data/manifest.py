"""
Split assignment for the manifest
"""

from fastcharge import config as cfg

MIXED_TYPES = {"Mixed cycle"}  # mixed cycles reuse parts of the held-out profiles, so they are excluded


def held_out(cell: str, test_type: str) -> bool:
    # True for the cell's held-out drive cycle family (prefix match: HWCUST covers HWCUST1)
    family = cfg.HELD_OUT_BY_CELL.get(cell, cfg.HELD_OUT_CYCLES)
    return any(test_type.startswith(name) for name in family)


def assign_split(cell: str, t_amb: float, test_type: str, flags: list) -> str:
    # Returns one of:
    # train, select, test
    # holdout_profile: held-out cycle family at a training or selection temperature
    # used_info: TU Berlin charges at a test temperature, never fitted or scored
    # excluded
    if flags:
        return "excluded"

    s = cfg.SPLITS[cell]

    # LG M50 validates by leaving one cell out, so it has no select temperature
    if cell == "LGM50":
        if t_amb in s["train"]:
            return "train"
        if t_amb in s["test"]:
            return "test"
        return "excluded"

    if test_type in MIXED_TYPES:
        return "excluded"

    if t_amb in s["test"]:
        return "used_info" if test_type in cfg.USED_INFO_TYPES.get(cell, set()) else "test"

    if held_out(cell, test_type):
        return "holdout_profile"
    if t_amb in s["train"]:
        return "train"
    if t_amb in s["select"]:
        return "select"
    return "excluded"