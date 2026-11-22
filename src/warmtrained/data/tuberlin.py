"""
Cutting a TU Berlin LG M50LT log (B4) into its tests
Each log holds four blocks per cell and temperature (dataset readme, test plan):
1C CC charge to 4.2 V, CV hold to C/50, 2 h rest, one drive cycle repeated down to 2.5 V
One piece per charge: CC + CV + the 2 h rest after it, so the CV end lies inside
the piece and the relaxation stays with its charge
One piece per drive cycle, starting at the end of that rest: a full, relaxed state
(row 0 SOC anchor in the soc module)
The 1 min rest at the start of the log is not used
"""

from warmtrained.config import TUBERLIN_CYCLES

# Helpers

def runs(cmd):
    # Yields (command, first row, end row) for every run of the same command
    # End row is exclusive, so log.iloc[first:end] is the whole run
    run_id = (cmd != cmd.shift()).cumsum()
    for _, group in cmd.groupby(run_id, sort=True):
        yield group.iloc[0], group.index[0], group.index[-1] + 1


# Splitting

def split_log(log, cell_id):
    # Returns [(name, test type, piece)] in time order, each piece's time restarts at 0
    cycles = TUBERLIN_CYCLES[cell_id]
    blocks = list(runs(log["cmd"]))

    # (label, test type, first row, end row) for every piece
    bounds = []
    n_drive = 0

    for k, (cmd, start, end) in enumerate(blocks):
        prev_is_pause = k > 0 and blocks[k - 1][0] == "Pause"
        next_is_pause = k + 1 < len(blocks) and blocks[k + 1][0] == "Pause"

        if cmd == "Charge":
            # CC and CV share the command, so they form one run
            # Keep the rest after it, except its last row, which starts the next drive piece
            # (no row ends up in two pieces)
            if next_is_pause:
                end = blocks[k + 1][2] - 1
            bounds.append(("charge", "Charge 1C", start, end))

        elif cmd == "Table":
            # Drive cycles run in the order given by the dataset readme
            name = cycles[n_drive]
            n_drive += 1

            # Start on the last rest row: relaxed, zero current
            if prev_is_pause:
                start -= 1
            bounds.append((name, name, start, end))

    if n_drive != len(cycles):
        raise ValueError(f"{cell_id}: found {n_drive} drive cycles, expected {len(cycles)}")

    # Cut the pieces and restart each one's time at zero
    pieces = []
    for i, (label, kind, start, end) in enumerate(bounds, start=1):
        piece = log.iloc[start:end].reset_index(drop=True).copy()
        piece["t"] -= piece["t"].iloc[0]
        pieces.append((f"{i:02d}_{label}", kind, piece))

    return pieces