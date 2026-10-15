import glob
import os
import re

import numpy as np
import pandas as pd

FS_HZ = 60e6
FC_HZ = 2.4375e9
# folder name -> drone; folder is the label (files in MP1/MP2 are prefixed MA1/MAV)
DRONES = {"AIR": "DJI Mavic 2 Air S", "DIS": "Parrot Disco", "INS": "DJI Inspire 2", "MIN": "DJI Mavic Mini",
          "MP1": "DJI Mavic Pro", "MP2": "DJI Mavic Pro 2", "PHA": "DJI Phantom 4"}
SUBSETS = {"00": "CLEAN", "01": "BLUE", "10": "WIFI", "11": "BOTH"}  # interference code -> subset folder
MODES = {"00": "ON", "01": "HO", "10": "FY"}                        # mode code -> switched on / hover / fly


def inventory(root: str) -> pd.DataFrame:
    """One row per recording: path, subset, drone, mode, k, n_samples, rec_id (unique across subsets)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*.dat"))):
        subset, folder, name = path.split(os.sep)[-3:]
        drone, mode = folder.split("_")
        m = re.fullmatch(r"\w{3}_([01]{2})([01]{2})_(\d{2})\.dat", name)
        if m is None or SUBSETS.get(m[1]) != subset or MODES.get(m[2]) != mode or drone not in DRONES:
            raise ValueError(f"file name does not match its folders: {path}")
        k = int(m[3])
        rows.append({"path": path, "subset": subset, "drone": drone, "mode": mode, "k": k,
                     "n_samples": os.path.getsize(path) // 8, "rec_id": f"{subset}_{drone}_{mode}_{k}"})
    return pd.DataFrame(rows)


def read(path: str) -> np.memmap:
    """Interleaved float32 I/Q -> complex64 memmap of the recording's actual length (not all are 1.2e8)."""
    return np.memmap(path, dtype=np.complex64, mode="r", shape=(os.path.getsize(path) // 8,))


def select_windows(x, size: int = 4000, cap: int = 1000, floor_pct: float = 10, margin_db: float = 6,
                   block: int = 2500):
    """Non-overlapping windows -> (active_starts, noise_starts, active_frac); starts capped at `cap`, evenly spread.

    Window power = variance (DC removed; DC offset is as large as the signal on some receivers).
    Floor = `floor_pct` percentile of window power; active >= floor + margin_db, noise <= floor.
    `active_frac` is the fraction of all windows that are active, before the cap.
    Reads `block` windows at a time so a memmap is never loaded whole.
    """
    n_win = len(x) // size
    power = np.empty(n_win)
    for b in range(0, n_win, block):
        w = np.asarray(x[b * size:min(b + block, n_win) * size]).reshape(-1, size)
        power[b:b + len(w)] = np.var(w, axis=1)  # complex var = mean|x - mean|^2
    floor = np.percentile(power, floor_pct)

    def spread(cand):
        if len(cand) > cap:
            cand = cand[np.unique(np.linspace(0, len(cand) - 1, cap).round().astype(int))]
        return cand * size

    active = np.flatnonzero(power >= floor * 10 ** (margin_db / 10))
    return spread(active), spread(np.flatnonzero(power <= floor)), len(active) / n_win


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    size, n_win = 4000, 400
    x = (rng.standard_normal(n_win * size) + 1j * rng.standard_normal(n_win * size)).astype(np.complex64)
    burst = rng.random(n_win) < 0.25
    for i in np.flatnonzero(burst):
        x[i * size:(i + 1) * size] *= 10
    x += 5 + 3j  # DC offset

    active, noise, frac = select_windows(x, size=size, cap=1000)
    assert frac == burst.mean()
    assert burst[active // size].all() and not burst[noise // size].any()
    assert len(active) == burst.sum()

    active, noise, _ = select_windows(x, size=size, cap=50)
    span = n_win * size
    for s in (active, noise):
        assert len(s) <= 50 and s.min() < 0.1 * span and s.max() > 0.9 * span

    # DC alone (no bursts) creates no active windows
    quiet = (rng.standard_normal(n_win * size) + 1j * rng.standard_normal(n_win * size)) + (5 + 3j)
    assert len(select_windows(quiet, size=size)[0]) == 0

    # a code that disagrees with its folders is rejected
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "CLEAN", "MIN_ON"))
        open(os.path.join(d, "CLEAN", "MIN_ON", "MIN_1000_00.dat"), "wb").close()
        try:
            inventory(d)
            raise AssertionError("mismatch not rejected")
        except ValueError:
            pass
    print("drone_detect self-check OK")
