import time
import numpy as np
from rfml_uav.signal_detection.methods import cyclo_detector, energy_detector


def stage1_score(x):
    return energy_detector(x)

def aggregate_bursts(scores, min_len=3):
    """
    scores: list of booleans (stage-1 decisions)
    returns list of (start_idx, end_idx)
    """
    bursts = []
    start = None

    for i, s in enumerate(scores):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_len:
                bursts.append((start, i))
            start = None

    if start is not None and len(scores) - start >= min_len:
        bursts.append((start, len(scores)))

    return bursts

def cyclo_on_burst(chunks, fs):
    x = np.concatenate(chunks)
    return cyclo_detector(x, fs=fs)


def cascade_detector(
    chunks,
    fs,
    thr,
    min_burst_len=3,
    return_meta=False,   # <-- NEW
):
    timings = {}

    # -------- Stage 1: cheap detector --------
    t0 = time.perf_counter()
    s1_scores = [stage1_score(x) for x in chunks]
    decisions = [s >= thr for s in s1_scores]
    timings["stage1"] = time.perf_counter() - t0

    # -------- Stage 2: burst aggregation --------
    t0 = time.perf_counter()
    bursts = aggregate_bursts(decisions, min_len=min_burst_len)
    timings["aggregation"] = time.perf_counter() - t0

    # -------- Stage 3: cyclostationary --------
    t0 = time.perf_counter()
    cyclo_scores = []
    for b0, b1 in bursts:
        cyclo_scores.append(cyclo_on_burst(chunks[b0:b1], fs))
    timings["cyclo"] = time.perf_counter() - t0

    final_score = max(cyclo_scores) if cyclo_scores else 0.0

    if return_meta:
        meta = {
            "cyclo_invoked": len(bursts) > 0,
            "n_bursts": len(bursts),
            "bursts": bursts,
        }
        return final_score, timings, meta

    return final_score, timings
