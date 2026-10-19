import numpy as np
from scipy.stats import wilcoxon


def paired_test(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired a - b over runs (e.g. 25 (seed, fold) pairs): mean diff, 95% percentile-bootstrap CI, Wilcoxon p.

    Same procedure as the inline Phase 0 / Phase 1 notebook code. p is nan when every difference is 0.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = np.random.default_rng(seed).choice(d, size=(n_boot, len(d))).mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    return {"diff": d.mean(), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5), "p": p}


if __name__ == "__main__":
    g = np.random.default_rng(0)
    b = g.uniform(0.6, 0.8, 25)
    r = paired_test(b + 0.05 + g.normal(0, 0.005, 25), b)
    assert r["p"] < 0.05 and r["ci_lo"] > 0 and abs(r["diff"] - 0.05) < 0.005
    r = paired_test(b, b)
    assert r["diff"] == 0 and r["ci_lo"] == r["ci_hi"] == 0
    print("stats self-check OK")
