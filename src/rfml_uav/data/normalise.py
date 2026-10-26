"""Receiver-noise normalisation (thesis 3.2): centre WPD features on the mean of noise-only windows.

Centring only. Dividing by the noise std was tried and rejected: the noise std reacts to receiver changes
differently from active windows (a 3 dB gain step shrinks it to ~0.56x), so the division inflates features.
"""
import numpy as np


def noise_mean(X):
    """Per-feature mean of noise-only windows."""
    return np.asarray(X, dtype=np.float64).mean(0)


def centre(X, mu):
    return (np.asarray(X, dtype=np.float64) - mu).astype(np.float32)


def session_normalise(X, rec_id, is_noise):
    """Every row centred on the noise mean of its own recording."""
    X, rec_id, is_noise = np.asarray(X), np.asarray(rec_id), np.asarray(is_noise, dtype=bool)
    out = np.empty(X.shape, dtype=np.float32)
    for r in np.unique(rec_id):
        k = rec_id == r
        out[k] = centre(X[k], noise_mean(X[k & is_noise]))
    return out


def stream_normalise(X, is_noise, beta: float = 0.01, k: int = 50, return_state: bool = False,
                     return_drift: bool = False):
    """One time-ordered stream -> (Xn, valid[, mu][, drift]).

    Warm-up: mean of the first k noise windows; windows up to the k-th noise window are not valid.
    Then each window is centred with the state before its own update, and only noise windows update
    the state (EMA, rate beta): no look-ahead, active windows never move it.
    drift (optional, per window): |mu_i - mu_0|_1 / |mu_0|_1 of the state used for window i against the warm-up
    mean mu_0 (noise-floor drift, the receiver gate's input); NaN before warm-up.
    """
    X = np.asarray(X, dtype=np.float64)
    is_noise = np.asarray(is_noise, dtype=bool)
    out, valid = np.zeros(X.shape, dtype=np.float32), np.zeros(len(X), dtype=bool)
    drift = np.full(len(X), np.nan)
    noise_idx = np.flatnonzero(is_noise)
    mu = None
    if len(noise_idx) >= k:
        mu = mu0 = noise_mean(X[noise_idx[:k]])
        norm0 = np.abs(mu0).sum()
        d = 0.0
        for i in range(noise_idx[k - 1] + 1, len(X)):
            out[i] = X[i] - mu
            valid[i] = True
            drift[i] = d
            if is_noise[i]:
                mu = mu + beta * (X[i] - mu)
                if return_drift:
                    d = np.abs(mu - mu0).sum() / norm0
    return (out, valid) + ((mu,) if return_state else ()) + ((drift,) if return_drift else ())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 2000, 8
    X = rng.normal(3, 2, (n, d))
    noise = rng.random(n) < 0.5
    X[~noise] += rng.normal(5, 4, ((~noise).sum(), d))  # active windows: signal on top
    rec = np.where(np.arange(n) < n // 2, "a", "b")

    # active rows never enter the mean
    Y = X.copy()
    Y[~noise] *= 100
    assert np.allclose(session_normalise(X, rec, noise)[noise], session_normalise(Y, rec, noise)[noise])
    # a per-feature additive receiver change applied to a whole session cancels out
    b = rng.normal(0, 5, d)
    assert np.allclose(session_normalise(X + b, rec, noise), session_normalise(X, rec, noise), atol=1e-4)
    # session-normalised noise is centred
    Xn = session_normalise(X, rec, noise)
    for r in ("a", "b"):
        assert np.allclose(Xn[(rec == r) & noise].mean(0), 0, atol=1e-4)

    # stream: active windows do not move the state
    Sn = rng.normal(3, 2, (400, d))
    base, _ = stream_normalise(Sn, np.ones(400, bool), k=50)
    with_active = np.insert(Sn, 200, rng.normal(50, 9, (30, d)), axis=0)
    flags = np.insert(np.ones(400, bool), 200, np.zeros(30, bool))
    out, _ = stream_normalise(with_active, flags, k=50)
    assert np.allclose(out[flags], base)
    # converges to the session mean on a long stationary noise stream (beta 0.001: EMA averages ~2000
    # windows, so the 5% bound is a statistical, not a lucky, margin)
    L = int(20 / 0.001)
    S = rng.normal(3, 2, (L, d))
    _, _, mu = stream_normalise(S, np.ones(L, bool), beta=0.001, k=50, return_state=True)
    assert np.all(np.abs(mu - noise_mean(S)) <= 0.05 * np.abs(noise_mean(S)))
    # no look-ahead
    S2 = S.copy()
    S2[-1] += 1000
    assert np.array_equal(stream_normalise(S, np.ones(L, bool))[0][:-1], stream_normalise(S2, np.ones(L, bool))[0][:-1])
    # warm-up: invalid until the k-th noise window has arrived
    _, valid = stream_normalise(X, noise, k=50)
    kth = np.flatnonzero(noise)[49]
    assert not valid[:kth + 1].any() and valid[kth + 1:].all()
    # drift: ~0 on stationary noise; -> |a - 1| when the floor is scaled by a; outputs unchanged
    S3 = rng.normal(3, 0.5, (6000, d))
    _, _, dr = stream_normalise(S3, np.ones(6000, bool), k=50, return_drift=True)
    assert np.nanmax(dr) < 0.05, np.nanmax(dr)
    S3[3000:] *= 1.5
    o1, v1, dr = stream_normalise(S3, np.ones(6000, bool), k=50, return_drift=True)
    assert abs(dr[-1] - 0.5) < 0.02 and dr[2999] < 0.05, (dr[2999], dr[-1])
    o2, v2 = stream_normalise(S3, np.ones(6000, bool), k=50)
    assert np.array_equal(o1, o2) and np.array_equal(v1, v2) and np.isnan(dr[:50]).all()
    print("normalise self-check OK")
