import numpy as np

# kind -> parameters per severity; index 0 = identity, 1-3 = E2 levels, 4 = escalation only
SEVERITY = {
    "awgn": [None, 20, 10, 5, 0],                                # SNR vs window power, dB
    "gain": [None, 3, 6, 12, 20],                                # offset and in-window step, dB
    "tilt": [None, 3, 6, 12, 20],                                # edge-to-edge spectral slope, dB
    "adc": [None, (10, 4), (8, 3), (6, 2), (4, 1.5)],            # (bits, clip at k x component RMS)
}
KIND_ID = {k: i for i, k in enumerate(SEVERITY)}


def _sign(rng):
    return 1.0 if rng.random() < 0.5 else -1.0


def _quantise(c, bits, k):
    """Clip a real component to +-k*RMS, then mid-rise quantise to 2**bits levels."""
    a = k * np.sqrt(np.mean(c ** 2))
    if a == 0:  # all-zero component: nothing to quantise
        return c
    step = 2 * a / 2 ** bits
    q = (np.floor(np.clip(c, -a, a) / step) + 0.5) * step
    return np.clip(q, -a + step / 2, a - step / 2)


def impair(x, kind: str, severity: int, rng: np.random.Generator, step: bool = True) -> np.ndarray:
    """One receiver impairment on a 1-D real or complex window; same length and kind out.

    rng: np.random.default_rng([seed, KIND_ID[kind], severity, row_id]) keeps every draw per-row.
    step=False drops the in-window gain step (constant offset only; gain-invariance check).
    """
    if kind not in SEVERITY or not 0 <= severity < len(SEVERITY[kind]):
        raise ValueError(f"unknown impairment {kind!r} / severity {severity}")
    if severity == 0:
        return x
    p = SEVERITY[kind][severity]
    x = np.asarray(x)
    cplx = np.iscomplexobj(x)
    y = x.astype(np.complex128 if cplx else np.float64)
    n = len(y)

    if kind == "awgn":
        sigma = np.sqrt(np.var(y) / 10 ** (p / 10))
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) if cplx else rng.standard_normal(n)
        y = y + sigma * noise
    elif kind == "gain":
        y = y * 10 ** (_sign(rng) * p / 20)
        if step:
            t0 = rng.integers(n // 4, 3 * n // 4)
            y[t0:] *= 10 ** (_sign(rng) * p / 20)
    elif kind == "tilt":
        # ponytail: circular FFT filter, window edges wrap; fine for independent 4000-sample windows
        if cplx:
            spec = np.fft.fftshift(np.fft.fft(y))
            ramp = 10 ** (_sign(rng) * np.linspace(-p / 2, p / 2, n) / 20)
            y = np.fft.ifft(np.fft.ifftshift(spec * ramp))
        else:
            spec = np.fft.rfft(y)
            ramp = 10 ** (_sign(rng) * np.linspace(-p / 2, p / 2, len(spec)) / 20)
            y = np.fft.irfft(spec * ramp, n=n)
    else:  # adc: full scale follows window RMS (per-window AGC), so gain and ADC stay separable
        bits, k = p
        y = _quantise(y.real, bits, k) + 1j * _quantise(y.imag, bits, k) if cplx else _quantise(y, bits, k)
    return y.astype(x.dtype if x.dtype.kind in "fc" else np.float64)


if __name__ == "__main__":
    from rfml_uav.data.wavelets import iq_wpd_features

    def db(a):
        return 10 * np.log10(a)

    def band_power(y):
        s = np.abs(np.fft.fftshift(np.fft.fft(y)) if np.iscomplexobj(y) else np.fft.rfft(y)) ** 2
        m = len(s) // 10
        return s[:m].mean(), s[-m:].mean()

    for cplx in (False, True):
        g = np.random.default_rng(1)
        w = g.standard_normal(40000) + (1j * g.standard_normal(40000) if cplx else 0)
        for kind in SEVERITY:
            assert impair(w, kind, 0, None) is w
            for s in range(1, 5):
                p = SEVERITY[kind][s]
                rng = lambda: np.random.default_rng([0, KIND_ID[kind], s, 7])
                y = impair(w, kind, s, rng())
                assert y.shape == w.shape and np.iscomplexobj(y) == cplx
                assert np.array_equal(y, impair(w, kind, s, rng()))
                if kind == "awgn":
                    assert abs(db(np.mean(np.abs(w) ** 2) / np.mean(np.abs(y - w) ** 2)) - p) < 0.5
                elif kind == "tilt":
                    (lo, hi), (lo0, hi0) = band_power(y), band_power(w)
                    # 10% bands are centred 0.9 of the span apart
                    assert abs(abs(db(hi / hi0) - db(lo / lo0)) - 0.9 * p) < 1, (kind, s, cplx)
                elif kind == "gain":
                    r0 = rng()
                    r0.random()  # replay the draws: offset sign, then step position
                    t0 = r0.integers(len(w) // 4, 3 * len(w) // 4)
                    r = db(np.mean(np.abs(y[t0:]) ** 2) / np.mean(np.abs(y[:t0]) ** 2) *
                           np.mean(np.abs(w[:t0]) ** 2) / np.mean(np.abs(w[t0:]) ** 2))
                    assert abs(abs(r) - p) < 0.5
                else:
                    bits, k = p
                    for c, c0 in ((y.real, w.real), (y.imag, w.imag)) if cplx else ((y, w),):
                        assert np.abs(c).max() <= k * np.sqrt(np.mean(c0 ** 2))
                        assert len(np.unique(c)) <= 2 ** bits
    for bad in (("cfo", 1), ("awgn", 5), ("adc", -1)):
        try:
            impair(w, *bad, np.random.default_rng(0))
            raise AssertionError(f"{bad} not rejected")
        except ValueError:
            pass

    # a constant gain offset does not change the unit-power I/Q features (DroneDetect gain invariance)
    g = np.random.default_rng(2)
    w = (g.standard_normal(4000) + 1j * g.standard_normal(4000)).astype(np.complex64)
    for s in (1, 2, 3):
        y = impair(w, "gain", s, np.random.default_rng([0, 1, s, 0]), step=False)
        assert np.allclose(iq_wpd_features(y), iq_wpd_features(w), rtol=1e-4, atol=1e-6)
    print("impairments self-check OK")
