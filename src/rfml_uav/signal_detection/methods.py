import numpy as np
import pywt


# %% Energy detector
def energy_detector(x):
    return np.mean(np.asarray(x)**2)

# %% Wavelet detector
def wavelet_detector(x, wavelet='db4', level=4):
    coeffs = pywt.wavedec(np.asarray(x), wavelet, level=level)
    return sum(np.sum(c**2) for c in coeffs)

# %% Cyclostationary detector (spectral correlation function)
def cyclo_detector(x, fs, Nfft=512, num_alphas=32):
    """
    Blind cyclostationary detector using the spectral correlation function (SCF).

    Splits the signal into overlapping Hann-windowed STFT frames, then estimates
    the SCF as the time-averaged cross-product X(f) * conj(X(f + alpha_k)) for
    num_alphas candidate cyclic frequencies alpha_k = k * fs / Nfft, k = 1..num_alphas.

    For stationary noise E[X(f)*conj(X(f+alpha))] = 0 for alpha != 0.
    For cyclostationary signals (modulated UAV RF) the SCF is non-zero at one or
    more alpha values, making max mean |SCF| a robust detection statistic.

    Returns the maximum mean |SCF| across all scanned cyclic frequencies.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < Nfft:
        Nfft = max(16, 2 ** int(np.floor(np.log2(N))))
    hop = Nfft // 2
    window = np.hanning(Nfft)
    frames = np.array([
        np.fft.rfft(x[i:i + Nfft] * window)
        for i in range(0, N - Nfft + 1, hop)
    ])
    if frames.shape[0] < 2:
        return 0.0
    K, F = frames.shape
    max_k = min(num_alphas, F - 1)
    scf_peak = 0.0
    for k in range(1, max_k + 1):
        cross = frames[:, :F - k] * np.conj(frames[:, k:])
        val = float(np.mean(np.abs(np.mean(cross, axis=0))))
        if val > scf_peak:
            scf_peak = val
    return scf_peak
