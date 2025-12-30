import numpy as np
import pywt


# %% Energy detector
def energy_detector(x):
    return np.mean(np.asarray(x)**2)

# %% Wavelet detector
def wavelet_detector(x, wavelet='db4', level=4):
    coeffs = pywt.wavedec(np.asarray(x), wavelet, level=level)
    return sum(np.sum(c**2) for c in coeffs)

# %% Cyclostationary detector (simple estimator)
def cyclo_detector(x, fs, alpha=10.0, Nfft=1024):
    N = len(x)
    t = np.arange(N) / fs
    x_shift = np.asarray(x) * np.exp(-1j * 2 * np.pi * alpha * t)
    Xf = np.fft.fftshift(np.fft.fft(x_shift, Nfft))
    power = np.abs(Xf)**2
    return float(np.mean(power))
