import numpy as np
import pywt


def wpd_features(x, wavelet: str = "dmey", level: int = 6) -> list[float]:
    """RMS and STD of every frequency-ordered wavelet-packet node at `level` (128 values for level 6)."""
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric")
    features = []
    for node in wp.get_level(level, "freq"):
        features.extend([np.sqrt(np.mean(np.square(node.data))), np.std(node.data)])
    return features


def iq_wpd_features(window, wavelet: str = "dmey", level: int = 6) -> np.ndarray:
    """Complex I/Q window -> DC removed, unit mean power, WPD(I) ‖ WPD(Q) (256 values for level 6)."""
    w = np.asarray(window, dtype=np.complex128)
    w = w - w.mean()
    w = w / np.sqrt(np.mean(np.abs(w) ** 2))
    return np.array(wpd_features(w.real, wavelet, level) + wpd_features(w.imag, wavelet, level), dtype=np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)).astype(np.complex64)
    f = iq_wpd_features(w)
    assert f.shape == (256,)
    # receiver gain and DC offset do not change the features
    assert np.allclose(iq_wpd_features(7 * w + (0.05 + 0.09j)), f, rtol=1e-4, atol=1e-6)
    print("wavelets self-check OK")
