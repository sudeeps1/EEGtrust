import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from .config import BANDPOWER_BANDS, SAMPLE_RATE


def compute_bandpower(data: np.ndarray, band: tuple, sfreq: int = SAMPLE_RATE) -> float:
    """
    Compute average bandpower in a frequency band for a 1D signal.
    """
    fmin, fmax = band
    freqs, psd = welch(data, sfreq, nperseg=sfreq*2)
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx_band], freqs[idx_band])


def compute_entropy(data: np.ndarray) -> float:
    """
    Compute Shannon entropy of a 1D signal.
    """
    hist, _ = np.histogram(data, bins=32, density=True)
    hist = hist + 1e-8  # avoid log(0)
    return scipy_entropy(hist, base=2)


def compute_variance(data: np.ndarray) -> float:
    return np.var(data)


def compute_features(window: np.ndarray, sfreq: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute features for a window of EEG data (channels, samples).
    Returns a 1D feature vector (all channels concatenated).
    Features: bandpower (delta, theta, alpha, beta, gamma), entropy, variance
    """
    feats = []
    for ch in window:
        # Bandpower for each band
        for band in BANDPOWER_BANDS.values():
            feats.append(compute_bandpower(ch, band, sfreq))
        # Entropy
        feats.append(compute_entropy(ch))
        # Variance
        feats.append(compute_variance(ch))
    return np.array(feats)

def compute_metrics(y_true, y_pred):
    # TODO: Compute AUC, sensitivity, specificity, false alarm rate, latency
    pass 