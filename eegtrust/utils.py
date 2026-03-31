import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from .config import BANDPOWER_BANDS, SAMPLE_RATE
from sklearn.metrics import roc_auc_score, confusion_matrix


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
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    y_bin = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind in {'f', 'c'} else y_pred.astype(int)

    cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "false_alarm_rate": float(false_alarm_rate),
        "confusion_matrix": cm.tolist(),
    }
    if np.unique(y_true).size > 1 and y_pred.dtype.kind in {'f', 'c'}:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred))
    return metrics