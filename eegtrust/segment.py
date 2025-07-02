import numpy as np
from typing import Tuple

def segment_eeg(raw, window_size, stride, labels, sfreq):
    """
    Segment EEG data into sliding windows.
    Args:
        raw: mne.io.Raw object
        window_size: seconds
        stride: seconds
        labels: list of (start, end, label) tuples
        sfreq: sampling frequency
    Returns:
        segments: np.ndarray (num_windows, channels, window_samples)
        segment_labels: list
    """
    data = raw.get_data()
    n_channels, n_samples = data.shape
    win_samples = int(window_size * sfreq)
    stride_samples = int(stride * sfreq)
    segments = []
    segment_labels = []
    for start in range(0, n_samples - win_samples + 1, stride_samples):
        end = start + win_samples
        seg = data[:, start:end]
        segments.append(seg)
        # Label: 1 if any seizure in window, else 0
        label = 0
        for (sz_start, sz_end, sz_label) in labels:
            if sz_start < end/sfreq and sz_end > start/sfreq:
                label = 1
                break
        segment_labels.append(label)
    return np.stack(segments), segment_labels

def sliding_window(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Segment EEG data into overlapping windows.
    Args:
        data: np.ndarray (channels, samples)
        window_size: int, number of samples per window
        stride: int, number of samples to move window each step
    Returns:
        windows: np.ndarray (num_windows, channels, window_size)
    """
    channels, n_samples = data.shape
    num_windows = 1 + (n_samples - window_size) // stride
    windows = np.zeros((num_windows, channels, window_size), dtype=data.dtype)
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[:, start:end]
    return windows 