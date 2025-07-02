import os
import mne
import numpy as np
import pandas as pd
from .config import EEG_DATA_ROOT, SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES
from typing import Tuple, List

# List all subjects in CHB-MIT
# (Update: not used in new pipeline, but kept for reference)
def get_chbmit_subjects():
    return [d for d in os.listdir(EEG_DATA_ROOT) if d.startswith('chb') and os.path.isdir(os.path.join(EEG_DATA_ROOT, d))]

# Load EDF file for a subject/session
def load_eeg_data(edf_path: str, seizure_intervals: List[Tuple[int, int]],
                  sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an EDF EEG file, preprocess (bandpass filter, resample, normalize),
    and return data and binary seizure labels per sample.
    Args:
        edf_path: Path to EDF file
        seizure_intervals: List of (start, end) tuples in seconds
        sample_rate: Target sample rate (Hz)
    Returns:
        data: np.ndarray (channels, samples)
        labels: np.ndarray (samples,) binary (1=seizure, 0=non-seizure)
    """
    # Load raw EEG
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick('eeg')
    # Bandpass filter (1-70 Hz)
    raw.filter(1., 70., fir_design='firwin', verbose=False)
    # Resample
    raw.resample(sample_rate, npad="auto", verbose=False)
    # Get data
    data = raw.get_data().astype(np.float32)  # shape: (n_channels, n_samples)
    # Normalize (z-score per channel)
    data = ((data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    n_samples = data.shape[1]
    # Create binary label vector (0=non-seizure, 1=seizure)
    labels = np.zeros(n_samples, dtype=np.float32)
    for start, end in seizure_intervals:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        labels[start_idx:end_idx] = 1.0
    return data, labels

# Extract metadata from summary file
def get_chbmit_metadata():
    # CHB-MIT does not provide age/sex per subject, but all are pediatric
    subjects = get_chbmit_subjects()
    meta = []
    for subj in subjects:
        meta.append({'subject_id': subj, 'age': np.nan, 'sex': 'F', 'region': 'unknown'})
    return pd.DataFrame(meta)

def filter_pediatric(metadata_df):
    # TODO: Filter metadata for pediatric subjects
    pass

def preprocess_eeg(raw, resample_freq=SAMPLE_RATE):
    # Normalize, resample, band-pass filter
    raw = raw.copy().resample(resample_freq)
    raw = raw.filter(0.5, 70., fir_design='firwin')
    # Z-score normalization per channel
    data = raw.get_data()
    data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    raw._data = data
    return raw

def remove_artifacts(raw):
    # TODO: Implement ICA or threshold-based artifact removal
    pass 

def load_eeg_data_chunked(edf_path: str, seizure_intervals: List[Tuple[int, int]],
                        sample_rate: int = SAMPLE_RATE, chunk_size: int = 1_000_000,
                        window_size: int = None, stride: int = None):
    """
    Load an EDF EEG file in chunks, yielding (window, label) pairs for each window in each chunk.
    Implements seizure buffer logic to exclude non-seizure windows within 1 second of seizures.
    Args:
        edf_path: Path to EDF file
        seizure_intervals: List of (start, end) tuples in seconds
        sample_rate: Target sample rate (Hz)
        chunk_size: Number of samples per chunk
        window_size: Number of samples per window (default: use config)
        stride: Number of samples per stride (default: use config)
    Yields:
        window: np.ndarray (channels, window_size)
        label: float (1.0=seizure, 0.0=non-seizure)
    """
    if window_size is None:
        window_size = WINDOW_SIZE_SAMPLES
    if stride is None:
        stride = STRIDE_SAMPLES
    
    # Load raw EEG with preload=True to enable filtering
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick('eeg')
    
    # Get original sample rate and total samples
    orig_sample_rate = raw.info['sfreq']
    n_samples = int(raw.n_times)
    n_channels = len(raw.ch_names)
    
    print(f"    Processing {n_samples} samples at {orig_sample_rate}Hz, resampling to {sample_rate}Hz")
    
    # Apply preprocessing to the entire file
    raw.filter(1., 70., fir_design='firwin', verbose=False)
    raw.resample(sample_rate, npad="auto", verbose=False)
    
    # Get the full preprocessed data
    data = raw.get_data().astype(np.float32)
    
    # Normalize (z-score per channel)
    data = ((data - np.mean(data, axis=1, keepdims=True)) / 
            (np.std(data, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    
    # Create binary label vector for the entire file
    labels = np.zeros(data.shape[1], dtype=np.float32)
    for start, end in seizure_intervals:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        if end_idx > start_idx and start_idx < data.shape[1]:
            labels[start_idx:min(end_idx, data.shape[1])] = 1.0
    
    # Create seizure buffer mask (1 second buffer around seizures)
    from .config import SEIZURE_BUFFER_SEC
    buffer_samples = int(SEIZURE_BUFFER_SEC * sample_rate)
    seizure_buffer = np.zeros(data.shape[1], dtype=bool)
    
    for start, end in seizure_intervals:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        
        # Mark seizure period
        if end_idx > start_idx and start_idx < data.shape[1]:
            seizure_buffer[start_idx:min(end_idx, data.shape[1])] = True
        
        # Mark buffer period before seizure
        buffer_start = max(0, start_idx - buffer_samples)
        seizure_buffer[buffer_start:start_idx] = True
        
        # Mark buffer period after seizure
        buffer_end = min(data.shape[1], end_idx + buffer_samples)
        seizure_buffer[end_idx:buffer_end] = True
    
    # Calculate total number of windows that will be generated
    total_windows = (data.shape[1] - window_size) // stride + 1
    print(f"    Will generate approximately {total_windows} windows")
    
    # Generate windows with seizure buffer logic
    window_count = 0
    excluded_count = 0
    
    for win_start in range(0, data.shape[1] - window_size + 1, stride):
        win_end = win_start + window_size
        
        # Ensure we don't go beyond the data bounds
        if win_end > data.shape[1]:
            break
        
        # Check if this window contains any seizure
        window_contains_seizure = np.any(labels[win_start:win_end])
        
        # Check if this window is entirely within seizure buffer (for non-seizure windows)
        window_in_buffer = np.all(seizure_buffer[win_start:win_end])
        
        # Skip non-seizure windows that are entirely within the seizure buffer
        if not window_contains_seizure and window_in_buffer:
            excluded_count += 1
            continue
            
        window = data[:, win_start:win_end]
        label = 1.0 if window_contains_seizure else 0.0
        
        window_count += 1
        
        # Progress indicator every 1000 windows
        if window_count % 1000 == 0:
            progress = (window_count / total_windows) * 100
            print(f"    Window progress: {progress:.1f}% ({window_count}/{total_windows} windows, {excluded_count} excluded)")
        
        yield window, label
    
    print(f"    Generated {window_count} windows total, excluded {excluded_count} windows in seizure buffer") 