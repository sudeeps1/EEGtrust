# EEGTrust Configuration File
# All project-wide constants and settings are defined here.

import os

# === Data Paths ===
EEG_DATA_ROOT = os.path.join('data', 'physionet.org', 'files', 'chbmit', '1.0.0', 'chb01')
CHBMIT_SUMMARY_FILE = 'chb01-summary.txt'  # Example, to be used per subject

# === Signal Processing ===
SAMPLE_RATE = 256  # Hz, CHB-MIT is typically 256Hz
WINDOW_SIZE_SEC = 10  # seconds per window
STRIDE_SEC = 5        # seconds between window starts

# Derived sample counts
WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_SEC
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SEC

# === Model Settings ===
ENCODER_TYPE = 'cnn'  # Options: 'cnn', 'transformer'
ENCODER_HIDDEN_DIM = 128
STGNN_NUM_LAYERS = 2
STGNN_ATTENTION_HEADS = 4

# === Feature Extraction ===
BANDPOWER_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}

# === Misc ===
SEED = 42
DEVICE = 'cuda'  # or 'cpu'

# === Seizure Detection Settings ===
SEIZURE_BUFFER_SEC = 1  # seconds buffer around seizures to exclude from non-seizure class 