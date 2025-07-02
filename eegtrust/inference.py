import numpy as np
from collections import deque
# from .model import EEGMetaNet
from .segment import sliding_window
from .utils import compute_features
from .explain import symbolic_explanation
import torch

def real_time_inference(model, stream, buffer_seconds=10, window_size=2, stride=1, sfreq=256):
    buffer = deque(maxlen=int(buffer_seconds * sfreq))
    for chunk in stream:
        buffer.extend(chunk)
        # Preprocess buffer (normalize, filter, etc.)
        # Segment buffer into windows
        # Predict per window
        # Post-process (majority voting, smoothing)
        # Trigger alert if needed
        pass 

def stream_eeg(data: np.ndarray, model, window_size: int, stride: int, threshold: float = 0.5):
    """
    Run inference on EEG data in a sliding window fashion.
    Args:
        data: np.ndarray (channels, samples)
        model: PyTorch model (encoder + classifier)
        window_size: int, samples per window
        stride: int, samples per step
        threshold: float, probability threshold for seizure alert
    Returns:
        List of (window_idx, alert, explanation)
    """
    alerts = []
    windows = sliding_window(data, window_size, stride)
    for i, window in enumerate(windows):
        feats = compute_features(window)
        feats_tensor = np.expand_dims(feats, 0)  # batch dim
        # Assume model returns logits or probabilities
        with torch.no_grad():
            prob = model(torch.tensor(feats_tensor, dtype=torch.float32)).softmax(-1)[0,1].item()
        if prob > threshold:
            explanation = symbolic_explanation(feats)
            alert = f"Seizure forecasted at window {i} (p={prob:.2f})"
            # Trigger mock webhook (placeholder)
            print(f"[WEBHOOK] {alert} | {explanation}")
            alerts.append((i, alert, explanation))
    return alerts 