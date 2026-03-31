import numpy as np
from collections import deque
# from .model import EEGMetaNet
from .segment import sliding_window
from .utils import compute_features
from .explain import symbolic_explanation
import torch

def real_time_inference(model, stream, buffer_seconds=10, window_size=2, stride=1, sfreq=256):
    buffer = deque(maxlen=int(buffer_seconds * sfreq))
    alerts = []
    for chunk in stream:
        buffer.extend(chunk)
        if len(buffer) < int(window_size * sfreq):
            continue
        buffer_arr = np.asarray(buffer, dtype=np.float32)
        if buffer_arr.ndim == 1:
            buffer_arr = buffer_arr[np.newaxis, :]
        alerts.extend(
            stream_eeg(
                data=buffer_arr,
                model=model,
                window_size=int(window_size * sfreq),
                stride=int(stride * sfreq),
            )
        )
    return alerts

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
    device = next(model.parameters()).device
    for i, window in enumerate(windows):
        window_tensor = np.expand_dims(window, 0)  # batch, channels, samples
        with torch.no_grad():
            window_t = torch.from_numpy(window_tensor).to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(window_t)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        if prob > threshold:
            feats = compute_features(window)
            explanation = symbolic_explanation(feats)
            alert = f"Seizure forecasted at window {i} (p={prob:.2f})"
            # Trigger mock webhook (placeholder)
            print(f"[WEBHOOK] {alert} | {explanation}")
            alerts.append((i, alert, explanation))
    return alerts 