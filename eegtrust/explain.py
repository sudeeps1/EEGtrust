import torch
from captum.attr import IntegratedGradients
# from .model import EEGMetaNet
import numpy as np

def compute_saliency(model, x_eeg, x_meta):
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute((x_eeg, x_meta), target=0, return_convergence_delta=True)
    return attributions, delta 

def symbolic_explanation(features: np.ndarray) -> str:
    """
    Given a feature vector, return a human-readable rule if logic is matched.
    Example: If bandpower in beta band is high and entropy is low, return a rule.
    """
    # Placeholder logic: if feature[beta_band_idx] > threshold and entropy < threshold
    beta_idx = 3  # assuming band order: delta, theta, alpha, beta, gamma
    entropy_idx = 5 * 5  # 5 bands * 5 channels, then first entropy
    if features[beta_idx] > 2.0 and features[entropy_idx] < 2.5:
        return "High beta power and low entropy detected: possible seizure onset."
    return "No symbolic rule matched." 