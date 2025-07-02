import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import model classes and config
from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from eegtrust.config import WINDOW_SIZE_SAMPLES

# Load model
model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

encoder = SSLPretrainedEncoder(23, WINDOW_SIZE_SAMPLES, 128)
stgnn = STGNN(128, num_layers=2, num_heads=4)
explainer = NeuroSymbolicExplainer(128)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
stgnn.load_state_dict(checkpoint['stgnn_state_dict'])
explainer.load_state_dict(checkpoint['explainer_state_dict'])

encoder.eval()
stgnn.eval()
explainer.eval()

encoder.to(device)
stgnn.to(device)
explainer.to(device)

# Load test data
windows = []
labels = []
for subj in ['chb01', 'chb02']:
    windows.append(np.load(f'prepared_data/{subj}_windows.npy'))
    labels.append(np.load(f'prepared_data/{subj}_labels.npy'))
X = np.concatenate(windows, axis=0)
y = np.concatenate(labels, axis=0).astype(np.int64)

# Inference
print(f"Running inference on {len(X)} windows...")
y_pred = []
y_prob = []
with torch.no_grad():
    for i in range(len(X)):
        eeg_window = X[i]
        # Normalize (z-score per channel)
        mean = np.mean(eeg_window, axis=1, keepdims=True)
        std = np.std(eeg_window, axis=1, keepdims=True)
        normalized = (eeg_window - mean) / (std + 1e-8)
        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
        features = encoder(input_tensor)
        features_seq = features.unsqueeze(1)
        stgnn_logits = stgnn(features_seq)
        explainer_logits = explainer(features)
        logits = (stgnn_logits + explainer_logits) / 2
        probs = torch.softmax(logits, dim=1)
        seizure_prob = probs[0, 1].item()
        pred = 1 if seizure_prob > 0.5 else 0
        y_pred.append(pred)
        y_prob.append(seizure_prob)

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

# Metrics
print("\n===== Manual Evaluation Metrics =====")
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, zero_division=0))
print("Recall:", recall_score(y, y_pred, zero_division=0))
print("F1 Score:", f1_score(y, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y, y_prob))

# Print class distribution
print("\nClass distribution in test set:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Label {label}: {count}")

# Analyze output probabilities
print("\nModel output probability stats (seizure class):")
print(f"  Min: {y_prob.min():.4f}")
print(f"  Max: {y_prob.max():.4f}")
print(f"  Mean: {y_prob.mean():.4f}")
print(f"  Median: {np.median(y_prob):.4f}")

# Try different thresholds
print("\nMetrics at different thresholds:")
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_thresh = (y_prob > thresh).astype(int)
    f1 = f1_score(y, y_pred_thresh, zero_division=0)
    recall = recall_score(y, y_pred_thresh, zero_division=0)
    precision = precision_score(y, y_pred_thresh, zero_division=0)
    print(f"  Threshold {thresh:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}") 