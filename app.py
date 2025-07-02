import os
from eegtrust.config import EEG_DATA_ROOT, SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES
from eegtrust.metadata import parse_seizure_summary
from eegtrust.data import load_eeg_data
from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from eegtrust.inference import stream_eeg

# Example: use chb01_03.edf
subject = 'chb01'
edf_file = 'chb01_03.edf'
edf_path = os.path.join(EEG_DATA_ROOT, subject, edf_file)
summary_path = os.path.join(EEG_DATA_ROOT, subject, f'{subject}-summary.txt')

# Parse seizure intervals
seizure_dict = parse_seizure_summary(summary_path)
seizure_intervals = seizure_dict.get(edf_file, [])

# Load and preprocess EEG data
print(f'Loading {edf_file}...')
data, labels = load_eeg_data(edf_path, seizure_intervals, SAMPLE_RATE)

# Instantiate model (random weights for demo)
input_channels = data.shape[0]
encoder = SSLPretrainedEncoder(input_channels, WINDOW_SIZE_SAMPLES)
stgnn = STGNN(encoder.encoder[-2].out_features)
explainer = NeuroSymbolicExplainer(2)

# Compose model pipeline
import torch
class SeizureModel(torch.nn.Module):
    def __init__(self, encoder, stgnn, explainer):
        super().__init__()
        self.encoder = encoder
        self.stgnn = stgnn
        self.explainer = explainer
    def forward(self, x):
        feats = self.encoder(x)
        feats = feats.unsqueeze(1)
        out = self.stgnn(feats)
        logits = self.explainer(out)
        return logits

model = SeizureModel(encoder, stgnn, explainer)
model.eval()

# Run streaming inference
print('Running inference...')
alerts = stream_eeg(data, model, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES, threshold=0.5)

# Print alerts and explanations
for idx, alert, explanation in alerts:
    print(f'ALERT: {alert}\nExplanation: {explanation}\n') 