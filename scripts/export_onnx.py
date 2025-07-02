import torch
from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from eegtrust.config import WINDOW_SIZE_SAMPLES

# Example model parameters (should match training)
input_channels = 23  # CHB-MIT has 23 EEG channels
hidden_dim = 128
window_size = WINDOW_SIZE_SAMPLES

# Instantiate model (should load trained weights in real use)
encoder = SSLPretrainedEncoder(input_channels, window_size, hidden_dim)
stgnn = STGNN(hidden_dim)
explainer = NeuroSymbolicExplainer(2)

# Example: simple pipeline (encoder -> stgnn -> explainer)
class SeizureModel(torch.nn.Module):
    def __init__(self, encoder, stgnn, explainer):
        super().__init__()
        self.encoder = encoder
        self.stgnn = stgnn
        self.explainer = explainer
    def forward(self, x):
        # x: (batch, channels, window_size)
        feats = self.encoder(x)
        # Add dummy seq_len dimension for STGNN
        feats = feats.unsqueeze(1)
        out = self.stgnn(feats)
        logits = self.explainer(out)
        return logits

model = SeizureModel(encoder, stgnn, explainer)
model.eval()

# Dummy input: batch=1, channels=23, window_size
x_dummy = torch.randn(1, input_channels, window_size)

# Export to ONNX
onnx_path = 'seizure_model.onnx'
torch.onnx.export(
    model, x_dummy, onnx_path,
    input_names=['eeg'], output_names=['logits'],
    dynamic_axes={'eeg': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17
)
print(f"Exported model to {onnx_path}") 