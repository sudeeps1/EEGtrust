import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- SSLPretrainedEncoder ---
class SSLPretrainedEncoder(nn.Module):
    """
    Self-supervised encoder for EEG windows. Can be CNN or Transformer-based.
    """
    def __init__(self, input_channels, window_size, hidden_dim=128, encoder_type='cnn'):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'cnn':
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
                nn.ReLU()
            )
        elif encoder_type == 'transformer':
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_channels, nhead=4),
                num_layers=2
            )
            self.fc = nn.Linear(window_size * input_channels, hidden_dim)
        else:
            raise ValueError('Unknown encoder type')

    def forward(self, x):
        # x: (batch, channels, window_size)
        if self.encoder_type == 'cnn':
            return self.encoder(x)
        elif self.encoder_type == 'transformer':
            # Transformer expects (window_size, batch, channels)
            x = x.permute(2, 0, 1)
            out = self.encoder(x)
            out = out.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            return F.relu(self.fc(out))

# --- STGNN ---
class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network with attention for EEG windows.
    """
    def __init__(self, input_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, 2)  # seizure/non-seizure

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        for attn in self.layers:
            x, _ = attn(x, x, x)
        x = x.mean(dim=1)
        return self.fc(x)

# --- NeuroSymbolicExplainer ---
class NeuroSymbolicExplainer(nn.Module):
    """
    Rule-based output layer for symbolic explanations.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, features):
        logits = self.fc(features)
        # Optionally, add rule-based logic here
        return logits

class EEGMetaNet(nn.Module):
    def __init__(self, num_channels, window_samples, meta_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        cnn_out_dim = 64 * (window_samples // 4)  # after 2x MaxPool1d(2)
        self.meta_stream = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x_eeg, x_meta):
        # x_eeg: (batch, channels, samples)
        # x_meta: (batch, meta_dim)
        eeg_feat = self.cnn(x_eeg)
        meta_feat = self.meta_stream(x_meta)
        x = torch.cat([eeg_feat, meta_feat], dim=1)
        return self.fusion(x) 