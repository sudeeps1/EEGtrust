import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from .model import EEGMetaNet, SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from .data import get_chbmit_subjects, load_eeg_data, preprocess_eeg, get_chbmit_metadata
from .segment import segment_eeg
from .metadata import extract_metadata, encode_metadata
from .config import WINDOW_SIZE, WINDOW_STRIDE, SAMPLING_FREQ
import os
from .utils import compute_features

# TODO: Implement EEGDataset class and data loading
class EEGDataset(Dataset):
    def __init__(self, X_eeg, X_meta, y):
        self.X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
        self.X_meta = torch.tensor(X_meta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_eeg[idx], self.X_meta[idx], self.y[idx]

# --- Dataset for SSL (SimCLR-style) ---
class EEGSimCLRDataset(Dataset):
    def __init__(self, windows, augment_fn):
        self.windows = windows  # shape: (num_windows, channels, window_samples)
        self.augment_fn = augment_fn
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        x = self.windows[idx]
        aug1 = self.augment_fn(x)
        aug2 = self.augment_fn(x)
        return aug1, aug2

# --- Dataset for supervised training ---
class EEGWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

# --- Simple augmentation for SSL ---
def eeg_augment(x):
    # x: (channels, window_samples)
    x_aug = x.copy()
    # Add Gaussian noise
    x_aug += np.random.normal(0, 0.1, x.shape)
    # Random channel dropout
    if np.random.rand() < 0.2:
        ch = np.random.randint(0, x.shape[0])
        x_aug[ch] = 0
    return x_aug

# --- SimCLR-style contrastive loss ---
def nt_xent_loss(z1, z2, temperature=0.5):
    # z1, z2: (batch, dim)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    labels = torch.arange(z1.size(0)).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(labels.size(0), dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    positives = torch.cat([torch.diag(similarity_matrix, z1.size(0)), torch.diag(similarity_matrix, -z1.size(0))], dim=0)
    negatives = similarity_matrix[~mask].view(labels.size(0), -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    loss = nn.CrossEntropyLoss()(logits, torch.zeros(labels.size(0), dtype=torch.long).to(z1.device))
    return loss

def prepare_data():
    subjects = get_chbmit_subjects()
    metadata_df = get_chbmit_metadata()
    X_eeg, X_meta, y, groups = [], [], [], []
    for subj in tqdm(subjects, desc='Subjects'):
        subj_meta = extract_metadata(subj, metadata_df)
        subj_dir = f'chbmit-edf/{subj}'
        edf_files = [f for f in os.listdir(f'../data/raw/{subj_dir}') if f.endswith('.edf')]
        for edf_file in edf_files:
            raw, labels = load_eeg_data(subj, edf_file)
            raw = preprocess_eeg(raw, SAMPLING_FREQ)
            segs, seg_labels = segment_eeg(raw, WINDOW_SIZE, WINDOW_STRIDE, labels, SAMPLING_FREQ)
            meta_vec = encode_metadata(subj_meta)
            X_eeg.append(segs)
            X_meta.append(np.tile(meta_vec, (len(segs), 1)))
            y.extend(seg_labels)
            groups.extend([subj]*len(segs))
    X_eeg = np.concatenate(X_eeg)
    X_meta = np.concatenate(X_meta)
    y = np.array(y)
    groups = np.array(groups)
    return X_eeg, X_meta, y, groups

def train():
    X_eeg, X_meta, y, groups = prepare_data()
    num_channels = X_eeg.shape[1]
    window_samples = X_eeg.shape[2]
    meta_dim = X_meta.shape[1]
    model = EEGMetaNet(num_channels, window_samples, meta_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_eeg, y, groups)):
        train_set = EEGDataset(X_eeg[train_idx], X_meta[train_idx], y[train_idx])
        val_set = EEGDataset(X_eeg[val_idx], X_meta[val_idx], y[val_idx])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        best_val_loss = float('inf')
        patience, patience_counter = 10, 0
        for epoch in range(100):
            model.train()
            train_loss = 0
            for x_eeg, x_meta, yb in train_loader:
                optimizer.zero_grad()
                out = model(x_eeg, x_meta)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*len(yb)
            train_loss /= len(train_set)
            # Validation
            model.eval()
            val_loss = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for x_eeg, x_meta, yb in val_loader:
                    out = model(x_eeg, x_meta)
                    loss = criterion(out, yb)
                    val_loss += loss.item()*len(yb)
                    y_true.extend(yb.cpu().numpy().flatten())
                    y_pred.extend(out.cpu().numpy().flatten())
            val_loss /= len(val_set)
            writer.add_scalars(f'Fold{fold}', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            print(f"Fold {fold} Epoch {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{fold}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break
    writer.close()

# --- SSL Pretraining ---
def train_ssl_encoder(encoder: SSLPretrainedEncoder, windows: np.ndarray, batch_size=64, epochs=50, device='cuda'):
    """
    Train the encoder using SimCLR contrastive loss.
    Args:
        encoder: SSLPretrainedEncoder
        windows: np.ndarray (num_windows, channels, window_samples)
    Returns:
        Trained encoder
    """
    encoder = encoder.to(device)
    dataset = EEGSimCLRDataset(windows, eeg_augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    for epoch in range(epochs):
        encoder.train()
        total_loss = 0
        for aug1, aug2 in tqdm(dataloader, desc=f'SSL Epoch {epoch+1}/{epochs}'):
            aug1 = torch.tensor(aug1, dtype=torch.float32).to(device)
            aug2 = torch.tensor(aug2, dtype=torch.float32).to(device)
            z1 = encoder(aug1)
            z2 = encoder(aug2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * aug1.size(0)
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}: SSL Loss = {avg_loss:.4f}')
    return encoder

# --- Supervised Fine-tuning ---
def train_seizure_model(encoder: SSLPretrainedEncoder, stgnn: STGNN, explainer: NeuroSymbolicExplainer,
                       windows: np.ndarray, labels: np.ndarray, batch_size=64, epochs=30, device='cuda'):
    """
    Fine-tune encoder with STGNN and explanation head for seizure detection.
    Args:
        encoder: Pretrained SSL encoder
        stgnn: Spatiotemporal GNN
        explainer: NeuroSymbolicExplainer
        windows: np.ndarray (num_windows, channels, window_samples)
        labels: np.ndarray (num_windows,)
    Returns:
        Trained seizure model
    """
    encoder = encoder.to(device)
    stgnn = stgnn.to(device)
    explainer = explainer.to(device)
    dataset = EEGWindowDataset(windows, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params = list(encoder.parameters()) + list(stgnn.parameters()) + list(explainer.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        encoder.train()
        stgnn.train()
        explainer.train()
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f'Supervised Epoch {epoch+1}/{epochs}'):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            feats = encoder(x)  # (batch, hidden_dim)
            feats_seq = feats.unsqueeze(1)  # (batch, seq_len=1, hidden_dim)
            logits = stgnn(feats_seq)  # (batch, 2)
            logits = explainer(logits)  # (batch, 2)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}: Supervised Loss = {avg_loss:.4f}')
    return encoder, stgnn, explainer

if __name__ == "__main__":
    train() 