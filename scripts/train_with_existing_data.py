#!/usr/bin/env python3
"""
Train EEGTrust model with existing preprocessed data (chb01 and chb02)
Memory-efficient version with class imbalance handling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
import os
import sys
import random
import bisect
import contextlib
from tqdm import tqdm

# Add the parent directory to the path to import eegtrust modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer, FocalLoss
from eegtrust.config import WINDOW_SIZE_SAMPLES, SAMPLE_RATE

DEFAULT_SEED = 42


def set_determinism(seed: int = DEFAULT_SEED):
    """Enable deterministic and reproducible execution across workers/devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MemoryEfficientEEGDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand"""
    
    def __init__(self, data_files, label_files, transform=None):
        """
        Args:
            data_files: List of paths to .npy files containing windows
            label_files: List of paths to .npy files containing labels
            transform: Optional transform to apply to data
        """
        self.data_files = data_files
        self.label_files = label_files
        self.transform = transform
        self._labels_memmap = [np.load(label_file, mmap_mode='r') for label_file in label_files]
        self._windows_memmap = [None] * len(data_files)
        
        # Calculate total length
        self.lengths = []
        
        for data_file in data_files:
            self.lengths.append(np.load(data_file, mmap_mode='r').shape[0])

        self._cumulative_lengths = np.cumsum(self.lengths, dtype=np.int64)
        
        self.total_length = sum(self.lengths)
        print(f"Dataset initialized with {self.total_length} samples across {len(data_files)} files")

    def _locate_index(self, idx: int):
        file_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        prev_cum = 0 if file_idx == 0 else int(self._cumulative_lengths[file_idx - 1])
        return file_idx, idx - prev_cum

    def get_label_by_global_idx(self, idx: int) -> int:
        """Fast label lookup without loading full window data."""
        file_idx, local_idx = self._locate_index(idx)
        if 0 <= file_idx < len(self._labels_memmap):
            return int(self._labels_memmap[file_idx][local_idx])
        raise IndexError(f"Index out of range: {idx}")
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Load the specific sample from the file
        try:
            file_idx, local_idx = self._locate_index(idx)
            if self._windows_memmap[file_idx] is None:
                self._windows_memmap[file_idx] = np.load(self.data_files[file_idx], mmap_mode='r')

            window = np.asarray(self._windows_memmap[file_idx][local_idx], dtype=np.float32)
            label = int(self._labels_memmap[file_idx][local_idx])
            
            # Convert to tensor
            window = torch.from_numpy(window)
            label = torch.tensor(label, dtype=torch.long)
            
            if self.transform:
                window = self.transform(window)
            
            return window, label
            
        except Exception as e:
            print(f"Error loading sample {idx} from file {file_idx}: {e}")
            # Return a dummy sample in case of error
            dummy_window = torch.zeros((23, 1024), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_window, dummy_label

def get_data_info():
    """Get information about the preprocessed data without loading it all"""
    print("Analyzing preprocessed data...")
    
    data_files = [
        'prepared_data/chb01_windows.npy',
        'prepared_data/chb02_windows.npy'
    ]
    label_files = [
        'prepared_data/chb01_labels.npy',
        'prepared_data/chb02_labels.npy'
    ]
    
    total_samples = 0
    class_counts = {0: 0, 1: 0}
    
    for data_file, label_file in zip(data_files, label_files):
        if os.path.exists(data_file) and os.path.exists(label_file):
            # Get shape info
            data_shape = np.load(data_file, mmap_mode='r').shape
            labels = np.load(label_file, mmap_mode='r')
            
            total_samples += data_shape[0]
            
            # Count classes
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u] += c
            
            print(f"{data_file}: {data_shape[0]} samples, shape {data_shape}")
        else:
            print(f"Warning: {data_file} or {label_file} not found")
    
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {class_counts}")
    if class_counts[1] > 0:
        print(f"Class imbalance ratio: {class_counts[0] / class_counts[1]:.2f}:1")
    else:
        print("Class imbalance ratio: inf (no seizure samples found)")
    
    return data_files, label_files, total_samples, class_counts

def calculate_class_weights(class_counts):
    """Calculate class weights for weighted loss"""
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        1.0,  # weight for class 0 (non-seizure)
        class_counts[0] / class_counts[1]  # weight for class 1 (seizure)
    ], dtype=torch.float32)
    
    print(f"Class weights: {class_weights}")
    return class_weights

def create_weighted_sampler(dataset, class_counts, seed: int = DEFAULT_SEED):
    """Create weighted random sampler for oversampling minority class"""
    # Calculate sampling weights
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(2)]
    
    sample_weights = np.empty(len(dataset), dtype=np.float32)
    print("Creating weighted sampler...")
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
        for i, global_idx in enumerate(dataset.indices):
            if i % 10000 == 0:
                print(f"  Processing sample {i}/{len(dataset.indices)}")
            label = base_dataset.get_label_by_global_idx(global_idx)
            sample_weights[i] = class_weights[int(label)]
    else:
        for i in range(len(dataset)):
            if i % 10000 == 0:
                print(f"  Processing sample {i}/{len(dataset)}")
            _, label = dataset[i]
            sample_weights[i] = class_weights[int(label)]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    
    print(f"Created weighted sampler with weights: {class_weights}")
    return sampler

def train_model(data_files, label_files, class_counts, device='cuda', epochs=50, batch_size=16, use_focal_loss=True, seed: int = DEFAULT_SEED):
    """Train the EEGTrust model with class imbalance handling"""
    
    # Create memory-efficient dataset
    dataset = MemoryEfficientEEGDataset(data_files, label_files)
    
    # Split into train/val/test (70/15/15)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=split_generator)
    
    # Create weighted sampler for the training set
    weighted_sampler = create_weighted_sampler(train_dataset, class_counts, seed=seed)
    
    # Create data loaders - use weighted sampler for training
    is_cuda = str(device).startswith("cuda")
    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": is_cuda,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=weighted_sampler,
        **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    # Get sample to determine dimensions
    sample_window, _ = dataset[0]
    num_channels = sample_window.shape[0]
    window_samples = sample_window.shape[1]
    hidden_dim = 128
    
    print(f"Model dimensions: channels={num_channels}, window_samples={window_samples}")
    
    # Initialize models
    encoder = SSLPretrainedEncoder(num_channels, window_samples, hidden_dim)
    stgnn = STGNN(hidden_dim, num_layers=2, num_heads=4)
    explainer = NeuroSymbolicExplainer(hidden_dim)
    
    # Move to device
    encoder = encoder.to(device)
    stgnn = stgnn.to(device)
    explainer = explainer.to(device)
    
    # Setup training
    params = list(encoder.parameters()) + list(stgnn.parameters()) + list(explainer.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)
    
    # Setup loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        class_weights = calculate_class_weights(class_counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropy Loss")
    
    # Training loop
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"Starting training on {device}...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"DataLoader workers: {num_workers}")

    def forward_logits(xb):
        features = encoder(xb)
        features_seq = features.unsqueeze(1)
        stgnn_logits = stgnn(features_seq)
        explainer_logits = explainer(features)
        return (stgnn_logits + explainer_logits) / 2
    
    for epoch in range(epochs):
        # Training
        encoder.train()
        stgnn.train()
        explainer.train()
        
        train_loss = 0
        train_predictions = []
        train_labels = []
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            x = x.to(device, non_blocking=is_cuda)
            y = y.to(device, non_blocking=is_cuda)
            
            optimizer.zero_grad(set_to_none=True)
            with (torch.cuda.amp.autocast() if is_cuda else contextlib.nullcontext()):
                logits = forward_logits(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            predicted = torch.argmax(logits.detach(), dim=1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_f1 = f1_score(train_labels, train_predictions, average='weighted')
        train_f1_seizure = f1_score(train_labels, train_predictions, average=None)[1] if 1 in train_labels else 0
        
        # Validation
        encoder.eval()
        stgnn.eval()
        explainer.eval()
        
        val_loss = 0
        val_predictions = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=is_cuda)
                y = y.to(device, non_blocking=is_cuda)
                with (torch.cuda.amp.autocast() if is_cuda else contextlib.nullcontext()):
                    logits = forward_logits(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                # Store predictions and probabilities for metrics
                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(y.detach().cpu().numpy())
                val_probs.extend(probs[:, 1].detach().cpu().numpy())  # Probability of seizure class
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')
        val_f1_seizure = f1_score(val_labels, val_predictions, average=None)[1] if 1 in val_labels else 0
        val_auroc = roc_auc_score(val_labels, val_probs) if 1 in val_labels else 0
        val_auprc = average_precision_score(val_labels, val_probs) if 1 in val_labels else 0
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train F1-Seizure: {train_f1_seizure:.4f}')
        print(f'          Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val F1-Seizure: {val_f1_seizure:.4f}')
        print(f'          Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}')
        
        # Early stopping based on F1-score for seizure class
        if val_f1_seizure > best_val_f1:
            best_val_f1 = val_f1_seizure
            patience_counter = 0
            # Save best model
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'stgnn_state_dict': stgnn.state_dict(),
                'explainer_state_dict': explainer.state_dict(),
                'epoch': epoch,
                'val_f1_seizure': val_f1_seizure,
                'val_auroc': val_auroc,
                'val_auprc': val_auprc
            }, 'best_model.pth')
            print(f"Saved best model with validation F1-seizure: {val_f1_seizure:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Load best model for testing
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    stgnn.load_state_dict(checkpoint['stgnn_state_dict'])
    explainer.load_state_dict(checkpoint['explainer_state_dict'])
    
    # Test
    encoder.eval()
    stgnn.eval()
    explainer.eval()
    
    test_predictions = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=is_cuda)
            y = y.to(device, non_blocking=is_cuda)
            with (torch.cuda.amp.autocast() if is_cuda else contextlib.nullcontext()):
                logits = forward_logits(x)
            
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(y.detach().cpu().numpy())
            test_probs.extend(probs[:, 1].detach().cpu().numpy())
    
    # Calculate test metrics
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    test_f1_seizure = f1_score(test_labels, test_predictions, average=None)[1] if 1 in test_labels else 0
    test_auroc = roc_auc_score(test_labels, test_probs) if 1 in test_labels else 0
    test_auprc = average_precision_score(test_labels, test_probs) if 1 in test_labels else 0
    
    print(f'\n=== FINAL TEST RESULTS ===')
    print(f'F1-Score (Weighted): {test_f1:.4f}')
    print(f'F1-Score (Seizure Class): {test_f1_seizure:.4f}')
    print(f'AUROC: {test_auroc:.4f}')
    print(f'AUPRC: {test_auprc:.4f}')
    
    # Print detailed metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Non-Seizure', 'Seizure']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    
    return encoder, stgnn, explainer

def main():
    set_determinism(DEFAULT_SEED)
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data info and file paths
    data_files, label_files, total_samples, class_counts = get_data_info()
    
    # Use smaller batch size for memory efficiency
    batch_size = 8 if device.type == 'cpu' else 16
    
    # Choose loss function (set to True to use Focal Loss)
    use_focal_loss = False
    
    # Train model
    encoder, stgnn, explainer = train_model(
        data_files, label_files, class_counts,
        device=device, 
        epochs=50, 
        batch_size=batch_size,
        use_focal_loss=use_focal_loss,
        seed=DEFAULT_SEED
    )
    
    print("\nTraining completed successfully!")
    print("Best model saved as 'best_model.pth'")

if __name__ == "__main__":
    import sys
    # If 'small' is passed as an argument, use the small dataset
    if len(sys.argv) > 1 and sys.argv[1] == 'small':
        data_files = ['prepared_data/small_windows.npy']
        label_files = ['prepared_data/small_labels.npy']
        # Count classes
        labels = np.load(label_files[0])
        class_counts = {0: int((labels==0).sum()), 1: int((labels==1).sum())}
        print(f"Using small dataset: {class_counts}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_determinism(DEFAULT_SEED)
        encoder, stgnn, explainer = train_model(
            data_files, label_files, class_counts,
            device=device,
            epochs=20,
            batch_size=8,
            use_focal_loss=False,
            seed=DEFAULT_SEED
        )
        print("\nSmall dataset training completed!")
    else:
        main() 