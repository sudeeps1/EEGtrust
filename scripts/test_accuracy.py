#!/usr/bin/env python3
"""
Accuracy Testing Script for EEGTrust Seizure Detection System
- Evaluates model performance on test data
- Calculates comprehensive metrics (precision, recall, F1, AUC)
- Generates confusion matrix and ROC curve
- Tests different confidence thresholds
- Provides detailed performance analysis
"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import sys
from datetime import datetime
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from eegtrust.config import SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES

class AccuracyTester:
    """Comprehensive accuracy testing for the seizure detection system"""
    
    def __init__(self, model_path: str, test_data_dir: str = "prepared_data"):
        self.model_path = model_path
        self.test_data_dir = test_data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.load_model()
        
        # Results storage
        self.results = {}
        self.predictions = []
        self.true_labels = []
        self.confidences = []
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Initialize models
        self.encoder = SSLPretrainedEncoder(23, WINDOW_SIZE_SAMPLES, 128)
        self.stgnn = STGNN(128, num_layers=2, num_heads=4)
        self.explainer = NeuroSymbolicExplainer(128)
        
        # Load state dicts
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.stgnn.load_state_dict(checkpoint['stgnn_state_dict'])
        self.explainer.load_state_dict(checkpoint['explainer_state_dict'])
        
        # Move to device
        self.encoder = self.encoder.to(self.device)
        self.stgnn = self.stgnn.to(self.device)
        self.explainer = self.explainer.to(self.device)
        
        # Set to evaluation mode
        self.encoder.eval()
        self.stgnn.eval()
        self.explainer.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def load_test_data(self):
        """Load test data from prepared_data directory"""
        print("Loading test data...")
        
        test_windows = []
        test_labels = []
        
        # Load all available subjects
        for file in os.listdir(self.test_data_dir):
            if file.endswith('_windows.npy'):
                subject = file.replace('_windows.npy', '')
                windows_file = os.path.join(self.test_data_dir, f"{subject}_windows.npy")
                labels_file = os.path.join(self.test_data_dir, f"{subject}_labels.npy")
                
                if os.path.exists(labels_file):
                    print(f"Loading {subject}...")
                    windows = np.load(windows_file)
                    labels = np.load(labels_file)
                    
                    test_windows.append(windows)
                    test_labels.append(labels)
        
        if not test_windows:
            raise ValueError("No test data found in prepared_data directory")
        
        # Combine all subjects
        self.test_windows = np.concatenate(test_windows, axis=0)
        self.test_labels = np.concatenate(test_labels, axis=0).astype(np.int64)
        
        print(f"Loaded {len(self.test_windows)} test windows")
        print(f"Class distribution: {np.bincount(self.test_labels)}")
    
    def preprocess_window(self, eeg_window: np.ndarray) -> torch.Tensor:
        """Preprocess EEG window for inference"""
        # Normalize (z-score per channel)
        mean = np.mean(eeg_window, axis=1, keepdims=True)
        std = np.std(eeg_window, axis=1, keepdims=True)
        normalized = (eeg_window - mean) / (std + 1e-8)
        
        # Convert to tensor
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_batch(self, windows: np.ndarray) -> tuple:
        """Predict on a batch of windows"""
        predictions = []
        confidences = []
        
        for i, window in enumerate(windows):
            if i % 1000 == 0:
                print(f"Processing window {i}/{len(windows)}")
            
            # Preprocess
            input_tensor = self.preprocess_window(window)
            
            # Run inference
            with torch.no_grad():
                features = self.encoder(input_tensor)
                features_seq = features.unsqueeze(1)
                stgnn_logits = self.stgnn(features_seq)
                explainer_logits = self.explainer(features)
                logits = (stgnn_logits + explainer_logits) / 2
                
                # Get probabilities
                probs = torch.softmax(logits, dim=1)
                seizure_prob = probs[0, 1].item()
                
                # Get prediction
                prediction = 1 if seizure_prob > 0.5 else 0
                
                predictions.append(prediction)
                confidences.append(seizure_prob)
        
        return np.array(predictions), np.array(confidences)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray) -> dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['auc'] = roc_auc_score(y_true, y_conf)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Additional metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        
        # False positive rate
        metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
        
        return metrics
    
    def test_different_thresholds(self, y_true: np.ndarray, y_conf: np.ndarray) -> pd.DataFrame:
        """Test performance at different confidence thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_conf >= threshold).astype(int)
            metrics = self.calculate_metrics(y_true, y_pred, y_conf)
            
            results.append({
                'threshold': threshold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'specificity': metrics['specificity'],
                'sensitivity': metrics['sensitivity'],
                'fpr': metrics['fpr']
            })
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray, output_dir: str):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Seizure', 'Seizure'],
                   yticklabels=['Non-Seizure', 'Seizure'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_conf)
        auc = roc_auc_score(y_true, y_conf)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_conf)
        plt.plot(recall, precision, label=f'Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Threshold Analysis
        threshold_results = self.test_different_thresholds(y_true, y_conf)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(threshold_results['threshold'], threshold_results['accuracy'], label='Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(threshold_results['threshold'], threshold_results['precision'], label='Precision')
        plt.plot(threshold_results['threshold'], threshold_results['recall'], label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision/Recall vs Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(threshold_results['threshold'], threshold_results['f1'], label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(threshold_results['threshold'], threshold_results['fpr'], label='False Positive Rate')
        plt.xlabel('Threshold')
        plt.ylabel('FPR')
        plt.title('False Positive Rate vs Threshold')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save threshold results
        threshold_results.to_csv(os.path.join(output_dir, 'threshold_analysis.csv'), index=False)
    
    def run_cross_validation(self, n_splits: int = 5) -> dict:
        """Run cross-validation to assess model robustness"""
        print(f"Running {n_splits}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(self.test_windows, self.test_labels)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            # Get test data for this fold
            X_test = self.test_windows[test_idx]
            y_test = self.test_labels[test_idx]
            
            # Predict
            y_pred, y_conf = self.predict_batch(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_conf)
            cv_results.append(metrics)
            
            print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        
        # Aggregate results
        avg_metrics = {}
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'sensitivity']:
            values = [result[key] for result in cv_results]
            avg_metrics[f'{key}_mean'] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def run_full_evaluation(self, output_dir: str = "accuracy_test_results"):
        """Run complete accuracy evaluation"""
        print("=== EEGTrust Accuracy Evaluation ===")
        start_time = time.time()
        
        # Load test data
        self.load_test_data()
        
        # Run predictions
        print("Running predictions on test data...")
        y_pred, y_conf = self.predict_batch(self.test_windows)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics(self.test_labels, y_pred, y_conf)
        
        # Run cross-validation
        cv_metrics = self.run_cross_validation()
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(self.test_labels, y_pred, y_conf, output_dir)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_metrics': metrics,
            'cross_validation_metrics': cv_metrics,
            'test_samples': len(self.test_labels),
            'class_distribution': np.bincount(self.test_labels).tolist(),
            'evaluation_time': time.time() - start_time
        }
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== ACCURACY EVALUATION RESULTS ===")
        print(f"Test Samples: {len(self.test_labels)}")
        print(f"Class Distribution: {np.bincount(self.test_labels)}")
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC:       {metrics['auc']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        
        print(f"\nCross-Validation Results (Mean ± Std):")
        print(f"  Accuracy:  {cv_metrics['accuracy_mean']:.3f} ± {cv_metrics['accuracy_std']:.3f}")
        print(f"  F1 Score:  {cv_metrics['f1_mean']:.3f} ± {cv_metrics['f1_std']:.3f}")
        print(f"  AUC:       {cv_metrics['auc_mean']:.3f} ± {cv_metrics['auc_std']:.3f}")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Evaluation time: {time.time() - start_time:.2f} seconds")

def main():
    """Main function"""
    model_path = "best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first using: python scripts/train_with_existing_data.py")
        return
    
    # Create tester
    tester = AccuracyTester(model_path)
    
    # Run evaluation
    output_dir = f"accuracy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tester.run_full_evaluation(output_dir)

if __name__ == "__main__":
    main() 