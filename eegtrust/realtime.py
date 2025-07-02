#!/usr/bin/env python3
"""
Real-Time Seizure Detection System
- <1 second latency
- Live EEG streaming with circular buffer
- Sliding window inference
- Real-time alerting with confidence thresholds
- Asynchronous explanation generation
- Comprehensive logging
"""

import numpy as np
import torch
import torch.nn as nn
import time
import threading
import queue
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import deque
import logging
from dataclasses import dataclass
import asyncio
import aiofiles

# Import our trained models
from .model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from .config import SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES

@dataclass
class SeizureAlert:
    """Data structure for seizure alerts"""
    timestamp: float
    confidence: float
    eeg_window: np.ndarray
    explanation: str
    alert_id: str

@dataclass
class ModelPrediction:
    """Data structure for model predictions"""
    timestamp: float
    confidence: float
    prediction: int  # 0: non-seizure, 1: seizure
    eeg_window: np.ndarray
    features: Optional[np.ndarray] = None

class CircularBuffer:
    """Circular buffer for real-time EEG data"""
    
    def __init__(self, max_samples: int, num_channels: int = 23):
        self.max_samples = max_samples
        self.num_channels = num_channels
        self.buffer = np.zeros((num_channels, max_samples), dtype=np.float32)
        self.head = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def add_samples(self, new_samples: np.ndarray):
        """Add new EEG samples to the buffer"""
        with self.lock:
            num_new = new_samples.shape[1]
            
            # Handle wraparound
            if self.head + num_new <= self.max_samples:
                # No wraparound needed
                self.buffer[:, self.head:self.head + num_new] = new_samples
            else:
                # Wraparound needed
                first_part = self.max_samples - self.head
                second_part = num_new - first_part
                
                self.buffer[:, self.head:] = new_samples[:, :first_part]
                self.buffer[:, :second_part] = new_samples[:, first_part:]
            
            # Update head and size
            self.head = (self.head + num_new) % self.max_samples
            self.size = min(self.size + num_new, self.max_samples)
    
    def get_latest_window(self, window_samples: int) -> Optional[np.ndarray]:
        """Get the latest window of EEG data"""
        with self.lock:
            if self.size < window_samples:
                return None
            
            # Calculate start index
            start_idx = (self.head - window_samples) % self.max_samples
            
            if start_idx < self.head:
                # No wraparound
                return self.buffer[:, start_idx:self.head].copy()
            else:
                # Wraparound
                first_part = self.max_samples - start_idx
                second_part = self.head
                
                window = np.zeros((self.num_channels, window_samples), dtype=np.float32)
                window[:, :first_part] = self.buffer[:, start_idx:]
                window[:, first_part:] = self.buffer[:, :second_part]
                
                return window

class EEGStreamSimulator:
    """Simulates real-time EEG stream from CHB-MIT data"""
    
    def __init__(self, data_file: str, chunk_duration: float = 0.25):
        """
        Args:
            data_file: Path to preprocessed EEG data (.npy file)
            chunk_duration: Duration of each chunk in seconds
        """
        self.data = np.load(data_file)
        self.chunk_samples = int(SAMPLE_RATE * chunk_duration)
        self.current_idx = 0
        self.lock = threading.Lock()
    
    def get_next_chunk(self) -> Optional[np.ndarray]:
        """Get the next chunk of EEG data"""
        with self.lock:
            if self.current_idx >= self.data.shape[1]:
                return None
            
            end_idx = min(self.current_idx + self.chunk_samples, self.data.shape[1])
            chunk = self.data[:, self.current_idx:end_idx]
            self.current_idx = end_idx
            
            return chunk
    
    def reset(self):
        """Reset the stream to the beginning"""
        with self.lock:
            self.current_idx = 0

class OptimizedSeizureDetector:
    """Optimized seizure detection model for real-time inference"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Pre-allocate tensors for efficiency (define before load_model)
        self.num_channels = 23
        self.window_samples = WINDOW_SIZE_SAMPLES
        self.hidden_dim = 128
        
        # Load the trained model
        self.encoder = None
        self.stgnn = None
        self.explainer = None
        self.load_model()
        
        # Set to evaluation mode
        self.encoder.eval()
        self.stgnn.eval()
        self.explainer.eval()
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
        # Warm up the model
        self._warmup()
    
    def load_model(self):
        """Load the trained model from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Initialize models
        self.encoder = SSLPretrainedEncoder(self.num_channels, self.window_samples, self.hidden_dim)
        self.stgnn = STGNN(self.hidden_dim, num_layers=2, num_heads=4)
        self.explainer = NeuroSymbolicExplainer(self.hidden_dim)
        
        # Load state dicts
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.stgnn.load_state_dict(checkpoint['stgnn_state_dict'])
        self.explainer.load_state_dict(checkpoint['explainer_state_dict'])
        
        # Move to device
        self.encoder = self.encoder.to(self.device)
        self.stgnn = self.stgnn.to(self.device)
        self.explainer = self.explainer.to(self.device)
    
    def _warmup(self):
        """Warm up the model with dummy data"""
        dummy_input = torch.randn(1, self.num_channels, self.window_samples).to(self.device)
        
        with torch.no_grad():
            for _ in range(3):  # Run a few warmup iterations
                features = self.encoder(dummy_input)
                features_seq = features.unsqueeze(1)
                stgnn_logits = self.stgnn(features_seq)
                explainer_logits = self.explainer(features)
                logits = (stgnn_logits + explainer_logits) / 2
    
    def preprocess_window(self, eeg_window: np.ndarray) -> torch.Tensor:
        """Preprocess EEG window for inference"""
        # Normalize (z-score per channel)
        mean = np.mean(eeg_window, axis=1, keepdims=True)
        std = np.std(eeg_window, axis=1, keepdims=True)
        normalized = (eeg_window - mean) / (std + 1e-8)
        
        # Convert to tensor
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, eeg_window: np.ndarray) -> ModelPrediction:
        """Run inference on EEG window"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_window(eeg_window)
        
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
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return ModelPrediction(
            timestamp=time.time(),
            confidence=seizure_prob,
            prediction=prediction,
            eeg_window=eeg_window.copy(),
            features=features.cpu().numpy()
        )
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000

class SeizureAlertManager:
    """Manages seizure alerts with confidence thresholds and voting"""
    
    def __init__(self, confidence_threshold: float = 0.85, voting_window: int = 3, min_votes: int = 2):
        self.confidence_threshold = confidence_threshold
        self.voting_window = voting_window
        self.min_votes = min_votes
        
        # Store recent predictions for voting
        self.recent_predictions = deque(maxlen=voting_window)
        
        # Alert tracking
        self.alerts = []
        self.alert_counter = 0
        
        # Performance stats
        self.total_predictions = 0
        self.positive_predictions = 0
        self.alerts_triggered = 0
    
    def process_prediction(self, prediction: ModelPrediction) -> Optional[SeizureAlert]:
        """Process a new prediction and potentially trigger an alert"""
        self.total_predictions += 1
        
        # Add to recent predictions
        self.recent_predictions.append(prediction)
        
        # Check if we have enough predictions for voting
        if len(self.recent_predictions) < self.voting_window:
            return None
        
        # Count positive predictions in voting window
        positive_votes = sum(1 for p in self.recent_predictions 
                           if p.confidence > self.confidence_threshold)
        
        # Check if alert should be triggered
        if positive_votes >= self.min_votes:
            # Create alert
            alert = SeizureAlert(
                timestamp=prediction.timestamp,
                confidence=prediction.confidence,
                eeg_window=prediction.eeg_window,
                explanation="",  # Will be filled by explanation generator
                alert_id=f"alert_{self.alert_counter:06d}"
            )
            
            self.alerts.append(alert)
            self.alert_counter += 1
            self.alerts_triggered += 1
            
            return alert
        
        return None
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_predictions': self.total_predictions,
            'positive_predictions': self.positive_predictions,
            'alerts_triggered': self.alerts_triggered,
            'alert_rate': self.alerts_triggered / max(self.total_predictions, 1)
        }

class ExplanationGenerator:
    """Generates natural language explanations for seizure predictions"""
    
    def __init__(self, model_detector: OptimizedSeizureDetector):
        self.model_detector = model_detector
        self.explanation_queue = queue.Queue()
        self.explanation_thread = None
        self.running = False
    
    def start(self):
        """Start the explanation generation thread"""
        self.running = True
        self.explanation_thread = threading.Thread(target=self._explanation_worker)
        self.explanation_thread.daemon = True
        self.explanation_thread.start()
    
    def stop(self):
        """Stop the explanation generation thread"""
        self.running = False
        if self.explanation_thread:
            self.explanation_thread.join()
    
    def generate_explanation(self, alert: SeizureAlert) -> str:
        """Generate explanation for a seizure alert"""
        # Extract key features
        eeg_window = alert.eeg_window
        
        # Calculate basic EEG features
        features = self._extract_eeg_features(eeg_window)
        
        # Generate explanation based on features
        explanation = self._create_explanation(features, alert.confidence)
        
        return explanation
    
    def _extract_eeg_features(self, eeg_window: np.ndarray) -> Dict:
        """Extract relevant EEG features for explanation"""
        features = {}
        
        # Calculate power in different frequency bands
        from scipy import signal
        
        for ch in range(min(5, eeg_window.shape[0])):  # Use first 5 channels
            # FFT
            fft_vals = np.abs(np.fft.fft(eeg_window[ch]))
            freqs = np.fft.fftfreq(len(eeg_window[ch]), 1/SAMPLE_RATE)
            
            # Power in different bands
            delta_mask = (freqs >= 0.5) & (freqs <= 4)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            gamma_mask = (freqs >= 30) & (freqs <= 70)
            
            features[f'ch{ch}_delta_power'] = np.mean(fft_vals[delta_mask])
            features[f'ch{ch}_theta_power'] = np.mean(fft_vals[theta_mask])
            features[f'ch{ch}_alpha_power'] = np.mean(fft_vals[alpha_mask])
            features[f'ch{ch}_beta_power'] = np.mean(fft_vals[beta_mask])
            features[f'ch{ch}_gamma_power'] = np.mean(fft_vals[gamma_mask])
        
        # Overall statistics
        features['mean_amplitude'] = np.mean(np.abs(eeg_window))
        features['std_amplitude'] = np.std(eeg_window)
        features['peak_to_peak'] = np.max(eeg_window) - np.min(eeg_window)
        
        return features
    
    def _create_explanation(self, features: Dict, confidence: float) -> str:
        """Create natural language explanation"""
        # Find dominant frequency bands
        band_powers = {}
        for ch in range(5):
            delta = features.get(f'ch{ch}_delta_power', 0)
            theta = features.get(f'ch{ch}_theta_power', 0)
            alpha = features.get(f'ch{ch}_alpha_power', 0)
            beta = features.get(f'ch{ch}_beta_power', 0)
            gamma = features.get(f'ch{ch}_gamma_power', 0)
            
            band_powers[f'ch{ch}'] = {
                'delta': delta, 'theta': theta, 'alpha': alpha,
                'beta': beta, 'gamma': gamma
            }
        
        # Find channels with highest activity
        total_powers = {ch: sum(bands.values()) for ch, bands in band_powers.items()}
        most_active_ch = max(total_powers.items(), key=lambda x: x[1])[0]
        
        # Create explanation
        explanation = f"Seizure detected with {confidence:.1%} confidence. "
        explanation += f"Analysis of the 2-second EEG window shows: "
        
        # Add frequency band information
        dominant_bands = []
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            avg_power = np.mean([bands[band] for bands in band_powers.values()])
            if avg_power > np.mean(list(band_powers[most_active_ch].values())):
                dominant_bands.append(band)
        
        if dominant_bands:
            explanation += f"Elevated activity in {', '.join(dominant_bands)} frequency bands. "
        
        explanation += f"Channel {most_active_ch} shows the highest activity. "
        explanation += f"Overall signal amplitude is {features['mean_amplitude']:.2f} μV. "
        
        if confidence > 0.95:
            explanation += "This represents a high-confidence seizure detection."
        elif confidence > 0.85:
            explanation += "This represents a moderate-confidence seizure detection."
        else:
            explanation += "This represents a low-confidence seizure detection."
        
        return explanation
    
    def _explanation_worker(self):
        """Background worker for generating explanations"""
        while self.running:
            try:
                # Get alert from queue (non-blocking)
                alert = self.explanation_queue.get(timeout=1.0)
                
                # Generate explanation
                explanation = self.generate_explanation(alert)
                
                # Update alert with explanation
                alert.explanation = explanation
                
                # Log the explanation
                logging.info(f"Generated explanation for {alert.alert_id}: {explanation}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error generating explanation: {e}")
    
    def queue_explanation(self, alert: SeizureAlert):
        """Queue an alert for explanation generation"""
        self.explanation_queue.put(alert)

class RealTimeSeizureDetector:
    """Main real-time seizure detection system"""
    
    def __init__(self, model_path: str, data_file: str, output_dir: str = "realtime_output"):
        self.model_path = model_path
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'realtime.log')),
                logging.StreamHandler()
            ]
        )
        
        # Initialize components
        self.eeg_stream = EEGStreamSimulator(data_file)
        self.circular_buffer = CircularBuffer(max_samples=int(3 * SAMPLE_RATE))  # 3 seconds
        self.detector = OptimizedSeizureDetector(model_path)
        self.alert_manager = SeizureAlertManager()
        self.explanation_generator = ExplanationGenerator(self.detector)
        
        # Control flags
        self.running = False
        self.processing_thread = None
        
        # Performance tracking
        self.start_time = None
        self.windows_processed = 0
        
        # Start explanation generator
        self.explanation_generator.start()
        
        logging.info("Real-time seizure detection system initialized")
    
    def start(self):
        """Start real-time detection"""
        self.running = True
        self.start_time = time.time()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logging.info("Real-time seizure detection started")
    
    def stop(self):
        """Stop real-time detection"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        self.explanation_generator.stop()
        
        # Save final statistics
        self._save_final_stats()
        
        logging.info("Real-time seizure detection stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        last_window_time = 0
        window_interval = 0.25  # Process window every 0.25 seconds
        
        while self.running:
            current_time = time.time()
            
            # Get new EEG chunk
            chunk = self.eeg_stream.get_next_chunk()
            if chunk is None:
                # End of data, restart
                self.eeg_stream.reset()
                continue
            
            # Add to circular buffer
            self.circular_buffer.add_samples(chunk)
            
            # Check if it's time to process a new window
            if current_time - last_window_time >= window_interval:
                # Get latest window
                window = self.circular_buffer.get_latest_window(WINDOW_SIZE_SAMPLES)
                
                if window is not None:
                    # Run prediction
                    prediction = self.detector.predict(window)
                    self.windows_processed += 1
                    
                    # Process with alert manager
                    alert = self.alert_manager.process_prediction(prediction)
                    
                    if alert:
                        # Log alert
                        logging.warning(f"SEIZURE ALERT: {alert.alert_id} - Confidence: {alert.confidence:.3f}")
                        
                        # Queue for explanation
                        self.explanation_generator.queue_explanation(alert)
                        
                        # Save alert data
                        self._save_alert(alert)
                    
                    # Log performance every 100 windows
                    if self.windows_processed % 100 == 0:
                        self._log_performance()
                
                last_window_time = current_time
            
            # Small sleep to prevent busy waiting
            time.sleep(0.01)
    
    def _save_alert(self, alert: SeizureAlert):
        """Save alert data to file"""
        alert_data = {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp,
            'confidence': alert.confidence,
            'explanation': alert.explanation
        }
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, f"{alert.alert_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        # Save EEG data
        eeg_file = os.path.join(self.output_dir, f"{alert.alert_id}_eeg.npy")
        np.save(eeg_file, alert.eeg_window)
        
        logging.info(f"Saved alert data: {alert.alert_id}")
    
    def _log_performance(self):
        """Log performance statistics"""
        elapsed_time = time.time() - self.start_time
        windows_per_second = self.windows_processed / elapsed_time
        avg_inference_time = self.detector.get_average_inference_time()
        
        stats = self.alert_manager.get_stats()
        
        logging.info(f"Performance: {windows_per_second:.2f} windows/sec, "
                    f"avg inference: {avg_inference_time:.2f}ms, "
                    f"alerts: {stats['alerts_triggered']}")
    
    def _save_final_stats(self):
        """Save final statistics"""
        final_stats = {
            'total_runtime': time.time() - self.start_time,
            'windows_processed': self.windows_processed,
            'avg_inference_time_ms': self.detector.get_average_inference_time(),
            'alert_manager_stats': self.alert_manager.get_stats(),
            'alerts': [
                {
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp,
                    'confidence': alert.confidence
                }
                for alert in self.alert_manager.alerts
            ]
        }
        
        stats_file = os.path.join(self.output_dir, 'final_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logging.info(f"Final statistics saved to {stats_file}")

def main():
    """Main function to run real-time seizure detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time seizure detection')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth)')
    parser.add_argument('--data', required=True, help='Path to EEG data file (.npy)')
    parser.add_argument('--output', default='realtime_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create and start detector
    detector = RealTimeSeizureDetector(args.model, args.data, args.output)
    
    try:
        detector.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping real-time detection...")
        detector.stop()
        print("Detection stopped. Check the output directory for results.")

if __name__ == "__main__":
    main() 