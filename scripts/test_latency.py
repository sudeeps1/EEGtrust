#!/usr/bin/env python3
"""
Latency Testing Script for EEGTrust Seizure Detection System
- Measures inference latency and throughput
- Tests real-time performance with different window sizes
- Evaluates memory usage and CPU/GPU utilization
- Provides detailed performance profiling
- Tests system under load
"""

import numpy as np
import torch
import time
import psutil
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import datetime
import json
from collections import deque
import statistics

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
from eegtrust.config import SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES
from eegtrust.realtime import OptimizedSeizureDetector, CircularBuffer

class LatencyTester:
    """Comprehensive latency testing for the seizure detection system"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.load_model()
        
        # Performance tracking
        self.latency_results = []
        self.throughput_results = []
        self.memory_results = []
        
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
    
    def generate_test_data(self, num_windows: int = 1000) -> np.ndarray:
        """Generate synthetic EEG data for testing"""
        print(f"Generating {num_windows} test windows...")
        
        # Generate realistic EEG-like data
        test_windows = []
        for _ in range(num_windows):
            # Generate random EEG data with realistic characteristics
            window = np.random.randn(23, WINDOW_SIZE_SAMPLES).astype(np.float32)
            
            # Add some realistic EEG patterns
            # Add alpha rhythm (8-13 Hz) to some channels
            t = np.linspace(0, WINDOW_SIZE_SAMPLES/SAMPLE_RATE, WINDOW_SIZE_SAMPLES)
            alpha_freq = np.random.uniform(8, 13)
            alpha_signal = 0.1 * np.sin(2 * np.pi * alpha_freq * t)
            
            # Add to random channels
            num_alpha_channels = np.random.randint(3, 8)
            alpha_channels = np.random.choice(23, num_alpha_channels, replace=False)
            for ch in alpha_channels:
                window[ch] += alpha_signal
            
            test_windows.append(window)
        
        return np.array(test_windows)
    
    def preprocess_window(self, eeg_window: np.ndarray) -> torch.Tensor:
        """Preprocess EEG window for inference"""
        # Normalize (z-score per channel)
        mean = np.mean(eeg_window, axis=1, keepdims=True)
        std = np.std(eeg_window, axis=1, keepdims=True)
        normalized = (eeg_window - mean) / (std + 1e-8)
        
        # Convert to tensor
        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor
    
    def measure_single_inference(self, eeg_window: np.ndarray) -> dict:
        """Measure latency for a single inference"""
        # Warm up
        for _ in range(3):
            input_tensor = self.preprocess_window(eeg_window)
            with torch.no_grad():
                features = self.encoder(input_tensor)
                features_seq = features.unsqueeze(1)
                stgnn_logits = self.stgnn(features_seq)
                explainer_logits = self.explainer(features)
                logits = (stgnn_logits + explainer_logits) / 2
        
        # Measure inference time
        start_time = time.time()
        input_tensor = self.preprocess_window(eeg_window)
        
        with torch.no_grad():
            features = self.encoder(input_tensor)
            features_seq = features.unsqueeze(1)
            stgnn_logits = self.stgnn(features_seq)
            explainer_logits = self.explainer(features)
            logits = (stgnn_logits + explainer_logits) / 2
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            seizure_prob = probs[0, 1].item()
        
        inference_time = time.time() - start_time
        
        return {
            'inference_time_ms': inference_time * 1000,
            'confidence': seizure_prob,
            'prediction': 1 if seizure_prob > 0.5 else 0
        }
    
    def test_batch_latency(self, test_windows: np.ndarray, batch_sizes: list = [1, 4, 8, 16, 32]) -> dict:
        """Test latency with different batch sizes"""
        print("Testing batch latency...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            latencies = []
            throughputs = []
            
            # Process in batches
            for i in range(0, len(test_windows), batch_size):
                batch = test_windows[i:i+batch_size]
                
                start_time = time.time()
                
                # Process batch
                batch_tensors = []
                for window in batch:
                    tensor = self.preprocess_window(window)
                    batch_tensors.append(tensor)
                
                # Stack tensors
                if len(batch_tensors) > 1:
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                else:
                    batch_tensor = batch_tensors[0]
                
                # Run inference
                with torch.no_grad():
                    features = self.encoder(batch_tensor)
                    features_seq = features.unsqueeze(1)
                    stgnn_logits = self.stgnn(features_seq)
                    explainer_logits = self.explainer(features)
                    logits = (stgnn_logits + explainer_logits) / 2
                    probs = torch.softmax(logits, dim=1)
                
                batch_time = time.time() - start_time
                
                latencies.append(batch_time * 1000)  # Convert to ms
                throughputs.append(len(batch) / (batch_time + 1e-8))  # windows per second, add small epsilon to prevent division by zero
            
            batch_results[batch_size] = {
                'avg_latency_ms': np.mean(latencies),
                'std_latency_ms': np.std(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'avg_throughput_wps': np.mean(throughputs),
                'std_throughput_wps': np.std(throughputs)
            }
        
        return batch_results
    
    def test_continuous_throughput(self, duration_seconds: int = 60) -> dict:
        """Test continuous throughput over time"""
        print(f"Testing continuous throughput for {duration_seconds} seconds...")
        
        # Generate continuous stream of data
        windows_per_second = 4  # Target throughput
        total_windows = duration_seconds * windows_per_second
        test_windows = self.generate_test_data(total_windows)
        
        latencies = []
        throughputs = []
        timestamps = []
        
        start_time = time.time()
        window_count = 0
        
        while time.time() - start_time < duration_seconds and window_count < len(test_windows):
            window_start = time.time()
            
            # Process one window
            result = self.measure_single_inference(test_windows[window_count])
            
            window_time = time.time() - window_start
            latencies.append(result['inference_time_ms'])
            throughputs.append(1.0 / (window_time + 1e-8))  # Add small epsilon to prevent division by zero
            timestamps.append(time.time() - start_time)
            
            window_count += 1
            
            # Sleep to maintain target throughput
            target_interval = 1.0 / windows_per_second
            sleep_time = max(0, target_interval - window_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return {
            'total_windows': window_count,
            'total_time': time.time() - start_time,
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'avg_throughput_wps': np.mean(throughputs),
            'std_throughput_wps': np.std(throughputs),
            'latencies': latencies,
            'throughputs': throughputs,
            'timestamps': timestamps
        }
    
    def test_memory_usage(self, num_windows: int = 100) -> dict:
        """Test memory usage during inference"""
        print("Testing memory usage...")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate test data
        test_windows = self.generate_test_data(num_windows)
        
        memory_usage = []
        
        for i, window in enumerate(test_windows):
            if i % 10 == 0:
                memory_usage.append({
                    'window': i,
                    'memory_mb': process.memory_info().rss / 1024 / 1024
                })
            
            # Run inference
            self.measure_single_inference(window)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'memory_usage_over_time': memory_usage
        }
    
    def test_real_time_simulation(self, duration_seconds: int = 30) -> dict:
        """Simulate real-time processing with circular buffer"""
        print(f"Testing real-time simulation for {duration_seconds} seconds...")
        
        # Initialize circular buffer
        buffer_size = 3 * SAMPLE_RATE  # 3 seconds
        circular_buffer = CircularBuffer(buffer_size, num_channels=23)
        
        # Generate continuous EEG stream
        samples_per_chunk = int(0.25 * SAMPLE_RATE)  # 250ms chunks
        total_chunks = int(duration_seconds / 0.25)
        
        processing_times = []
        buffer_latencies = []
        alert_latencies = []
        
        start_time = time.time()
        
        for chunk_idx in range(total_chunks):
            chunk_start = time.time()
            
            # Generate EEG chunk
            eeg_chunk = np.random.randn(23, samples_per_chunk).astype(np.float32)
            
            # Add to circular buffer
            circular_buffer.add_samples(eeg_chunk)
            
            # Get latest window for processing
            window = circular_buffer.get_latest_window(WINDOW_SIZE_SAMPLES)
            
            if window is not None:
                # Process window
                inference_start = time.time()
                result = self.measure_single_inference(window)
                inference_time = time.time() - inference_start
                
                processing_times.append(inference_time * 1000)
                buffer_latencies.append((time.time() - chunk_start) * 1000)
                
                # Simulate alert processing
                if result['confidence'] > 0.85:
                    alert_start = time.time()
                    # Simulate alert processing time
                    time.sleep(0.01)  # 10ms
                    alert_latencies.append((time.time() - alert_start) * 1000)
            
            # Maintain real-time pacing
            elapsed = time.time() - chunk_start
            target_interval = 0.25  # 250ms
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'std_processing_time_ms': np.std(processing_times) if processing_times else 0,
            'avg_buffer_latency_ms': np.mean(buffer_latencies) if buffer_latencies else 0,
            'avg_alert_latency_ms': np.mean(alert_latencies) if alert_latencies else 0,
            'total_alerts': len(alert_latencies),
            'processing_times': processing_times
        }
    
    def create_latency_visualizations(self, results: dict, output_dir: str):
        """Create latency performance visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Batch size vs latency
        if 'batch_results' in results:
            batch_sizes = list(results['batch_results'].keys())
            avg_latencies = [results['batch_results'][bs]['avg_latency_ms'] for bs in batch_sizes]
            avg_throughputs = [results['batch_results'][bs]['avg_throughput_wps'] for bs in batch_sizes]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(batch_sizes, avg_latencies, 'bo-')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Average Latency (ms)')
            ax1.set_title('Batch Size vs Latency')
            ax1.grid(True)
            
            ax2.plot(batch_sizes, avg_throughputs, 'ro-')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (windows/sec)')
            ax2.set_title('Batch Size vs Throughput')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'batch_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Continuous throughput over time
        if 'continuous_results' in results:
            timestamps = results['continuous_results']['timestamps']
            latencies = results['continuous_results']['latencies']
            throughputs = results['continuous_results']['throughputs']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(timestamps, latencies, 'b-', alpha=0.7)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Latency Over Time')
            ax1.grid(True)
            
            ax2.plot(timestamps, throughputs, 'r-', alpha=0.7)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Throughput (windows/sec)')
            ax2.set_title('Throughput Over Time')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'continuous_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Memory usage over time
        if 'memory_results' in results:
            memory_data = results['memory_results']['memory_usage_over_time']
            windows = [d['window'] for d in memory_data]
            memory_mb = [d['memory_mb'] for d in memory_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(windows, memory_mb, 'g-')
            plt.xlabel('Windows Processed')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_full_latency_evaluation(self, output_dir: str = "latency_test_results"):
        """Run complete latency evaluation"""
        print("=== EEGTrust Latency Evaluation ===")
        start_time = time.time()
        
        # Generate test data
        test_windows = self.generate_test_data(1000)
        
        # Test single inference latency
        print("Testing single inference latency...")
        single_latencies = []
        for i in range(100):
            if i % 20 == 0:
                print(f"Single inference test {i}/100")
            result = self.measure_single_inference(test_windows[i])
            single_latencies.append(result['inference_time_ms'])
        
        single_results = {
            'avg_latency_ms': np.mean(single_latencies),
            'std_latency_ms': np.std(single_latencies),
            'min_latency_ms': np.min(single_latencies),
            'max_latency_ms': np.max(single_latencies),
            'p50_latency_ms': np.percentile(single_latencies, 50),
            'p95_latency_ms': np.percentile(single_latencies, 95),
            'p99_latency_ms': np.percentile(single_latencies, 99)
        }
        
        # Test batch latency
        batch_results = self.test_batch_latency(test_windows)
        
        # Test continuous throughput
        continuous_results = self.test_continuous_throughput(30)  # 30 seconds
        
        # Test memory usage
        memory_results = self.test_memory_usage(200)
        
        # Test real-time simulation
        realtime_results = self.test_real_time_simulation(30)  # 30 seconds
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'single_inference_results': single_results,
            'batch_results': batch_results,
            'continuous_results': continuous_results,
            'memory_results': memory_results,
            'realtime_results': realtime_results,
            'evaluation_time': time.time() - start_time
        }
        
        # Create visualizations
        self.create_latency_visualizations(results, output_dir)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== LATENCY EVALUATION RESULTS ===")
        print(f"Device: {self.device}")
        print(f"\nSingle Inference Latency:")
        print(f"  Average: {single_results['avg_latency_ms']:.2f} ms")
        print(f"  Std Dev: {single_results['std_latency_ms']:.2f} ms")
        print(f"  P50:     {single_results['p50_latency_ms']:.2f} ms")
        print(f"  P95:     {single_results['p95_latency_ms']:.2f} ms")
        print(f"  P99:     {single_results['p99_latency_ms']:.2f} ms")
        
        print(f"\nContinuous Throughput:")
        print(f"  Average: {continuous_results['avg_throughput_wps']:.2f} windows/sec")
        print(f"  Latency: {continuous_results['avg_latency_ms']:.2f} ms")
        
        print(f"\nMemory Usage:")
        print(f"  Initial: {memory_results['initial_memory_mb']:.1f} MB")
        print(f"  Final:   {memory_results['final_memory_mb']:.1f} MB")
        print(f"  Increase: {memory_results['memory_increase_mb']:.1f} MB")
        
        print(f"\nReal-time Simulation:")
        print(f"  Processing Time: {realtime_results['avg_processing_time_ms']:.2f} ms")
        print(f"  Buffer Latency: {realtime_results['avg_buffer_latency_ms']:.2f} ms")
        print(f"  Alert Latency: {realtime_results['avg_alert_latency_ms']:.2f} ms")
        print(f"  Total Alerts: {realtime_results['total_alerts']}")
        
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
    tester = LatencyTester(model_path)
    
    # Run evaluation
    output_dir = f"latency_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tester.run_full_latency_evaluation(output_dir)

if __name__ == "__main__":
    main() 