#!/usr/bin/env python3
"""
Real-time dashboard for seizure detection monitoring
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
import sys

# Add the parent directory to the path to import eegtrust modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RealTimeDashboard:
    """Real-time dashboard for seizure detection monitoring"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.running = True
        
        # Data storage
        self.timestamps = []
        self.confidences = []
        self.predictions = []
        self.alert_times = []
        self.alert_confidences = []
        
        # Performance metrics
        self.windows_processed = 0
        self.avg_inference_time = 0
        self.alerts_triggered = 0
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Setup matplotlib
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-Time Seizure Detection Dashboard', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_output)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _setup_plots(self):
        """Setup the dashboard plots"""
        # Plot 1: Real-time confidence
        self.axes[0, 0].set_title('Seizure Detection Confidence')
        self.axes[0, 0].set_ylabel('Confidence')
        self.axes[0, 0].set_ylim(0, 1)
        self.axes[0, 0].grid(True)
        self.confidence_line, = self.axes[0, 0].plot([], [], 'b-', label='Confidence')
        self.threshold_line, = self.axes[0, 0].plot([], [], 'r--', label='Alert Threshold (0.85)')
        self.axes[0, 0].legend()
        
        # Plot 2: Alert history
        self.axes[0, 1].set_title('Seizure Alerts')
        self.axes[0, 1].set_ylabel('Confidence')
        self.axes[0, 1].set_ylim(0, 1)
        self.axes[0, 1].grid(True)
        self.alert_scatter = self.axes[0, 1].scatter([], [], c='red', s=100, alpha=0.7)
        
        # Plot 3: Performance metrics
        self.axes[1, 0].set_title('Performance Metrics')
        self.axes[1, 0].axis('off')
        self.perf_text = self.axes[1, 0].text(0.1, 0.9, '', transform=self.axes[1, 0].transAxes, 
                                             fontsize=12, verticalalignment='top')
        
        # Plot 4: Recent predictions distribution
        self.axes[1, 1].set_title('Recent Predictions Distribution')
        self.axes[1, 1].set_ylabel('Count')
        self.pred_hist = self.axes[1, 1].bar(['Non-Seizure', 'Seizure'], [0, 0], 
                                           color=['blue', 'red'], alpha=0.7)
        
        plt.tight_layout()
    
    def _monitor_output(self):
        """Monitor the output directory for new data"""
        while self.running:
            try:
                # Check for final stats
                stats_file = os.path.join(self.output_dir, 'final_stats.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    with self.data_lock:
                        self.windows_processed = stats.get('windows_processed', 0)
                        self.avg_inference_time = stats.get('avg_inference_time_ms', 0)
                        self.alerts_triggered = stats.get('alert_manager_stats', {}).get('alerts_triggered', 0)
                
                # Check for new alert files
                if os.path.exists(self.output_dir):
                    for filename in os.listdir(self.output_dir):
                        if filename.endswith('_metadata.json'):
                            alert_id = filename.replace('_metadata.json', '')
                            alert_file = os.path.join(self.output_dir, filename)
                            
                            # Check if we've already processed this alert
                            with self.data_lock:
                                if alert_id not in [alert['alert_id'] for alert in self.alert_times]:
                                    try:
                                        with open(alert_file, 'r') as f:
                                            alert_data = json.load(f)
                                        
                                        self.alert_times.append(alert_data['timestamp'])
                                        self.alert_confidences.append(alert_data['confidence'])
                                    except Exception as e:
                                        print(f"Error reading alert file {alert_file}: {e}")
                
                # Simulate real-time data (in a real system, this would come from the detector)
                current_time = time.time()
                with self.data_lock:
                    if not self.timestamps or current_time - self.timestamps[-1] >= 0.25:
                        # Simulate a prediction
                        confidence = np.random.beta(2, 10)  # Most predictions are low confidence
                        prediction = 1 if confidence > 0.5 else 0
                        
                        self.timestamps.append(current_time)
                        self.confidences.append(confidence)
                        self.predictions.append(prediction)
                        
                        # Keep only last 1000 points
                        if len(self.timestamps) > 1000:
                            self.timestamps = self.timestamps[-1000:]
                            self.confidences = self.confidences[-1000:]
                            self.predictions = self.predictions[-1000:]
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(1)
    
    def update_plots(self):
        """Update all plots with current data"""
        with self.data_lock:
            if not self.timestamps:
                return
            
            # Make copies of data for plotting
            timestamps = self.timestamps.copy()
            confidences = self.confidences.copy()
            predictions = self.predictions.copy()
            alert_times = self.alert_times.copy()
            alert_confidences = self.alert_confidences.copy()
            windows_processed = self.windows_processed
            avg_inference_time = self.avg_inference_time
            alerts_triggered = self.alerts_triggered
        
        try:
            # Update confidence plot
            if timestamps:
                times = [t - timestamps[0] for t in timestamps]
                if len(times) == len(confidences):
                    self.confidence_line.set_data(times, confidences)
                    threshold_values = [0.85] * len(times)
                    self.threshold_line.set_data(times, threshold_values)
                    
                    if times:
                        max_time = max(times)
                        if max_time > 0:
                            self.axes[0, 0].set_xlim(0, max_time)
                        else:
                            self.axes[0, 0].set_xlim(0, 10)
                    else:
                        self.axes[0, 0].set_xlim(0, 10)
            
            # Update alert plot
            if alert_times and timestamps:
                alert_times_relative = [t - timestamps[0] for t in alert_times]
                if len(alert_times_relative) == len(alert_confidences):
                    alert_data = np.column_stack([alert_times_relative, alert_confidences])
                    self.alert_scatter.set_offsets(alert_data)
                    
                    if timestamps:
                        max_time = max([t - timestamps[0] for t in timestamps])
                        if max_time > 0:
                            self.axes[0, 1].set_xlim(0, max_time)
                        else:
                            self.axes[0, 1].set_xlim(0, 10)
                    else:
                        self.axes[0, 1].set_xlim(0, 10)
            
            # Update performance metrics
            current_confidence = f"{confidences[-1]:.3f}" if confidences else "0.000"
            last_prediction = 'Seizure' if predictions and predictions[-1] == 1 else 'Non-Seizure' if predictions else 'None'
            time_running = f"{time.time() - timestamps[0]:.1f}s" if timestamps else "0.0s"
            
            perf_text = f"""Performance Metrics:
Windows Processed: {windows_processed}
Avg Inference Time: {avg_inference_time:.2f} ms
Alerts Triggered: {alerts_triggered}
Current Confidence: {current_confidence}
Last Prediction: {last_prediction}
Time Running: {time_running}"""
            
            self.perf_text.set_text(perf_text)
            
            # Update predictions distribution
            if predictions:
                non_seizure_count = sum(1 for p in predictions[-100:] if p == 0)
                seizure_count = sum(1 for p in predictions[-100:] if p == 1)
                
                self.pred_hist[0].set_height(non_seizure_count)
                self.pred_hist[1].set_height(seizure_count)
                max_count = max(non_seizure_count, seizure_count, 1)
                self.axes[1, 1].set_ylim(0, max_count)
            
            # Save the plot to a file instead of displaying
            output_file = os.path.join(self.output_dir, 'dashboard.png')
            self.fig.savefig(output_file, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"Error in dashboard: {e}")
    
    def run(self):
        """Run the dashboard"""
        print("Starting real-time dashboard...")
        print("Press Ctrl+C to stop")
        print(f"Dashboard will save plots to {self.output_dir}/dashboard.png")
        
        try:
            while self.running:
                self.update_plots()
                time.sleep(0.5)  # Update every 500ms
                
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            self.running = False
            plt.close()

def main():
    """Main function to run the dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time seizure detection dashboard')
    parser.add_argument('--output', default='realtime_output', help='Output directory to monitor')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        print(f"Error: Output directory {args.output} not found!")
        print("Please run the real-time detection system first")
        return
    
    # Create and run dashboard
    dashboard = RealTimeDashboard(args.output)
    dashboard.run()

if __name__ == "__main__":
    main() 