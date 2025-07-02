#!/usr/bin/env python3
"""
Simple text-based real-time dashboard for seizure detection monitoring
"""

import os
import json
import time
import threading
import sys
from datetime import datetime

# Add the parent directory to the path to import eegtrust modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleDashboard:
    """Simple text-based dashboard for seizure detection monitoring"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.running = True
        
        # Performance metrics
        self.windows_processed = 0
        self.avg_inference_time = 0
        self.alerts_triggered = 0
        self.start_time = time.time()
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_output)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
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
                    alert_files = [f for f in os.listdir(self.output_dir) if f.endswith('_metadata.json')]
                    if alert_files:
                        print(f"\n[ALERT] Found {len(alert_files)} alert(s) in output directory")
                        for alert_file in alert_files[-3:]:  # Show last 3 alerts
                            try:
                                with open(os.path.join(self.output_dir, alert_file), 'r') as f:
                                    alert_data = json.load(f)
                                print(f"  - {alert_file}: Confidence {alert_data.get('confidence', 0):.3f}")
                            except Exception as e:
                                print(f"  - Error reading {alert_file}: {e}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Error in monitoring thread: {e}")
                time.sleep(2)
    
    def display_status(self):
        """Display current status"""
        with self.data_lock:
            windows_processed = self.windows_processed
            avg_inference_time = self.avg_inference_time
            alerts_triggered = self.alerts_triggered
        
        runtime = time.time() - self.start_time
        
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("           REAL-TIME SEIZURE DETECTION DASHBOARD")
        print("=" * 60)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Output Directory: {self.output_dir}")
        print("-" * 60)
        
        print("PERFORMANCE METRICS:")
        print(f"  Windows Processed: {windows_processed}")
        print(f"  Average Inference Time: {avg_inference_time:.2f} ms")
        print(f"  Alerts Triggered: {alerts_triggered}")
        
        if windows_processed > 0:
            windows_per_second = windows_processed / runtime
            print(f"  Processing Rate: {windows_per_second:.1f} windows/second")
        
        print("-" * 60)
        print("SYSTEM STATUS:")
        
        # Check if detection system is running
        if os.path.exists(os.path.join(self.output_dir, 'realtime.log')):
            try:
                with open(os.path.join(self.output_dir, 'realtime.log'), 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"  Last Log Entry: {last_line}")
                    else:
                        print("  Status: No log entries yet")
            except Exception as e:
                print(f"  Status: Error reading log ({e})")
        else:
            print("  Status: Waiting for detection system to start...")
        
        print("-" * 60)
        print("RECENT ALERTS:")
        
        # Show recent alerts
        if os.path.exists(self.output_dir):
            alert_files = [f for f in os.listdir(self.output_dir) if f.endswith('_metadata.json')]
            if alert_files:
                # Sort by modification time (newest first)
                alert_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)), reverse=True)
                
                for alert_file in alert_files[:5]:  # Show last 5 alerts
                    try:
                        with open(os.path.join(self.output_dir, alert_file), 'r') as f:
                            alert_data = json.load(f)
                        
                        timestamp = alert_data.get('timestamp', 0)
                        confidence = alert_data.get('confidence', 0)
                        alert_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                        
                        print(f"  [{alert_time}] Confidence: {confidence:.3f} - {alert_file}")
                    except Exception as e:
                        print(f"  Error reading {alert_file}: {e}")
            else:
                print("  No alerts detected yet")
        
        print("-" * 60)
        print("Press Ctrl+C to stop")
        print("=" * 60)
    
    def run(self):
        """Run the dashboard"""
        print("Starting simple real-time dashboard...")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                self.display_status()
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            self.running = False

def main():
    """Main function to run the dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple real-time seizure detection dashboard')
    parser.add_argument('--output', default='realtime_output', help='Output directory to monitor')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        print(f"Error: Output directory {args.output} not found!")
        print("Please run the real-time detection system first")
        return
    
    # Create and run dashboard
    dashboard = SimpleDashboard(args.output)
    dashboard.run()

if __name__ == "__main__":
    main() 