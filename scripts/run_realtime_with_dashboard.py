#!/usr/bin/env python3
"""
Combined script to run real-time seizure detection with dashboard
"""

import subprocess
import threading
import time
import sys
import os
import signal
from datetime import datetime

def run_detection_system(output_dir):
    """Run the real-time detection system"""
    print("Starting real-time detection system...")
    try:
        # Run the detection system with the specified output directory
        process = subprocess.Popen([
            sys.executable, "scripts/run_realtime_detection.py",
            "--model", "best_model.pth",
            "--data", "prepared_data/chb01_windows.npy",
            "--output", output_dir
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[DETECTION] {output.strip()}")
        
        return_code = process.poll()
        if return_code != 0:
            print(f"Detection system exited with code {return_code}")
            
    except Exception as e:
        print(f"Error running detection system: {e}")

def run_dashboard(output_dir):
    """Run the simple dashboard"""
    print("Starting simple dashboard...")
    try:
        # Wait a bit for detection system to start
        time.sleep(3)
        
        # Run the simple dashboard
        process = subprocess.Popen([
            sys.executable, "scripts/simple_dashboard.py",
            "--output", output_dir
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[DASHBOARD] {output.strip()}")
        
        return_code = process.poll()
        if return_code != 0:
            print(f"Dashboard exited with code {return_code}")
            
    except Exception as e:
        print(f"Error running dashboard: {e}")

def main():
    """Main function to run both systems"""
    # Create timestamped output directory
    output_dir = f"realtime_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=== Real-Time Seizure Detection with Dashboard ===")
    print("Model: best_model.pth")
    print("Data: prepared_data/chb01_windows.npy")
    print(f"Output: {output_dir}")
    print(f"Start time: {datetime.now()}")
    print("=" * 60)
    
    # Check if required files exist
    if not os.path.exists("best_model.pth"):
        print("Error: best_model.pth not found!")
        print("Please train the model first using: python scripts/train_with_existing_data.py")
        return
    
    if not os.path.exists("prepared_data/chb01_windows.npy"):
        print("Error: prepared_data/chb01_windows.npy not found!")
        print("Please prepare the data first using: python scripts/prepare_training_data_v2.py")
        return
    
    print("Starting both detection system and dashboard...")
    print("Press Ctrl+C to stop both systems")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start detection system in a thread
    detection_thread = threading.Thread(target=run_detection_system, args=(output_dir,))
    detection_thread.daemon = True
    detection_thread.start()
    
    # Start dashboard in a thread
    dashboard_thread = threading.Thread(target=run_dashboard, args=(output_dir,))
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    try:
        # Wait for both threads
        while detection_thread.is_alive() and dashboard_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping both systems...")
        print("Please wait for processes to terminate...")
        
        # Give some time for graceful shutdown
        time.sleep(2)
        
        print("Systems stopped.")

if __name__ == "__main__":
    main() 