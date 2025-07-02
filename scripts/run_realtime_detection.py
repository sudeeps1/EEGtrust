#!/usr/bin/env python3
"""
Script to run real-time seizure detection with the trained EEGTrust model
"""

import os
import sys
import time
import numpy as np
import argparse
from datetime import datetime

# Add the parent directory to the path to import eegtrust modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegtrust.realtime import RealTimeSeizureDetector

def main():
    """Run real-time seizure detection"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time seizure detection')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--data', default='prepared_data/chb01_windows.npy', help='Path to data file')
    parser.add_argument('--output', help='Output directory (default: timestamped)')
    
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model
    data_file = args.data
    
    # Use provided output directory or create timestamped one
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"realtime_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train a model first using train_with_existing_data.py")
        return
    
    # Check if data exists
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please prepare data first using prepare_training_data.py")
        return
    
    print("=== Real-Time Seizure Detection System ===")
    print(f"Model: {model_path}")
    print(f"Data: {data_file}")
    print(f"Output: {output_dir}")
    print(f"Start time: {datetime.now()}")
    print("=" * 50)
    
    # Create and start detector
    detector = RealTimeSeizureDetector(model_path, data_file, output_dir)
    
    try:
        print("Starting real-time detection...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        detector.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Stopping real-time detection...")
        detector.stop()
        
        print(f"\nDetection completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Check the log file for detailed information")
        print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main() 