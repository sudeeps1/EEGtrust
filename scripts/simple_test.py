#!/usr/bin/env python3
"""Simple System Test for EEGTrust"""

import os
import sys
import numpy as np
from datetime import datetime

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_data_files():
    """Test data files exist"""
    print("🔍 Testing data files...")
    
    required_files = [
        "best_model.pth",
        "prepared_data/chb01_windows.npy",
        "prepared_data/chb01_labels.npy"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found")
        return True

def test_data_loading():
    """Test data loading without PyTorch"""
    print("🔍 Testing data loading...")
    try:
        windows = np.load("prepared_data/chb01_windows.npy")
        labels = np.load("prepared_data/chb01_labels.npy")
        
        print(f"✅ Data loaded successfully")
        print(f"   Windows shape: {windows.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Windows dtype: {windows.dtype}")
        print(f"   Labels dtype: {labels.dtype}")
        
        # Count classes safely
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"   Class distribution: {dict(zip(unique_labels, counts))}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_system_resources():
    """Test system resources"""
    print("🔍 Testing system resources...")
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"✅ System monitoring working")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory Usage: {memory.percent:.1f}%")
        print(f"   Available Memory: {memory.available / (1024**3):.1f} GB")
        
        return True
    except Exception as e:
        print(f"❌ System monitoring failed: {e}")
        return False

def main():
    """Run simple system test"""
    print("🚀 EEGTrust Simple System Test")
    print("=" * 40)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Files", test_data_files),
        ("Data Loading", test_data_loading),
        ("System Resources", test_system_resources)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All basic tests passed!")
        print("💡 Note: PyTorch tests skipped due to memory constraints")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print(f"\n💡 To run full tests with PyTorch:")
    print("   1. Increase Windows paging file size")
    print("   2. Or run: python scripts/quick_test.py")

if __name__ == "__main__":
    main() 