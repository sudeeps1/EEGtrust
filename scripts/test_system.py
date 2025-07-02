#!/usr/bin/env python3
"""
System Test Script for EEGTrust
- Verifies all components are working correctly
- Tests model loading, data loading, and basic functionality
- Provides a quick health check of the system
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test imports"""
    print("🔍 Testing imports...")
    try:
        from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
        from eegtrust.config import SAMPLE_RATE, WINDOW_SIZE_SAMPLES
        import psutil
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("🔍 Testing model loading...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
        print(f"✅ Model loaded on {device}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("🔍 Testing data loading...")
    try:
        windows = np.load("prepared_data/chb01_windows.npy")
        labels = np.load("prepared_data/chb01_labels.npy")
        print(f"✅ Data loaded: {len(windows)} windows, {len(labels)} labels")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_inference():
    """Test basic model inference"""
    print("\n🔍 Testing model inference...")
    
    try:
        from eegtrust.model import SSLPretrainedEncoder, STGNN, NeuroSymbolicExplainer
        from eegtrust.config import WINDOW_SIZE_SAMPLES
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        encoder = SSLPretrainedEncoder(23, WINDOW_SIZE_SAMPLES, 128)
        stgnn = STGNN(128, num_layers=2, num_heads=4)
        explainer = NeuroSymbolicExplainer(128)
        
        # Move to device
        encoder = encoder.to(device)
        stgnn = stgnn.to(device)
        explainer = explainer.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 23, WINDOW_SIZE_SAMPLES).to(device)
        
        # Run inference
        with torch.no_grad():
            features = encoder(dummy_input)
            features_seq = features.unsqueeze(1)
            stgnn_logits = stgnn(features_seq)
            explainer_logits = explainer(features)
            logits = (stgnn_logits + explainer_logits) / 2
            probs = torch.softmax(logits, dim=1)
        
        print(f"✅ Model inference successful on {device}")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Seizure probability: {probs[0, 1].item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run model inference: {e}")
        return False

def test_system_resources():
    """Test system resource monitoring"""
    print("\n🔍 Testing system resources...")
    
    try:
        import psutil
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"✅ System monitoring working")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory Usage: {memory.percent:.1f}%")
        print(f"   Available Memory: {memory.available / (1024**3):.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to monitor system resources: {e}")
        return False

def run_system_test():
    """Run complete system test"""
    print("🚀 EEGTrust System Health Check")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Model Inference", test_model_inference),
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
    print(f"\n{'='*60}")
    print("SYSTEM TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    duration = datetime.now() - start_time
    print(f"Test duration: {duration.total_seconds():.2f} seconds")
    
    return passed == len(results)

if __name__ == "__main__":
    run_system_test() 