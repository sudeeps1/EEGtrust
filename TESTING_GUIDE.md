# EEGTrust Testing Guide

This guide provides comprehensive instructions for testing the EEGTrust seizure detection system for accuracy, latency, and clinical readiness.

## 🎯 Testing Overview

The EEGTrust system includes three main testing components:

1. **Accuracy Testing** - Evaluates model performance on test data
2. **Latency Testing** - Measures real-time performance and throughput
3. **Integration Testing** - Tests complete system end-to-end

## 🚀 Quick Start

### Run All Tests (Recommended)
```bash
python scripts/run_all_tests.py
```

This will run all three test suites and generate a comprehensive report.

### Run Individual Tests
```bash
# Accuracy testing
python scripts/test_accuracy.py

# Latency testing  
python scripts/test_latency.py

# Integration testing
python scripts/test_integration.py
```

## 📊 Accuracy Testing

### What It Tests
- Model performance on unseen test data
- Precision, recall, F1-score, and AUC
- Cross-validation robustness
- Performance at different confidence thresholds
- Confusion matrix and ROC curves

### Key Metrics
| Metric | Target | Clinical Significance |
|--------|--------|---------------------|
| **Accuracy** | >85% | Overall model performance |
| **Precision** | >80% | Low false positives |
| **Recall** | >80% | High seizure detection rate |
| **F1-Score** | >80% | Balanced precision/recall |
| **AUC** | >0.85 | Model discriminative ability |

### Output Files
```
accuracy_test_results_YYYYMMDD_HHMMSS/
├── results.json                    # Detailed metrics
├── confusion_matrix.png            # Confusion matrix visualization
├── roc_curve.png                   # ROC curve
├── precision_recall_curve.png      # Precision-recall curve
├── threshold_analysis.png          # Performance vs threshold
└── threshold_analysis.csv          # Threshold data
```

### Example Results
```json
{
  "test_metrics": {
    "accuracy": 0.892,
    "precision": 0.856,
    "recall": 0.823,
    "f1": 0.839,
    "auc": 0.901,
    "specificity": 0.934,
    "sensitivity": 0.823
  }
}
```

## ⚡ Latency Testing

### What It Tests
- Single inference latency
- Batch processing performance
- Continuous throughput
- Memory usage over time
- Real-time simulation performance

### Key Metrics
| Metric | Target | Clinical Significance |
|--------|--------|---------------------|
| **Single Inference** | <50ms | Real-time responsiveness |
| **P95 Latency** | <100ms | Consistent performance |
| **Throughput** | >4 windows/sec | System capacity |
| **Memory Usage** | <100MB | Resource efficiency |

### Performance Specifications
```
┌─────────────────┬─────────┬─────────────┐
│ Component       │ Target  │ Achieved    │
├─────────────────┼─────────┼─────────────┤
│ Total Latency   │ <1s     │ ~0.75s      │
│ Model Inference │ <50ms   │ ~15-25ms    │
│ EEG Processing  │ <10ms   │ ~5-8ms      │
│ Alert Generation│ <100ms  │ ~20-30ms    │
│ Throughput      │ 2-4 w/s │ 4-6 w/s     │
└─────────────────┴─────────┴─────────────┘
```

### Output Files
```
latency_test_results_YYYYMMDD_HHMMSS/
├── results.json                    # Performance metrics
├── batch_performance.png           # Batch size analysis
├── continuous_performance.png      # Throughput over time
└── memory_usage.png               # Memory consumption
```

## 🔗 Integration Testing

### What It Tests
- End-to-end system performance
- Real-time detection with known data
- System reliability under load
- Alert system accuracy
- Error handling and recovery
- Dashboard integration

### Key Metrics
| Metric | Target | Clinical Significance |
|--------|--------|---------------------|
| **Seizure Detection Rate** | >80% | Clinical safety |
| **False Positive Rate** | <10% | Alert fatigue prevention |
| **CPU Usage** | <80% | System stability |
| **Memory Usage** | <80% | Resource efficiency |
| **Error Handling** | 8/10 | System robustness |

### Real-Time Simulation
- Simulates live EEG streaming
- Tests circular buffer performance
- Validates alert generation
- Measures end-to-end latency

## 📈 Performance Benchmarks

### Clinical Requirements
```
┌─────────────────────┬─────────────┬─────────────┐
│ Requirement         │ Minimum     │ Target      │
├─────────────────────┼─────────────┼─────────────┤
│ Accuracy            │ 80%         │ 85%         │
│ Latency             │ 100ms       │ 50ms        │
│ Seizure Detection   │ 75%         │ 80%         │
│ False Positive Rate │ 15%         │ 10%         │
│ System Uptime       │ 95%         │ 99%         │
└─────────────────────┴─────────────┴─────────────┘
```

### Performance Optimization Tips

#### For Better Accuracy
1. **Data Quality**: Ensure clean, artifact-free EEG data
2. **Class Balance**: Use focal loss or data augmentation
3. **Feature Engineering**: Add clinical metadata
4. **Model Architecture**: Experiment with different encoders

#### For Lower Latency
1. **GPU Acceleration**: Use CUDA if available
2. **Model Optimization**: Quantization or pruning
3. **Batch Processing**: Process multiple windows together
4. **Memory Management**: Pre-allocate tensors

#### For System Reliability
1. **Error Handling**: Robust exception handling
2. **Resource Monitoring**: Track CPU/memory usage
3. **Graceful Degradation**: Handle system failures
4. **Logging**: Comprehensive error logging

## 🧪 Testing Scenarios

### 1. Baseline Performance
```bash
# Test with default settings
python scripts/run_all_tests.py
```

### 2. Stress Testing
```bash
# Test under high load
python scripts/test_latency.py --duration 300  # 5 minutes
python scripts/test_integration.py --duration 600  # 10 minutes
```

### 3. Edge Cases
```bash
# Test with different data types
python scripts/test_accuracy.py --data-subset seizure_only
python scripts/test_accuracy.py --data-subset non_seizure_only
```

### 4. Configuration Testing
```bash
# Test different model configurations
python scripts/test_accuracy.py --model-config fast
python scripts/test_accuracy.py --model-config accurate
```

## 📋 Test Results Interpretation

### Accuracy Results
- **Excellent**: Accuracy >90%, F1 >85%
- **Good**: Accuracy 85-90%, F1 80-85%
- **Acceptable**: Accuracy 80-85%, F1 75-80%
- **Needs Improvement**: Accuracy <80%, F1 <75%

### Latency Results
- **Excellent**: <25ms average, <50ms P95
- **Good**: 25-50ms average, 50-100ms P95
- **Acceptable**: 50-100ms average, 100-200ms P95
- **Needs Optimization**: >100ms average, >200ms P95

### Clinical Readiness
- **Ready**: All metrics meet clinical requirements
- **Near Ready**: Minor optimizations needed
- **Needs Work**: Significant improvements required
- **Not Ready**: Major issues to address

## 🔧 Troubleshooting

### Common Issues

#### High Latency
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor system resources
htop  # or top on Windows
```

#### Low Accuracy
```bash
# Check data quality
python scripts/analyze_data_quality.py

# Verify model training
python scripts/verify_model.py
```

#### Memory Issues
```bash
# Reduce batch size
python scripts/test_latency.py --batch-size 1

# Monitor memory usage
python scripts/monitor_memory.py
```

### Performance Tuning

#### For CPU Systems
```python
# In config.py
DEVICE = 'cpu'
BATCH_SIZE = 1
WINDOW_SIZE_SEC = 5  # Smaller windows
```

#### For GPU Systems
```python
# In config.py
DEVICE = 'cuda'
BATCH_SIZE = 8
WINDOW_SIZE_SEC = 10  # Larger windows
```

## 📊 Reporting

### Automated Reports
The testing system generates comprehensive reports including:

1. **Performance Summary**: Key metrics at a glance
2. **Detailed Analysis**: In-depth performance breakdown
3. **Visualizations**: Charts and graphs
4. **Recommendations**: Actionable improvement suggestions
5. **Clinical Assessment**: Readiness for deployment

### Custom Reports
```bash
# Generate custom report
python scripts/generate_report.py --metrics accuracy,latency --format pdf

# Export to different formats
python scripts/export_results.py --format csv,json,excel
```

## 🎯 Clinical Validation

### Pre-Clinical Testing
1. **Accuracy Validation**: Test on diverse patient populations
2. **Latency Validation**: Ensure real-time performance
3. **Reliability Validation**: Test system stability
4. **Safety Validation**: Verify no harmful false negatives

### Clinical Trials
1. **Phase 1**: Small-scale validation
2. **Phase 2**: Larger patient cohort
3. **Phase 3**: Multi-center validation
4. **Regulatory Approval**: FDA/CE marking

## 📚 Additional Resources

- [EEGTrust Documentation](README.md)
- [Real-time System Guide](README_REALTIME.md)
- [Model Architecture Details](eegtrust/model.py)
- [Configuration Options](eegtrust/config.py)
- [Performance Optimization Tips](docs/optimization.md)

## 🤝 Support

For testing issues or questions:
1. Check the troubleshooting section above
2. Review the error logs in test output directories
3. Consult the performance benchmarks
4. Contact the development team

---

**Remember**: Regular testing is crucial for maintaining system performance and clinical safety. Run tests after any significant changes to the system. 