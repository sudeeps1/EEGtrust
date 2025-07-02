#!/usr/bin/env python3
"""
Master Test Runner for EEGTrust Seizure Detection System
- Runs all accuracy, latency, and integration tests
- Generates comprehensive performance report
- Provides recommendations for optimization
- Creates summary dashboard
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def run_test_script(script_name: str, description: str) -> dict:
    """Run a test script and capture results"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            print(f"Output: {result.stdout}")
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return {
                'status': 'failed',
                'error': result.stderr,
                'duration': time.time() - start_time
            }
        
        return {
            'status': 'success',
            'duration': time.time() - start_time,
            'output': result.stdout
        }
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 30 minutes")
        return {
            'status': 'timeout',
            'duration': 1800,
            'error': 'Test timed out'
        }
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return {
            'status': 'error',
            'duration': time.time() - start_time,
            'error': str(e)
        }

def find_latest_results(test_type: str) -> str:
    """Find the latest results directory for a test type"""
    base_dir = f"{test_type}_test_results"
    
    # Look for existing directories
    existing_dirs = [d for d in os.listdir('.') if d.startswith(base_dir)]
    
    if not existing_dirs:
        return None
    
    # Return the most recent one
    return max(existing_dirs, key=lambda x: os.path.getctime(x))

def load_test_results(results_dir: str) -> dict:
    """Load test results from JSON file"""
    results_file = os.path.join(results_dir, 'results.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        return None

def create_summary_dashboard(results: dict, output_dir: str):
    """Create a comprehensive summary dashboard"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEGTrust System Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Metrics
    if 'accuracy_results' in results and results['accuracy_results']:
        acc_data = results['accuracy_results']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        values = [acc_data.get(f'test_metrics', {}).get(m, 0) for m in metrics]
        
        axes[0, 0].bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'])
        axes[0, 0].set_title('Accuracy Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Latency Performance
    if 'latency_results' in results and results['latency_results']:
        lat_data = results['latency_results']
        single_results = lat_data.get('single_inference_results', {})
        
        latency_metrics = ['avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms']
        latency_values = [single_results.get(m, 0) for m in latency_metrics]
        
        axes[0, 1].bar([m.replace('_', ' ').title() for m in latency_metrics], 
                      latency_values, color='#2E86AB')
        axes[0, 1].set_title('Latency Performance')
        axes[0, 1].set_ylabel('Latency (ms)')
        for i, v in enumerate(latency_values):
            axes[0, 1].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
    
    # 3. System Resources
    if 'integration_results' in results and results['integration_results']:
        int_data = results['integration_results']
        rel_data = int_data.get('reliability_results', {})
        
        if rel_data:
            axes[0, 2].bar(['CPU Usage', 'Memory Usage'], 
                          [rel_data.get('avg_cpu_usage', 0), rel_data.get('avg_memory_usage', 0)],
                          color=['#A23B72', '#F18F01'])
            axes[0, 2].set_title('System Resource Usage')
            axes[0, 2].set_ylabel('Usage (%)')
            axes[0, 2].set_ylim(0, 100)
    
    # 4. Alert System Performance
    if 'integration_results' in results and results['integration_results']:
        alert_data = int_data.get('alert_results', {})
        
        if alert_data:
            alert_metrics = ['Seizure Detection Rate', 'False Positive Rate']
            alert_values = [alert_data.get('seizure_detection_rate', 0), 
                          alert_data.get('false_positive_rate', 0)]
            
            axes[1, 0].bar(alert_metrics, alert_values, color=['#C73E1D', '#592E83'])
            axes[1, 0].set_title('Alert System Performance')
            axes[1, 0].set_ylabel('Rate')
            axes[1, 0].set_ylim(0, 1)
            for i, v in enumerate(alert_values):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 5. Throughput Performance
    if 'latency_results' in results and results['latency_results']:
        cont_data = lat_data.get('continuous_results', {})
        
        if cont_data:
            axes[1, 1].bar(['Throughput'], [cont_data.get('avg_throughput_wps', 0)], color='#2E86AB')
            axes[1, 1].set_title('System Throughput')
            axes[1, 1].set_ylabel('Windows/Second')
            axes[1, 1].text(0, cont_data.get('avg_throughput_wps', 0) + 0.1, 
                           f"{cont_data.get('avg_throughput_wps', 0):.1f}", ha='center', va='bottom')
    
    # 6. Error Handling Score
    if 'integration_results' in results and results['integration_results']:
        error_data = int_data.get('error_results', {})
        
        if error_data:
            score = error_data.get('error_handling_score', 0)
            axes[1, 2].bar(['Error Handling'], [score], color='#F18F01')
            axes[1, 2].set_title('Error Handling Score')
            axes[1, 2].set_ylabel('Score / 10')
            axes[1, 2].set_ylim(0, 10)
            axes[1, 2].text(0, score + 0.1, f'{score}/10', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_recommendations(results: dict) -> list:
    """Generate recommendations based on test results"""
    recommendations = []
    
    # Accuracy recommendations
    if 'accuracy_results' in results and results['accuracy_results']:
        acc_data = results['accuracy_results']
        test_metrics = acc_data.get('test_metrics', {})
        
        if test_metrics.get('accuracy', 0) < 0.85:
            recommendations.append("🔴 Consider retraining the model with more data or different architecture")
        
        if test_metrics.get('precision', 0) < 0.8:
            recommendations.append("🟡 High false positive rate - consider adjusting confidence thresholds")
        
        if test_metrics.get('recall', 0) < 0.8:
            recommendations.append("🟡 Low seizure detection rate - consider data augmentation or class balancing")
    
    # Latency recommendations
    if 'latency_results' in results and results['latency_results']:
        lat_data = results['latency_results']
        single_results = lat_data.get('single_inference_results', {})
        
        if single_results.get('avg_latency_ms', 0) > 50:
            recommendations.append("🔴 High latency - consider model optimization or GPU acceleration")
        
        if single_results.get('p99_latency_ms', 0) > 100:
            recommendations.append("🟡 High 99th percentile latency - check for memory leaks or system bottlenecks")
    
    # System resource recommendations
    if 'integration_results' in results and results['integration_results']:
        int_data = results['integration_results']
        rel_data = int_data.get('reliability_results', {})
        
        if rel_data.get('avg_cpu_usage', 0) > 80:
            recommendations.append("🟡 High CPU usage - consider optimizing preprocessing or using more efficient models")
        
        if rel_data.get('avg_memory_usage', 0) > 80:
            recommendations.append("🟡 High memory usage - consider reducing batch sizes or implementing memory cleanup")
    
    # Alert system recommendations
    if 'integration_results' in results and results['integration_results']:
        alert_data = int_data.get('alert_results', {})
        
        if alert_data.get('seizure_detection_rate', 0) < 0.8:
            recommendations.append("🔴 Low seizure detection rate - critical for clinical safety")
        
        if alert_data.get('false_positive_rate', 0) > 0.1:
            recommendations.append("🟡 High false positive rate - may cause alert fatigue")
    
    # General recommendations
    if not recommendations:
        recommendations.append("✅ System performance meets clinical requirements")
    
    return recommendations

def create_test_report(results: dict, test_status: dict, output_dir: str):
    """Create a comprehensive test report"""
    report_file = os.path.join(output_dir, 'test_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# EEGTrust System Test Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test Status Summary
        f.write("## Test Status Summary\n\n")
        f.write("| Test Type | Status | Duration |\n")
        f.write("|-----------|--------|----------|\n")
        
        for test_type, status in test_status.items():
            duration = status.get('duration', 0)
            status_icon = "✅" if status.get('status') == 'success' else "❌"
            f.write(f"| {test_type} | {status_icon} {status.get('status', 'unknown')} | {duration:.1f}s |\n")
        
        f.write("\n")
        
        # Performance Summary
        f.write("## Performance Summary\n\n")
        
        if results.get('accuracy_results'):
            acc_data = results['accuracy_results']
            test_metrics = acc_data.get('test_metrics', {})
            f.write("### Accuracy Metrics\n\n")
            f.write(f"- **Accuracy:** {test_metrics.get('accuracy', 0):.3f}\n")
            f.write(f"- **Precision:** {test_metrics.get('precision', 0):.3f}\n")
            f.write(f"- **Recall:** {test_metrics.get('recall', 0):.3f}\n")
            f.write(f"- **F1 Score:** {test_metrics.get('f1', 0):.3f}\n")
            f.write(f"- **AUC:** {test_metrics.get('auc', 0):.3f}\n\n")
        
        if results.get('latency_results'):
            lat_data = results['latency_results']
            single_results = lat_data.get('single_inference_results', {})
            f.write("### Latency Performance\n\n")
            f.write(f"- **Average Latency:** {single_results.get('avg_latency_ms', 0):.1f} ms\n")
            f.write(f"- **P50 Latency:** {single_results.get('p50_latency_ms', 0):.1f} ms\n")
            f.write(f"- **P95 Latency:** {single_results.get('p95_latency_ms', 0):.1f} ms\n")
            f.write(f"- **P99 Latency:** {single_results.get('p99_latency_ms', 0):.1f} ms\n\n")
        
        if results.get('integration_results'):
            int_data = results['integration_results']
            rel_data = int_data.get('reliability_results', {})
            alert_data = int_data.get('alert_results', {})
            f.write("### System Performance\n\n")
            f.write(f"- **Avg CPU Usage:** {rel_data.get('avg_cpu_usage', 0):.1f}%\n")
            f.write(f"- **Avg Memory Usage:** {rel_data.get('avg_memory_usage', 0):.1f}%\n")
            f.write(f"- **Seizure Detection Rate:** {alert_data.get('seizure_detection_rate', 0):.3f}\n")
            f.write(f"- **False Positive Rate:** {alert_data.get('false_positive_rate', 0):.3f}\n\n")
        
        # Recommendations
        recommendations = generate_recommendations(results)
        f.write("## Recommendations\n\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
        
        f.write("\n")
        
        # Clinical Readiness Assessment
        f.write("## Clinical Readiness Assessment\n\n")
        
        # Define clinical requirements
        clinical_requirements = {
            'accuracy': 0.85,
            'latency_ms': 50,
            'seizure_detection_rate': 0.8,
            'false_positive_rate': 0.1
        }
        
        readiness_score = 0
        total_requirements = len(clinical_requirements)
        
        if results.get('accuracy_results'):
            test_metrics = results['accuracy_results'].get('test_metrics', {})
            if test_metrics.get('accuracy', 0) >= clinical_requirements['accuracy']:
                readiness_score += 1
                f.write("✅ **Accuracy:** Meets clinical requirements\n")
            else:
                f.write("❌ **Accuracy:** Below clinical requirements\n")
        
        if results.get('latency_results'):
            single_results = results['latency_results'].get('single_inference_results', {})
            if single_results.get('avg_latency_ms', 1000) <= clinical_requirements['latency_ms']:
                readiness_score += 1
                f.write("✅ **Latency:** Meets clinical requirements\n")
            else:
                f.write("❌ **Latency:** Above clinical requirements\n")
        
        if results.get('integration_results'):
            alert_data = results['integration_results'].get('alert_results', {})
            if alert_data.get('seizure_detection_rate', 0) >= clinical_requirements['seizure_detection_rate']:
                readiness_score += 1
                f.write("✅ **Seizure Detection:** Meets clinical requirements\n")
            else:
                f.write("❌ **Seizure Detection:** Below clinical requirements\n")
            
            if alert_data.get('false_positive_rate', 1) <= clinical_requirements['false_positive_rate']:
                readiness_score += 1
                f.write("✅ **False Positive Rate:** Meets clinical requirements\n")
            else:
                f.write("❌ **False Positive Rate:** Above clinical requirements\n")
        
        readiness_percentage = (readiness_score / total_requirements) * 100
        f.write(f"\n**Overall Clinical Readiness:** {readiness_percentage:.1f}% ({readiness_score}/{total_requirements})\n")
        
        if readiness_percentage >= 80:
            f.write("\n🎉 **System is ready for clinical deployment!**\n")
        elif readiness_percentage >= 60:
            f.write("\n⚠️ **System needs optimization before clinical deployment.**\n")
        else:
            f.write("\n🚨 **System requires significant improvements before clinical deployment.**\n")

def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description='Run comprehensive EEGTrust system tests')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy tests')
    parser.add_argument('--skip-latency', action='store_true', help='Skip latency tests')
    parser.add_argument('--skip-integration', action='store_true', help='Skip integration tests')
    parser.add_argument('--output-dir', default='comprehensive_test_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🚀 EEGTrust Comprehensive System Testing")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("best_model.pth"):
        print("❌ Error: Model file 'best_model.pth' not found!")
        print("Please train the model first using: python scripts/train_with_existing_data.py")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define tests to run
    tests = []
    if not args.skip_accuracy:
        tests.append(('test_accuracy.py', 'Accuracy Testing'))
    if not args.skip_latency:
        tests.append(('test_latency.py', 'Latency Testing'))
    if not args.skip_integration:
        tests.append(('test_integration.py', 'Integration Testing'))
    
    if not tests:
        print("❌ No tests selected to run!")
        return
    
    # Run tests
    test_status = {}
    results = {}
    
    for script_name, description in tests:
        status = run_test_script(script_name, description)
        test_status[description] = status
        
        # Try to load results
        if status['status'] == 'success':
            results_dir = find_latest_results(description.lower().replace(' ', '_'))
            if results_dir:
                test_results = load_test_results(results_dir)
                if test_results:
                    results[f"{description.lower().replace(' ', '_')}_results"] = test_results
    
    # Create summary
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}")
    
    # Create visualizations
    create_summary_dashboard(results, output_dir)
    
    # Generate report
    create_test_report(results, test_status, output_dir)
    
    # Print summary
    print(f"\n✅ Testing completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📄 Report: {output_dir}/test_report.md")
    print(f"📈 Dashboard: {output_dir}/performance_summary.png")
    
    # Print quick summary
    successful_tests = sum(1 for status in test_status.values() if status['status'] == 'success')
    total_tests = len(test_status)
    
    print(f"\n📋 Test Summary: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed. Check the detailed report for more information.")

if __name__ == "__main__":
    main() 