# EEGTrust

EEGTrust is a pediatric EEG seizure detection project with:
- data preprocessing and windowing utilities,
- deep-learning training pipelines,
- real-time inference and alerting,
- explainability helpers,
- performance and integration test scripts.

This README focuses on getting the repository running quickly and reliably.

## Quick Start

### 1) Create a Python environment

Use Python 3.9-3.11 for best dependency compatibility.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For real-time-only dependencies:

```bash
pip install -r requirements_realtime.txt
```

### 2) Prepare data

Most training/testing scripts expect preprocessed NumPy files in `prepared_data/`, for example:
- `prepared_data/chb01_windows.npy`
- `prepared_data/chb01_labels.npy`
- `prepared_data/chb02_windows.npy`
- `prepared_data/chb02_labels.npy`

Optional helpers:

```bash
python scripts/download_data.py
python scripts/prepare_training_data.py
python scripts/prepare_training_data_v2.py
```

### 3) Train

Primary training entrypoint:

```bash
python scripts/train_with_existing_data.py
```

Small/smoke mode (requires `prepared_data/small_windows.npy` and `prepared_data/small_labels.npy`):

```bash
python scripts/train_with_existing_data.py small
```

### 4) Run tests

```bash
python scripts/run_all_tests.py
```

Or run individual suites:

```bash
python scripts/test_accuracy.py
python scripts/test_latency.py
python scripts/test_integration.py
python scripts/test_metadata.py
```

### 5) Real-time inference

```bash
python scripts/run_realtime_detection.py
```

With dashboard:

```bash
python scripts/run_realtime_with_dashboard.py
```

Standalone dashboard tools:

```bash
python scripts/realtime_dashboard.py
python scripts/simple_dashboard.py
```

## Project Layout

```text
EEGtrust/
├── eegtrust/                      # Core package (data, model, train, inference, utils)
├── scripts/                       # Training, evaluation, realtime and utility scripts
├── dashboard/                     # Dashboard app code
├── requirements.txt
├── requirements_realtime.txt
├── README_REALTIME.md             # Detailed realtime system docs
└── TESTING_GUIDE.md               # Detailed testing docs
```

## Core Commands

```bash
# Quick sanity check
python scripts/quick_test.py

# Full system test
python scripts/test_system.py

# Export ONNX
python scripts/export_onnx.py
```

## Notes on Current Training Paths

- `scripts/train_with_existing_data.py` is the maintained practical training path.
- `eegtrust/train.py` contains reusable training functions and legacy code paths; its default `prepare_data()` flow is intentionally guarded and not the primary entrypoint.

## Troubleshooting

- **`ModuleNotFoundError` for `numpy` or `torch`**  
  Confirm you activated your virtual environment and installed dependencies from `requirements.txt`.

- **Tests fail with `prepared_data` not found**  
  Create/populate the `prepared_data/` directory with expected `*_windows.npy` and `*_labels.npy` files.

- **Slow training / high CPU usage**  
  Prefer GPU when available and avoid running heavy dashboard processes in parallel with training.

- **Real-time script cannot find model**  
  Ensure `best_model.pth` exists at repo root, or update script arguments/path accordingly.

## Additional Documentation

- Real-time system details: `README_REALTIME.md`
- Testing details and interpretation: `TESTING_GUIDE.md`