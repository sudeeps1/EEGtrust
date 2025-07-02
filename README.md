# Pediatric EEG Seizure Detection Pipeline

This project implements a full pipeline for pediatric EEG seizure detection using deep learning. It includes data preparation, preprocessing, model training, real-time inference, explainability, and a visualization dashboard.

## Features
- Data loading and preprocessing (filtering, normalization, artifact removal)
- Sliding window segmentation and labeling
- Metadata extraction and encoding
- Deep learning model (EEG feature extractor + metadata stream)
- Stratified subject split, metrics, early stopping, logging
- Real-time inference pipeline
- Explainability (saliency mapping)
- Streamlit dashboard for visualization
- ONNX export for edge deployment

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download EEG datasets (see `scripts/download_data.py`)

## Usage
- Preprocess and segment data: `python -m eegtrust.data`
- Train model: `python -m eegtrust.train`
- Run real-time inference: `python -m eegtrust.inference`
- Launch dashboard: `streamlit run dashboard/app.py`
- Export model to ONNX: `python scripts/export_onnx.py`

## Directory Structure
```
EEGtrust/
├── data/                # Downloaded EEG data
├── notebooks/           # Jupyter notebooks for EDA
├── eegtrust/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── segment.py
│   ├── metadata.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── explain.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── scripts/
│   ├── download_data.py
│   └── export_onnx.py
├── requirements.txt
└── README.md
```