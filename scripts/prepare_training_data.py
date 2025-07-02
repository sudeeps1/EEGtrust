import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eegtrust.config import SAMPLE_RATE, WINDOW_SIZE_SAMPLES, STRIDE_SAMPLES
from eegtrust.metadata import parse_seizure_summary
from eegtrust.data import load_eeg_data_chunked

# Directory containing all subject folders (e.g., chb01, chb02, ...)
DATA_ROOT = os.path.join('data', 'physionet.org', 'files', 'chbmit', '1.0.0')

# Subjects to use for training
subjects = [f'chb{str(i).zfill(2)}' for i in range(1, 25)]

# Create output directory
os.makedirs("prepared_data", exist_ok=True)

# Process each subject individually
for subject in subjects:
    print(f'\nProcessing {subject}...')
    subj_dir = os.path.join(DATA_ROOT, subject)
    summary_path = os.path.join(subj_dir, f'{subject}-summary.txt')
    
    if not os.path.exists(summary_path):
        print(f'Summary file not found for {subject}, skipping.')
        continue
    
    seizure_dict = parse_seizure_summary(summary_path)
    subject_windows = []
    subject_labels = []
    
    print(f'  Found {len(seizure_dict)} EDF files for {subject}')
    
    for i, edf_file in enumerate(seizure_dict.keys()):
        edf_path = os.path.join(subj_dir, edf_file)
        if not os.path.exists(edf_path):
            print(f'    File not found: {edf_file}')
            continue
        
        # Check file size - skip very large files that might cause issues
        file_size_mb = os.path.getsize(edf_path) / (1024 * 1024)
        if file_size_mb > 500:  # Skip files larger than 500MB
            print(f'    Skipping {edf_file} - file too large ({file_size_mb:.1f}MB)')
            continue
        
        print(f'  Processing {edf_file} ({i+1}/{len(seizure_dict)}) - {file_size_mb:.1f}MB...')
        seizure_intervals = seizure_dict[edf_file]
        print(f'    Found {len(seizure_intervals)} seizure intervals')
        
        try:
            # Use chunked loading to avoid memory errors for large files
            window_count = 0
            start_time = time.time()
            
            for window, label in load_eeg_data_chunked(
                edf_path, seizure_intervals, SAMPLE_RATE, chunk_size=50_000, 
                window_size=WINDOW_SIZE_SAMPLES, stride=STRIDE_SAMPLES):
                subject_windows.append(window)
                subject_labels.append(label)
                window_count += 1
                
                # Progress indicator every 1000 windows
                if window_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"    Generated {window_count} windows in {elapsed:.1f}s...")
            
            elapsed = time.time() - start_time
            print(f"    Completed {edf_file}: {window_count} windows in {elapsed:.1f}s")
            
        except Exception as e:
            print(f'    Error processing {edf_file}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    if subject_windows:
        # Save subject data
        subject_windows = np.stack(subject_windows)
        subject_labels = np.array(subject_labels)
        
        np.save(f'prepared_data/{subject}_windows.npy', subject_windows)
        np.save(f'prepared_data/{subject}_labels.npy', subject_labels)
        
        print(f'  Saved {subject}: {len(subject_windows)} windows')
    else:
        print(f'  No valid data for {subject}')

# Combine all subject data
print('\nCombining all subject data...')
all_windows = []
all_labels = []

for subject in subjects:
    windows_file = f'prepared_data/{subject}_windows.npy'
    labels_file = f'prepared_data/{subject}_labels.npy'
    
    if os.path.exists(windows_file) and os.path.exists(labels_file):
        windows = np.load(windows_file)
        labels = np.load(labels_file)
        all_windows.append(windows)
        all_labels.append(labels)
        print(f'  Added {subject}: {len(windows)} windows')

if all_windows:
    windows = np.concatenate(all_windows, axis=0)
    window_labels = np.concatenate(all_labels, axis=0)
    
    print(f'\nFinal dataset:')
    print(f'windows shape: {windows.shape}')
    print(f'labels shape: {window_labels.shape}')
    
    # Save combined data
    np.save('prepared_data/windows.npy', windows)
    np.save('prepared_data/window_labels.npy', window_labels)
    print('Saved combined windows and labels to prepared_data/')
else:
    print('No data was successfully processed!') 