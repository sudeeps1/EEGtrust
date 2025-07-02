import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

def extract_metadata(subject_id, metadata_df):
    row = metadata_df[metadata_df['subject_id'] == subject_id]
    if row.empty:
        return {'age': np.nan, 'sex': 'F', 'region': 'unknown'}
    return row.iloc[0].to_dict()

def bin_age(age):
    # All pediatric in CHB-MIT
    return 'child'

def one_hot_encode_region(region, all_regions=['unknown']):
    vec = np.zeros(len(all_regions))
    idx = all_regions.index(region)
    vec[idx] = 1
    return vec

def encode_metadata(metadata_row, all_regions=['unknown']):
    """Encode metadata as a fixed vector. For CHB-MIT, all pediatric, sex F, region unknown."""
    age_vec = np.array([1, 0, 0, 0])  # child
    region_vec = one_hot_encode_region(metadata_row['region'], all_regions)
    sex_vec = np.array([1])  # F
    return np.concatenate([age_vec, region_vec, sex_vec])

# Example output:
# {
#   'chb01_03.edf': [(2996, 3036)],
#   'chb01_16.edf': [(327, 420), (1862, 1963)],
#   ...
# }
def parse_seizure_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a CHB-MIT summary file and return a mapping from EEG file names to lists of (start, end) seizure intervals in seconds.
    """
    seizure_dict = {}
    current_file = None
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                current_file = line.split(':', 1)[1].strip()
                seizure_dict[current_file] = []
            elif line.startswith('Seizure Start Time:'):
                # Extract integer seconds
                start_str = line.split(':', 1)[1].strip().split()[0]
                start = int(start_str)
            elif line.startswith('Seizure End Time:'):
                end_str = line.split(':', 1)[1].strip().split()[0]
                end = int(end_str)
                if current_file:
                    seizure_dict[current_file].append((start, end))
    return seizure_dict


def get_seizure_intervals_for_file(summary_dict: Dict[str, List[Tuple[int, int]]], edf_file: str) -> List[Tuple[int, int]]:
    """
    Given a parsed summary dictionary and an EDF file name, return the list of (start, end) seizure intervals.
    """
    return summary_dict.get(edf_file, []) 