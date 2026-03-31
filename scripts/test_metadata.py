from eegtrust.metadata import parse_seizure_summary
import os
import sys

summary_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "data", "physionet.org", "files", "chbmit", "1.0.0", "chb01", "chb01-summary.txt"
)
seizure_dict = parse_seizure_summary(summary_path)
print(seizure_dict)
