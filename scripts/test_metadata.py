from eegtrust.metadata import parse_seizure_summary

summary_path = r"C:\Users\sudee\EEGtrust\data\physionet.org\files\chbmit\1.0.0\chb01\chb01-summary.txt"
seizure_dict = parse_seizure_summary(summary_path)
print(seizure_dict)
