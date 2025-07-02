import numpy as np

labels = []
for subj in ['chb01', 'chb02']:
    labels.append(np.load(f'prepared_data/{subj}_labels.npy'))
y = np.concatenate(labels, axis=0)

unique, counts = np.unique(y, return_counts=True)
print("Class distribution in test set:")
for label, count in zip(unique, counts):
    print(f"  Label {label}: {count}") 