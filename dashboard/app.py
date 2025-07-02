import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Pediatric EEG Seizure Detection Dashboard")
    # TODO: Load EEG, predictions, metadata, saliency
    st.write("EEG trace, prediction score, metadata, and saliency map will appear here.")
    # Example EEG plot
    eeg = np.random.randn(19, 512)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eeg.T)
    st.pyplot(fig)

if __name__ == "__main__":
    main() 