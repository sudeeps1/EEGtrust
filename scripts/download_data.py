import os
import requests
from zipfile import ZipFile

CHBMIT_URL = "https://physionet.org/static/published-projects/chbmit/chbmit-edf/1.0.0/chbmit-edf.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'chbmit')
ZIP_PATH = os.path.join(DATA_DIR, 'chbmit-edf.zip')

os.makedirs(DATA_DIR, exist_ok=True)

def download_chbmit():
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading CHB-MIT dataset to {ZIP_PATH}...")
        r = requests.get(CHBMIT_URL, stream=True)
        with open(ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("CHB-MIT zip already exists.")
    print("Extracting...")
    with ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    download_chbmit()
    print("\nFor TUH EEG Seizure Corpus, please register and download manually from https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml") 