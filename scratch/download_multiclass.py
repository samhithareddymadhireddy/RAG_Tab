import sys
sys.path.insert(0, ".")
from src.data.download_openml import download_dataset

print("Downloading multiclass dermatology...")
download_dataset("dermatology", as_multiclass=True)

print("Downloading multiclass vehicle...")
download_dataset("vehicle", as_multiclass=True)

print("Downloading multiclass segment...")
download_dataset("segment", as_multiclass=True)

print("Done downloading multiclass datasets!")
