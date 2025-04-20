import os
import csv
import sys

dataset_path = "./dataset_wajah"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    print(f"Created dataset directory at {dataset_path}")

metadata_path = os.path.join(dataset_path, "metadata.csv")

allowed_extensions = (".jpg", ".jpeg", ".png")

files = [
    f for f in os.listdir(dataset_path)
    if os.path.isfile(os.path.join(dataset_path, f))
    and f.lower().endswith(allowed_extensions)
]

if not files:
    print(f"Warning: No image files found in {dataset_path}")
    print("You can add image files to this directory and run this script again.")

with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "nama", "keturunan"])
    for filename in files:
        nama = os.path.splitext(filename)[0]
        keturunan = "Unknown"
        writer.writerow([filename, nama, keturunan])

print(f"File metadata.csv has been created at {metadata_path}")
print(f"Found {len(files)} image files in the dataset directory")