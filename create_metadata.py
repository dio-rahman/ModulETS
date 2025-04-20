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

def guess_keturunan(filename):
    filename_lower = filename.lower()
    if "cina" in filename_lower:
        return "Cina"
    elif "jawa" in filename_lower:
        return "Jawa"
    elif "sunda" in filename_lower:
        return "Sunda"
    else:
        return "Unknown"

def guess_ekspresi(filename):
    filename_lower = filename.lower()
    for expr in ["senyum", "serius", "terkejut", "marah", "cemberut", "datar", "merem", "bingung"]:
        if expr in filename_lower:
            return expr
    return "Unknown"

def guess_posisi(filename):
    filename_lower = filename.lower()
    for pos in ["frontal", "miring", "profile"]:
        if pos in filename_lower:
            return pos
    return "Unknown"

def guess_pencahayaan(filename):
    filename_lower = filename.lower()
    for light in ["terang", "redup", "gelap"]:
        if light in filename_lower:
            return light
    return "Unknown"

def guess_jarak(filename):
    filename_lower = filename.lower()
    for dist in ["dekat", "sedang", "jauh"]:
        if dist in filename_lower:
            return dist
    return "Unknown"

with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    writer.writerow(["filename", "nama", "keturunan", "ekspresi", "posisi", "pencahayaan", "jarak"])
    for filename in files:
        nama = os.path.splitext(filename)[0]
        keturunan = guess_keturunan(filename)
        ekspresi = guess_ekspresi(filename)
        posisi = guess_posisi(filename)
        pencahayaan = guess_pencahayaan(filename)
        jarak = guess_jarak(filename)
        writer.writerow([filename, nama, keturunan, ekspresi, posisi, pencahayaan, jarak])

print(f"File metadata.csv has been created at {metadata_path}")
print(f"Found {len(files)} image files in the dataset directory")
print("Please review and edit the metadata values in the CSV file if needed.")