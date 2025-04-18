import os
    import csv

    dataset_path = "./dataset_wajah"

    metadata_path = os.path.join(dataset_path, "metadata.csv")

    allowed_extensions = (".jpg", ".jpeg", ".png")

    files = [
        f for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
        and f.lower().endswith(allowed_extensions)
    ]

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "nama", "keturunan"])
        for filename in files:
            nama = os.path.splitext(filename)[0]
            keturunan = "Unknown"
            writer.writerow([filename, nama, keturunan])

    print(f"File metadata.csv telah dibuat di {metadata_path}")