from fastapi import FastAPI, Form, HTTPException
import os
import csv
import cv2
from typing import Optional
from app import detect_faces, extract_face, face_embedding_model, db

app = FastAPI()

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def load_metadata(metadata_path: Optional[str]) -> dict:
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                filename = row.get('filename', '')
                if filename:
                    keturunan = row.get('keturunan', '').capitalize()
                    if keturunan not in ["Jawa", "Sunda", "Cina"]:
                        keturunan = 'Unknown'
                    metadata[filename] = {
                        'nama': row.get('nama', os.path.splitext(filename)[0]),
                        'keturunan': keturunan
                    }
    return metadata

def process_image(file_path: str, metadata: dict, filename: str) -> tuple[bool, str]:
    if filename in metadata:
        nama = metadata[filename].get('nama', os.path.splitext(filename)[0])
        keturunan = metadata[filename].get('keturunan', 'Unknown')
    else:
        nama = os.path.splitext(filename)[0]
        keturunan = 'Unknown'
        filename_lower = filename.lower()
        if 'cina' in filename_lower:
            keturunan = 'Cina'
        elif 'jawa' in filename_lower:
            keturunan = 'Jawa'
        elif 'sunda' in filename_lower:
            keturunan = 'Sunda'

    try:
        image = cv2.imread(file_path)
        if image is None:
            return False, "Could not read image"

        faces = detect_faces(image)
        if not faces:
            return False, "No faces detected"

        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]

        face_img = extract_face(image, main_face)
        if face_img is None:
            return False, "Could not extract face"

        embedding = face_embedding_model.get_embedding(face_img)

        if keturunan not in ["Jawa", "Sunda", "Cina", "Unknown"]:
            keturunan = "Unknown"

        db.add_person(nama, keturunan, embedding)
        return True, ""
    except Exception as e:
        return False, str(e)

@app.post("/import_dataset")
async def import_dataset_endpoint(dataset_path: str = Form(...), metadata_path: Optional[str] = Form(None)):
    try:
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset path {dataset_path} does not exist")

        metadata = load_metadata(metadata_path)
        imported_count = 0
        failed_files = []

        for filename in os.listdir(dataset_path):
            if not filename.lower().endswith(ALLOWED_EXTENSIONS):
                continue

            file_path = os.path.join(dataset_path, filename)
            success, error = process_image(file_path, metadata, filename)
            if success:
                imported_count += 1
            else:
                failed_files.append(f"{filename}: {error}")

        return {
            "status": "success",
            "imported_count": imported_count,
            "failed_files": failed_files,
            "message": f"Imported {imported_count} faces. {len(failed_files)} files failed."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing dataset: {str(e)}")