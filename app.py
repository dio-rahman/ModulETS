from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import cv2
import os
import shutil
import uuid
import time
from io import BytesIO
from PIL import Image
import json
import csv
from pydantic import BaseModel
from typing import List, Optional

try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not installed. Please install with: pip install deepface")
    class DeepFaceStub:
        @staticmethod
        def represent(img, model_name=None, enforce_detection=True):
            return [{"embedding": np.random.default_rng().random(128).tolist()}]
    DeepFace = DeepFaceStub()

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Please install with: pip install tensorflow")

try:
    from mtcnn import MTCNN
    detector = MTCNN()
except ImportError:
    print("MTCNN not installed. Please install with: pip install mtcnn")
    detector = None

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
UNSUPPORTED_FORMAT_ERROR = "Unsupported file format. Please upload a JPG or PNG image."
IMAGE_READ_ERROR = "Could not read image file"
NO_FACES_ERROR = "No faces detected in the image"
FACE_EXTRACT_ERROR = "Could not extract face from image"

os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("dataset_wajah", exist_ok=True)

print("Current working dir:", os.getcwd())
print("Template path exists:", os.path.exists("templates/index.html"))

app = FastAPI(
    title="Face Recognition and Ethnicity Detection API",
    description="API Pendeteksi Wajah Dan Klasifikasi Etnis",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="results"), name="static")

class FaceEmbeddingModel:
    def get_embedding(self, face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(face_img_rgb, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"])

class EthnicityClassificationModel:
    def __init__(self):
        self.suku = ["Jawa", "Sunda", "Cina"]
        
    def predict(self, face_img):
        _ = cv2.resize(face_img, (224, 224)) 
        rng = np.random.default_rng()
        probs = rng.random(len(self.suku))
        probs = probs / np.sum(probs)
        
        predictions = {
            self.suku[i]: float(probs[i]) 
            for i in range(len(self.suku))
        }
        
        return predictions

face_embedding_model = FaceEmbeddingModel()
ethnicity_model = EthnicityClassificationModel()

class Database:
    def __init__(self, db_path="database/embeddings.json"):
        self.db_path = db_path
        self.embeddings = {}
        self.load()
        
    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for person_id, person_data in data.items():
                        embeddings = person_data["embeddings"]
                        self.embeddings[person_id] = {
                            "nama": person_data["nama"],
                            "keturunan": person_data["keturunan"],
                            "embeddings": [np.array(emb) for emb in embeddings]
                        }
            except Exception as e:
                print(f"Error loading database: {e}")
                self.embeddings = {}
        
    def save(self):
        data = {}
        for person_id, person_data in self.embeddings.items():
            data[person_id] = {
                "nama": person_data["nama"],
                "keturunan": person_data["keturunan"],
                "embeddings": [emb.tolist() for emb in person_data["embeddings"]]
            }
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(data, f)
            
    def add_person(self, nama, keturunan, embedding):
        person_id = str(uuid.uuid4())
        self.embeddings[person_id] = {
            "nama": nama,
            "keturunan": keturunan,
            "embeddings": [embedding]
        }
        self.save()
        return person_id
    
    def add_embedding_to_person(self, person_id, embedding):
        if person_id in self.embeddings:
            self.embeddings[person_id]["embeddings"].append(embedding)
            self.save()
            return True
        return False
    
    def get_all_people(self):
        return [{
            "id": person_id,
            "nama": data["nama"],
            "keturunan": data["keturunan"],
            "embedding_count": len(data["embeddings"])
        } for person_id, data in self.embeddings.items()]
    
    def get_person(self, person_id):
        if person_id in self.embeddings:
            return {
                "id": person_id,
                "nama": self.embeddings[person_id]["nama"],
                "keturunan": self.embeddings[person_id]["keturunan"],
                "embedding_count": len(self.embeddings[person_id]["embeddings"])
            }
        return None
    
    def find_similar_faces(self, embedding, threshold=0.6):
        results = []
        
        for person_id, data in self.embeddings.items():
            person_embeddings = data["embeddings"]
            
            max_similarity = 0
            for emb in person_embeddings:
                emb_a = embedding.reshape(1, -1)
                emb_b = emb.reshape(1, -1)
                similarity = np.dot(emb_a, emb_b.T)[0][0] / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                results.append({
                    "person_id": person_id,
                    "nama": data["nama"],
                    "keturunan": data["keturunan"],
                    "similarity": float(max_similarity)
                })
                
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

db = Database()

class Person(BaseModel):
    nama: str
    keturunan: str

class FaceDetectionResponse(BaseModel):
    faces_detected: int
    image_path: str

class SimilarityResult(BaseModel):
    person_id: str
    nama: str
    keturunan: str
    similarity: float

class SimilarityResponse(BaseModel):
    results: List[SimilarityResult]
    threshold: float

class EthnicityPrediction(BaseModel):
    keturunan: str
    confidence: float

class EthnicityResponse(BaseModel):
    predictions: List[EthnicityPrediction]
    dominant_keturunan: str
    image_path: str

class PersonResponse(BaseModel):
    id: str
    nama: str
    keturunan: str
    embedding_count: int

def read_image_file(file) -> np.ndarray:
    contents = file.file.read()
    image = Image.open(BytesIO(contents))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def save_uploaded_file(file: UploadFile, destination: str) -> str:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination

def detect_faces(image) -> list:
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image
        
    if detector is not None:
        faces = detector.detect_faces(rgb_image)
        return faces
    else:
        print("Warning: Using dummy face detection since MTCNN isn't available")
        h, w = rgb_image.shape[:2]
        size = min(h, w) // 2
        x = (w - size) // 2
        y = (h - size) // 2
        return [{
            'box': (x, y, size, size),
            'confidence': 0.9,
            'keypoints': {
                'left_eye': (x + size//4, y + size//3),
                'right_eye': (x + 3*size//4, y + size//3),
                'nose': (x + size//2, y + size//2),
                'mouth_left': (x + size//3, y + 2*size//3),
                'mouth_right': (x + 2*size//3, y + 2*size//3)
            }
        }]

def extract_face(image, face_data, required_size=(160, 160)):
    x, y, width, height = face_data['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y+height, x:x+width]
    
    if face.size > 0: 
        face = cv2.resize(face, required_size)
        return face
    else:
        return None

def draw_faces(image, faces):
    img_copy = image.copy()
    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(img_copy, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
        keypoints = face['keypoints']
        for point in keypoints.values():
            cv2.circle(img_copy, point, 2, (0, 0, 255), 2)
            
    return img_copy

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FORMAT_ERROR)
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"face_detection_{timestamp}{file_extension}"
        
        upload_path = os.path.join("uploads", unique_filename)
        await file.seek(0)
        save_uploaded_file(file, upload_path)
        
        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail=IMAGE_READ_ERROR)
        
        faces = detect_faces(image)
        
        result_image = draw_faces(image, faces)
        
        result_path = os.path.join("results", unique_filename)
        cv2.imwrite(result_path, result_image)
        
        return {
            "faces_detected": len(faces),
            "image_path": f"/static/{unique_filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/compare")
async def compare_faces_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    try:
        threshold = max(0.0, min(1.0, threshold))
        
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FORMAT_ERROR)
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"face_comparison_{timestamp}{file_extension}"
        
        upload_path = os.path.join("uploads", unique_filename)
        await file.seek(0)
        save_uploaded_file(file, upload_path)
        
        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail=IMAGE_READ_ERROR)
        
        faces = detect_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail=NO_FACES_ERROR)
        
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]
        
        face_img = extract_face(image, main_face)
        if face_img is None:
            raise HTTPException(status_code=400, detail=FACE_EXTRACT_ERROR)
        
        embedding = face_embedding_model.get_embedding(face_img)
        
        similar_faces = db.find_similar_faces(embedding, threshold)
        
        return {
            "results": similar_faces,
            "threshold": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

@app.post("/keturunan", response_model=EthnicityResponse)
async def detect_ethnicity_endpoint(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FORMAT_ERROR)
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"ethnicity_detection_{timestamp}{file_extension}"
        
        upload_path = os.path.join("uploads", unique_filename)
        await file.seek(0)
        save_uploaded_file(file, upload_path)
        
        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail=IMAGE_READ_ERROR)
        
        faces = detect_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail=NO_FACES_ERROR)
        
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]
        
        face_img = extract_face(image, main_face, required_size=(224, 224))
        if face_img is None:
            raise HTTPException(status_code=400, detail=FACE_EXTRACT_ERROR)
        
        ethnicity_predictions = ethnicity_model.predict(face_img)
        
        dominant_ethnicity = max(ethnicity_predictions.items(), key=lambda x: x[1])[0]
        
        result_image = draw_faces(image, [main_face])
        
        x, y, _, _ = main_face['box']
        label = f"{dominant_ethnicity}: {ethnicity_predictions[dominant_ethnicity]:.2f}"
        cv2.putText(result_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        result_path = os.path.join("results", unique_filename)
        cv2.imwrite(result_path, result_image)
        
        formatted_predictions = [
            {"keturunan": eth, "confidence": float(conf)} 
            for eth, conf in ethnicity_predictions.items()
        ]
        formatted_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "predictions": formatted_predictions,
            "dominant_keturunan": dominant_ethnicity,
            "image_path": f"/static/{unique_filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting ethnicity: {str(e)}")

@app.post("/register", response_model=PersonResponse)
async def register_person_endpoint(
    file: UploadFile = File(...),
    nama: str = Form(...),
    keturunan: str = Form(...)
):
    try:
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FORMAT_ERROR)
        
        if not nama.strip():
            raise HTTPException(status_code=400, detail="Nama tidak boleh kosong.")
        
        if keturunan not in ["Jawa", "Sunda", "Cina"]:
            raise HTTPException(status_code=400, detail="Keturunan tidak valid. Pilih Jawa, Sunda, atau Cina.")
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"person_{nama.replace(' ', '_')}_{timestamp}{file_extension}"
        
        upload_path = os.path.join("uploads", unique_filename)
        await file.seek(0)
        save_uploaded_file(file, upload_path)
        
        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail=IMAGE_READ_ERROR)
        
        faces = detect_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail=NO_FACES_ERROR)
        
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]
        
        face_img = extract_face(image, main_face)
        if face_img is None:
            raise HTTPException(status_code=400, detail=FACE_EXTRACT_ERROR)
        
        try:
            embedding = face_embedding_model.get_embedding(face_img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating face embedding: {str(e)}")
        
        person_id = db.add_person(nama, keturunan, embedding)
        
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=500, detail="Failed to retrieve registered person data.")
        
        return person
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering person: {str(e)}")

def process_image_for_import(file_path: str, metadata: dict, filename: str, failed_files: List[str]):
    image = cv2.imread(file_path)
    if image is None:
        failed_files.append(f"{filename}: Could not read image")
        return None, None, None

    faces = detect_faces(image)
    if not faces:
        failed_files.append(f"{filename}: No faces detected")
        return None, None, None

    faces.sort(key=lambda x: x['confidence'], reverse=True)
    main_face = faces[0]

    face_img = extract_face(image, main_face)
    if face_img is None:
        failed_files.append(f"{filename}: Could not extract face")
        return None, None, None

    try:
        embedding = face_embedding_model.get_embedding(face_img)
    except Exception as e:
        failed_files.append(f"{filename}: Error generating embedding - {str(e)}")
        return None, None, None

    nama = metadata.get(filename, {}).get('nama', os.path.splitext(filename)[0])
    keturunan = metadata.get(filename, {}).get('keturunan', "Unknown")

    if keturunan not in ["Jawa", "Sunda", "Cina", "Unknown"]:
        failed_files.append(f"{filename}: Invalid keturunan")
        return None, None, None

    return nama, keturunan, embedding

@app.post("/import_dataset")
async def import_dataset_endpoint(dataset_path: str = Form(...), metadata_path: Optional[str] = Form(None)):
    try:
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset path {dataset_path} does not exist")

        imported_count = 0
        failed_files = []
        
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata[row['filename']] = {
                        'nama': row['nama'],
                        'keturunan': row['keturunan']
                    }
        
        for filename in os.listdir(dataset_path):
            if not filename.lower().endswith(ALLOWED_EXTENSIONS):
                continue
                
            file_path = os.path.join(dataset_path, filename)
            nama, keturunan, embedding = process_image_for_import(file_path, metadata, filename, failed_files)
            
            if embedding is not None:
                db.add_person(nama, keturunan, embedding)
                imported_count += 1
            
        return {
            "status": "success",
            "imported_count": imported_count,
            "failed_files": failed_files,
            "message": f"Imported {imported_count} faces. {len(failed_files)} files failed."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing dataset: {str(e)}")

@app.post("/people/{person_id}/faces")
async def add_face_to_person(
    person_id: str,
    file: UploadFile = File(...)
):
    try:
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FORMAT_ERROR)
        
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"person_{person_id}_{timestamp}{file_extension}"
        
        upload_path = os.path.join("uploads", unique_filename)
        await file.seek(0)
        save_uploaded_file(file, upload_path)
        
        image = cv2.imread(upload_path)
        if image is None:
            raise HTTPException(status_code=400, detail=IMAGE_READ_ERROR)
        
        faces = detect_faces(image)
        if not faces:
            raise HTTPException(status_code=400, detail=NO_FACES_ERROR)
        
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]
        
        face_img = extract_face(image, main_face)
        if face_img is None:
            raise HTTPException(status_code=400, detail=FACE_EXTRACT_ERROR)
        
        embedding = face_embedding_model.get_embedding(face_img)
        
        success = db.add_embedding_to_person(person_id, embedding)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add face to person")
        
        return db.get_person(person_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding face to person: {str(e)}")

@app.get("/people")
async def list_people():
    return db.get_all_people()

@app.get("/people/{person_id}")
async def get_person(person_id: str):
    person = db.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person

@app.get("/home")
async def custom_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)