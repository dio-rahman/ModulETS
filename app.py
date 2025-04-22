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
from deepface import DeepFace

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
EXPRESSION_DETECT_ERROR = "Could not detect facial expression"

os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("dataset_wajah", exist_ok=True)

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
                # Pastikan dimensi sesuai dengan mengecek shape dan menyesuaikan
                try:
                    # Flatten kedua embedding untuk memastikan dimensi yang benar
                    emb_flat = emb.flatten()
                    embedding_flat = embedding.flatten()
                    
                    # Hitung dot product
                    dot_product = np.dot(emb_flat, embedding_flat)
                    
                    # Hitung norm (magnitude) dari kedua vektor
                    norm_emb = np.linalg.norm(emb_flat)
                    norm_embedding = np.linalg.norm(embedding_flat)
                    
                    # Hitung cosine similarity
                    if norm_emb > 0 and norm_embedding > 0:
                        similarity = dot_product / (norm_emb * norm_embedding)
                        max_similarity = max(max_similarity, similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
            
            if max_similarity >= threshold:
                results.append({
                    "person_id": person_id,
                    "nama": data["nama"],
                    "keturunan": data["keturunan"],
                    "similarity": float(max_similarity)
                })
                    
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

class FaceEmbeddingModel:
    def get_embedding(self, face_img):
        try:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(face_img_rgb, model_name="Facenet", enforce_detection=False)
            
            # Pastikan embedding selalu memiliki bentuk yang konsisten
            emb_array = np.array(embedding[0]["embedding"])
            
            # Debug info
            print(f"Embedding shape: {emb_array.shape}")
            
            # Pastikan bentuknya 1D atau reshape ke vektor 1D jika tidak
            if len(emb_array.shape) > 1:
                emb_array = emb_array.flatten()
                
            return emb_array
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            # Fallback ke dummy embedding dengan ukuran yang benar
            return np.zeros(128)  # Sesuaikan dengan ukuran embedding yang diharapkan

class EthnicityClassificationModel:
    def __init__(self):
        self.suku = ["Jawa", "Sunda", "Cina"]
        self.database = Database()
        
    def predict(self, face_img):
        # Mendapatkan embedding dari wajah yang diunggah
        embedding = face_embedding_model.get_embedding(face_img)
        
        # Mencari wajah yang mirip dalam database
        similar_faces = self.database.find_similar_faces(embedding, threshold=0.5)
        
        # Jika tidak ada wajah yang mirip, gunakan pendekatan dummy
        if not similar_faces:
            print("No similar faces found, using random prediction")
            # Fallback ke pendekatan dummy jika tidak ada kecocokan
            rng = np.random.default_rng(seed=42)
            probs = rng.random(len(self.suku))
            probs = probs / np.sum(probs)
            
            predictions = {
                self.suku[i]: float(probs[i]) 
                for i in range(len(self.suku))
            }
            return predictions
        
        # Menghitung distribusi keturunan dari wajah-wajah yang mirip
        ethnicity_counts = {"Jawa": 0, "Sunda": 0, "Cina": 0}
        total_similarity = 0
        
        for face in similar_faces:
            similarity = face["similarity"]
            keturunan = face["keturunan"]
            
            if keturunan in ethnicity_counts:
                ethnicity_counts[keturunan] += similarity
                total_similarity += similarity
        
        # Normalisasi distribusi
        predictions = {}
        if total_similarity > 0:
            for ethnicity, count in ethnicity_counts.items():
                predictions[ethnicity] = float(count / total_similarity)
        else:
            # Fallback jika tidak ada total similarity (tidak seharusnya terjadi)
            for ethnicity in ethnicity_counts:
                predictions[ethnicity] = 1.0 / len(ethnicity_counts)
                
        return predictions

face_embedding_model = FaceEmbeddingModel()
ethnicity_model = EthnicityClassificationModel()

db = Database()

class Person(BaseModel):
    nama: str
    keturunan: str

class FaceDetectionResponse(BaseModel):
    faces_detected: int
    image_path: str
    expressions: List[str]
    lighting: str

class SimilarityResult(BaseModel):
    person_id: str
    nama: str
    keturunan: str
    similarity: float
    expression: str

class SimilarityResponse(BaseModel):
    results: List[SimilarityResult]
    threshold: float

class EthnicityPrediction(BaseModel):
    keturunan: str
    confidence: float

class EthnicityResponse(BaseModel):
    predictions: List[EthnicityPrediction]
    dominant_keturunan: str
    lighting: str
    expression: str
    image_path: str

class PersonResponse(BaseModel):
    id: str
    nama: str
    keturunan: str
    embedding_count: int
    expression: Optional[str] = None

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

def draw_faces(image, faces, expressions=None, include_expression=True):
    img_copy = image.copy()
    
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(img_copy, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
        keypoints = face['keypoints']
        for point in keypoints.values():
            cv2.circle(img_copy, point, 2, (0, 0, 255), 2)
        
        if include_expression and expressions and i < len(expressions):
            label = f"Expression: {expressions[i]}"
            cv2.putText(img_copy, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return img_copy

def detect_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 100:
        return "Redup"
    else:
        return "Terang"

def detect_expression(face_img):
    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_map = {
            'happy': 'Senyum',
            'sad': 'Sedih',
            'angry': 'Marah',
            'surprise': 'Terkejut',
            'neutral': 'Datar',
            'fear': 'Serius',
            'disgust': 'Serius'
        }
        return emotion_map.get(dominant_emotion, 'Unknown')
    except Exception as e:
        print(f"Error detecting expression: {e}")
        return "Unknown"

@app.on_event("startup")
async def startup_event():
    # Import dataset dari ModulETS/dataset_wajah jika tersedia
    dataset_path = "ModulETS/dataset_wajah"
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    
    if os.path.exists(dataset_path) and os.path.exists(metadata_path):
        print(f"Importing dataset from {dataset_path}")
        metadata = load_metadata(metadata_path)
        
        imported_count = 0
        for filename in os.listdir(dataset_path):
            if not filename.lower().endswith(ALLOWED_EXTENSIONS):
                continue
                
            file_path = os.path.join(dataset_path, filename)
            success, error = process_image(file_path, metadata, filename)
            if success:
                imported_count += 1
        
        print(f"Imported {imported_count} faces from dataset")

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
        if not faces:
            raise HTTPException(status_code=400, detail=NO_FACES_ERROR)
        
        expressions = []
        for face_data in faces:
            face_img = extract_face(image, face_data)
            if face_img is not None:
                expression = detect_expression(face_img)
                expressions.append(expression)
            else:
                expressions.append("Unknown")
        
        lighting = detect_lighting(image)
        
        result_image = draw_faces(image, faces, expressions, include_expression=True)
        
        result_path = os.path.join("results", unique_filename)
        cv2.imwrite(result_path, result_image)
        
        return {
            "faces_detected": len(faces),
            "image_path": f"/static/{unique_filename}",
            "expressions": expressions,
            "lighting": lighting
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

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
        
        # Dapatkan embedding untuk perbandingan
        embedding = face_embedding_model.get_embedding(face_img)
        
        # Cari wajah serupa untuk referensi
        similar_faces = db.find_similar_faces(embedding, threshold=0.5)
        
        # Dapatkan prediksi keturunan
        ethnicity_predictions = ethnicity_model.predict(face_img)
        
        dominant_ethnicity = max(ethnicity_predictions.items(), key=lambda x: x[1])[0]
        
        lighting = detect_lighting(image)
        expression = detect_expression(face_img)
        
        result_image = draw_faces(image, [main_face])
        
        x, y, _, _ = main_face['box']
        ethnicity_label = f"{dominant_ethnicity}: {ethnicity_predictions[dominant_ethnicity]:.2f}"
        cv2.putText(result_image, ethnicity_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        expression_label = f"Expression: {expression}"
        cv2.putText(result_image, expression_label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add similar faces information if available
        if similar_faces:
            similar_label = f"Similar to: {similar_faces[0]['nama']} ({similar_faces[0]['similarity']:.2f})"
            cv2.putText(result_image, similar_label, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
            "lighting": lighting,
            "expression": expression,
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
        
        expression = detect_expression(face_img)
        
        person_id = db.add_person(nama, keturunan, embedding)
        
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=500, detail="Failed to retrieve registered person data.")
        
        person["expression"] = expression
        
        return person
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering person: {str(e)}")

def load_metadata(metadata_path):
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

def process_image(file_path, metadata, filename):
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
        
        expression = detect_expression(face_img)
        
        success = db.add_embedding_to_person(person_id, embedding)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add face to person")
        
        person_info = db.get_person(person_id)
        person_info["expression"] = expression
        
        return person_info
        
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