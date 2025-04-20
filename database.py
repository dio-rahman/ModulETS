import os
import json
import numpy as np
import uuid

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
                similarity = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
                if similarity > max_similarity:
                    max_similarity = similarity
            
            if max_similarity >= threshold:
                results.append({
                    "person_id": person_id,
                    "nama": data["nama"],
                    "keturunan": data["keturunan"],
                    "similarity": float(max_similarity)
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results