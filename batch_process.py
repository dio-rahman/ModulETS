import os
import cv2
import csv
import argparse
from app import detect_faces, extract_face, face_embedding_model, ethnicity_model

ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def process_folder(folder_path, output_csv):
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    results = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(ALLOWED_EXTENSIONS):
            continue
        
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read {filename}")
            continue
        
        faces = detect_faces(image)
        if not faces:
            print(f"Warning: No faces detected in {filename}")
            continue
        
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        main_face = faces[0]
        face_img = extract_face(image, main_face)
        if face_img is None:
            print(f"Warning: Could not extract face from {filename}")
            continue
        
        expression = detect_expression(face_img)
        ethnicity_predictions = ethnicity_model.predict(face_img)
        dominant_ethnicity = max(ethnicity_predictions.items(), key=lambda x: x[1])[0]
        
        results.append({
            "filename": filename,
            "expression": expression,
            "dominant_keturunan": dominant_ethnicity,
            "predictions": ";".join([f"{k}:{v:.2f}" for k, v in ethnicity_predictions.items()])
        })
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["filename", "expression", "dominant_keturunan", "predictions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processed {len(results)} images. Report saved to {output_csv}")

def detect_expression(face_img):
    try:
        from deepface import DeepFace
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images for face detection and ethnicity classification.")
    parser.add_argument("folder_path", help="Path to the folder containing images")
    parser.add_argument("output_csv", help="Path to the output CSV report file")
    args = parser.parse_args()
    
    process_folder(args.folder_path, args.output_csv)