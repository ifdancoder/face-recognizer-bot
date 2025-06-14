import os
import json
import face_recognition
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
import cv2
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "face_encodings.json"

# Create directory if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def load_known_faces():
    """Load known face encodings from JSON file"""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_known_faces(encodings):
    """Save face encodings to JSON file"""
    with open(ENCODINGS_FILE, 'w') as f:
        json.dump(encodings, f)

@app.post("/train")
async def train_face(name: str = Form(...), images: List[UploadFile] = File(...)):
    """Train the model with multiple images for a single name"""
    try:
        known_faces = load_known_faces()
        if name not in known_faces:
            known_faces[name] = []

        successful_encodings = 0
        for image in images:
            # Read and save the image
            contents = await image.read()
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            with open(image_path, "wb") as f:
                f.write(contents)
            
            # Load the image and get face encoding
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                os.remove(image_path)
                continue
            
            # Add all face encodings from the image
            for encoding in face_encodings:
                known_faces[name].append(encoding.tolist())
                successful_encodings += 1
        
        if successful_encodings == 0:
            raise HTTPException(status_code=400, detail="No faces detected in any of the provided images")
        
        save_known_faces(known_faces)
        return {
            "message": f"Successfully trained {successful_encodings} face encodings for {name}",
            "total_encodings": len(known_faces[name])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize_face(images: List[UploadFile] = File(...)):
    """Recognize faces in multiple uploaded images and return unique faces"""
    try:
        all_face_encodings = []
        all_face_locations = []

        for image in images:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_image)

            if not face_locations:
                continue

            for loc in face_locations:
                if len(loc) != 4:
                    continue

            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if face_encodings:
                for enc in face_encodings:
                    if enc.shape != (128,):
                        continue
                all_face_encodings.extend(face_encodings)
                all_face_locations.extend(face_locations)

        if not all_face_encodings:
            return {"message": "No faces detected in any of the images"}

        known_faces = load_known_faces()
        unique_faces = set()
        results = []

        for face_encoding in all_face_encodings:
            if not isinstance(face_encoding, np.ndarray):
                face_encoding = np.array(face_encoding)
            if face_encoding.ndim == 1:
                face_encoding = face_encoding.reshape(1, -1)

            matches = []
            for name, known_encodings in known_faces.items():
                known_encodings_list = np.array([np.array(enc).flatten() for enc in known_encodings])

                if known_encodings_list.ndim == 1:
                    known_encodings_list = known_encodings_list.reshape(1, -1)

                matches_for_name = face_recognition.compare_faces(
                    known_encodings_list,
                    face_encoding,
                    tolerance=0.6
                )
                if any(matches_for_name):
                    matches.append(name)

            if matches:
                unique_faces.update(matches)
                results.append({"names": matches})
            else:
                results.append({"names": ["Unknown"]})

        return {
            "results": results,
            "unique_faces": list(unique_faces),
            "total_faces_detected": len(all_face_encodings)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/known-faces")
async def get_known_faces():
    """Get list of all known faces with their encoding counts"""
    known_faces = load_known_faces()
    return {
        "faces": [
            {
                "name": name,
                "encoding_count": len(encodings)
            }
            for name, encodings in known_faces.items()
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 