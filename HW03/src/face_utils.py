import face_recognition
import numpy as np
from PIL import Image
import os

def load_known_faces(known_faces_dir="known_faces"):
    known_encodings = []
    known_names = []
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        return known_encodings, known_names

    for img_name in os.listdir(known_faces_dir):
        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(known_faces_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_encodings.append(face_encodings[0])
                name = os.path.splitext(img_name)[0]
                known_names.append(name)
    return known_encodings, known_names

def detect_faces(image: Image.Image):
    img_np = np.array(image.convert("RGB"))
    face_locations = face_recognition.face_locations(img_np)
    return face_locations

def get_face_encodings(image: Image.Image, face_locations):
    img_np = np.array(image.convert("RGB"))
    encodings = face_recognition.face_encodings(img_np, face_locations)
    return encodings

def recognize_faces(face_encodings, known_encodings, known_names, tolerance=0.6):
    results = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if len(face_distances) > 0:
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = known_names[best_match_idx]
        results.append(name)
    return results
