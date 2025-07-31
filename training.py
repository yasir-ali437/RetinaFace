# encode_faces.py
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

def create_driver_embeddings(dataset_path="/home/adlytic/Yasir Adlytic/Dataset/Driver_Dataset_145_Driver_Original_31_July/", save_path="retinaface_driver_embeddings_31_July.pkl"):
    app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
    app.prepare(ctx_id=0)  # GPU: 0, CPU: -1

    embeddings = []
    names = []

    for driver_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, driver_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            # Get image dimensions
            height, width, _ = img.shape
            mid = width // 2

            if width > 1000 :
                # Crop left half
                left_half = img[:, :mid]
            else:
                left_half = img
            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                continue

            faces = app.get(image_rgb)
            if len(faces) == 0:
                continue

            face = faces[0]  # assume first face
            embeddings.append(face.embedding)
            names.append(driver_name)

    data = {"embeddings": np.array(embeddings), "names": names}
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(names)} embeddings to {save_path}")

# Run this to generate the database
if __name__ == "__main__":
    create_driver_embeddings()
