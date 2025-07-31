# inference.py
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# import face_recognition
import os
from openpyxl import Workbook

dataset_path = "/home/adlytic/Yasir Adlytic/Dataset/Driver_Dataset_145_Driver_Original_31_July/"
# /home/adlytic/Yasir Adlytic/Dataset/Driver_Faces_23_July/
DATABASE_PATH = "retinaface_driver_embeddings_31_July.pkl"

# Load existing data
if os.path.exists(DATABASE_PATH):
    with open(DATABASE_PATH, "rb") as f:
        db_data = pickle.load(f)
    embeddings_db = list(db_data["embeddings"])
    names_db = list(db_data["names"])
else:
    embeddings_db = []
    names_db = []

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1

def save_embeddings():
    with open(DATABASE_PATH, "wb") as f:
        pickle.dump({
            "embeddings": np.array(embeddings_db),
            "names": names_db
        }, f)
    print("Saved updated embeddings database.")

def variance_of_laplacian(image):
    # Measures image sharpness (higher is sharper)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def recognize_face(frame_faces_list, threshold=0.42):
    face_emb = frame_faces_list[0]
    frame_list = frame_faces_list[1]
    faces_list = frame_faces_list[2]
    drivers_count = len(os.listdir(dataset_path))
    similarities = cosine_similarity(face_emb, embeddings_db)[0]

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        name = names_db[best_idx]
        return str(name)
    else:
        name = "Unknown"
        print("Saving frames...for unknown driver")

        for i, f in enumerate(frame_list):
            if not os.path.isdir(os.path.join(dataset_path,f'd{drivers_count+1}')):
                os.makedirs(os.path.join(dataset_path,f'd{drivers_count+1}'))
            filename = os.path.join(dataset_path,f'd{drivers_count+1}', f"frame_{i}.jpg")
            cv2.imwrite(filename, f)
            # print(f"Saved {filename}")

        for i, f in enumerate(faces_list):
            embeddings_db.append(f)
            names_db.append(str(f'd{drivers_count+1}'))
        
        return name

def select_best_frame( video_path ):

    # Initialize RetinaFace (or SCRFD) for face detection
    print("Selecting Best Frame....")

    # Load video
    # video_path = 'input_video.mp4'
    cap = cv2.VideoCapture(video_path)

    best_score = -1
    best_frame = None
    face_emb = None
    frame_interval = 15  # Analyze 1 frame per second
    faces_list = []
    frame_list = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:

            height, width, _ = frame.shape
            mid = width // 2

            # Crop left half
            left_half = frame[:, :mid]

            # Convert to RGB (face_recognition uses RGB)
            frame = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
            faces = app.get(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if len(faces) > 0:
                try:
                    face = faces[0]
                    frame_list.append(frame.copy())  # Store frame
                    faces_list.append(face.embedding)  # Store face embedding
                    # print("SCRORE: ", face.pose )
                    # print("POSe: ", face.det_score )
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    face_crop = frame[y1:y2, x1:x2]

                    # Compute sharpness
                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    # sharpness = variance_of_laplacian(gray_face)

                    # Score = sharpness × face size
                    face_size = (x2 - x1) * (y2 - y1)
                    # score = sharpness * face_size

                    if  face.det_score > best_score:
                        best_score =  face.det_score
                        # best_frame = frame.copy()
                        face_emb = face.embedding.reshape(1, -1)

                except:
                    print("Error processing face detection in frame, skipping...")
                    continue

        frame_id += 1

    # if best_frame is not None:
    #     cv2.imwrite('best_frame.jpg', best_frame)

    if face_emb is not None:
        return (face_emb,frame_list,faces_list)
    else:
        print("No faces detected in the image.")
        return "No face detected"




alert_path = "/home/adlytic/Yasir Adlytic/Dataset/New_Alerts/distractedDriving/2025-07-22/3693878/3693878_20250722_062750_video.mp4"
print("Processing File: ", alert_path)

frame_faces_list = select_best_frame(alert_path)

if frame_faces_list != 'No face detected':
    name = recognize_face(frame_faces_list)
    print("Output: ", name)
else:
    print("Output: ", frame_faces_list)

            