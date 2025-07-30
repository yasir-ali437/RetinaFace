# inference.py
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import os
import training

dataset_path = "/home/adlytic/Yasir Adlytic/Adlytic_Internship/Face Recognition/Driver_Faces_Updated/"

def recognize_face(frame_list, img_path, database_path="retinaface_driver_embeddings.pkl", threshold=0.45):

    drivers_count = len(os.listdir(dataset_path))
    with open(database_path, "rb") as f:
        data = pickle.load(f)

    embeddings_db = data["embeddings"]
    names_db = data["names"]

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)  # GPU: 0, CPU: -1

    img = cv2.imread(img_path)
    # Get image dimensions
    height, width, _ = img.shape
    mid = width // 2

    # Crop left half
    left_half = img[:, :mid]

    # Convert to RGB (face_recognition uses RGB)
    image_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
    
    faces = app.get(image_rgb)

    for face in faces:
        emb = face.embedding.reshape(1, -1)
        similarities = cosine_similarity(emb, embeddings_db)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > threshold:
            name = names_db[best_idx]
            print("Saving frames...for known driver: ",name)
            drivers_facepics_count = len(os.listdir(os.path.join(dataset_path,name)))
            if drivers_facepics_count < 50:
                for i, f in enumerate(frame_list):
                    filename = os.path.join(dataset_path,name, f"frame_{drivers_facepics_count+1}.jpg")
                    cv2.imwrite(filename, f)
                    print(f"Saved {filename}")
                    drivers_facepics_count+=1
                training.create_driver_embeddings()
        else:
            name = "Unknown"
            print("Saving frames...for unknown driver")
            for i, f in enumerate(frame_list):
                filename = os.path.join(dataset_path,f'd{drivers_count+1}', f"frame_{i}.jpg")
                cv2.imwrite(filename, f)
                print(f"Saved {filename}")
            training.create_driver_embeddings()

        # Draw result
        # box = face.bbox.astype(int)
        # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.putText(img, f"{name} ({best_score:.2f})", (box[0], box[1]-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # cv2.imwrite("retinaresult.jpg", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def select_best_frame(video_path, every_n_frames=5, cnnflag=False):
    video = cv2.VideoCapture(video_path)

    best_frame = None
    best_score = 0
    frame_count = 0
    frame_list = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Save every 15th frame into the list
        if frame_count % 15 == 0:
            frame_list.append(frame.copy())  # Store frame

        if frame_count % every_n_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get image dimensions
            height, width, _ = rgb_frame.shape
            mid = width // 2

            # Crop left half
            left_half = rgb_frame[:, :mid]

            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('best_frame1.jpg', frame)
            if cnnflag:
                face_locations = face_recognition.face_locations(image_rgb, model='cnn')
            else:
                face_locations = face_recognition.face_locations(image_rgb)
                
            face_landmarks = face_recognition.face_landmarks(image_rgb)
            # print("face_locations", face_locations, "face_landmarks", face_landmarks)
            # Basic scoring: face must exist + have eyes & nose
            if face_locations and face_landmarks:
                landmarks = face_landmarks[0]
                score = 0
                if 'left_eye' in landmarks: score += 1
                if 'right_eye' in landmarks: score += 1
                if 'nose_tip' in landmarks: score += 1
                if 'chin' in landmarks: score += 1
                if 'top_lip' in landmarks: score += 0.5
                if 'bottom_lip' in landmarks: score += 0.5

                if score >= best_score:
                    best_score = score
                    best_frame = frame.copy()
                    
            elif face_locations and face_landmarks==[]:
                best_frame = frame.copy()
                
            # elif face_locations==[] and face_landmarks==[]:
            #     print("Running again with CNN model....")
            #     select_best_frame(video_path, every_n_frames, cnnflag=True)
                
        frame_count += 1

    video.release()
    if best_frame is None and cnnflag==False:
        print("No suitable frame found.")
        print("Running again with CNN model....")
        select_best_frame(video_path, every_n_frames, cnnflag=True)
    else:
        print("Skipping Video........no face detected")
        return False 

    if best_frame is not None:
        cv2.imwrite('best_frame.jpg', best_frame)

    return frame_list

file_names = os.listdir("../Data/Evidence")
# print(file_names)

for file_name in file_names:
    print("Processing File: ", file_name)
    frame_list = select_best_frame(f"../Data/Evidence/{file_name}")
    if frame_list:
        recognize_face(frame_list, "best_frame.jpg")
