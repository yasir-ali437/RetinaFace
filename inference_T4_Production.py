# inference.py
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# import face_recognition
import os
from openpyxl import Workbook

dataset_path = "/home/adlytic/Yasir Adlytic/Dataset/Drivers_Dataset_29_July/"
# /home/adlytic/Yasir Adlytic/Dataset/Driver_Faces_23_July/
DATABASE_PATH = "retinaface_driver_embeddings_29_July.pkl"

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

def recognize_face(alert_folder, date, sheet, frame_faces_list, img_path, threshold=0.42):
    face_emb = frame_faces_list[0]
    frame_list = frame_faces_list[1]
    faces_list = frame_faces_list[2]
    drivers_count = len(os.listdir(dataset_path))
    similarities = cosine_similarity(face_emb, embeddings_db)[0]

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        name = names_db[best_idx]
        sheet.append([date, alert_folder, name])
        # drivers_facepics_count = len(os.listdir(os.path.join(dataset_path,name)))
        # if drivers_facepics_count < 50:
        #     print("Saving frames...for known driver: ",name, "curent frame count: ",drivers_facepics_count)

        #     for i, f in enumerate(frame_list):
        #         filename = os.path.join(dataset_path,name, f"frame_{drivers_facepics_count+1}.jpg")
        #         cv2.imwrite(filename, f)
        #         # print(f"Saved {filename}")
        #         drivers_facepics_count+=1
            
        #     for i, f in enumerate(faces_list):
        #         embeddings_db.append(f)
        #         names_db.append(name)

            # training.create_driver_embeddings(dataset_path="/home/adlytic/Yasir Adlytic/Dataset/Driver_Faces_Updated_New/")
        
    else:
        name = "Unknown"
        print("Saving frames...for unknown driver")
        sheet.append([date, alert_folder, "d" + str(drivers_count+1)+" New Added Driver"])

        for i, f in enumerate(frame_list):
            if not os.path.isdir(os.path.join(dataset_path,f'd{drivers_count+1}')):
                os.makedirs(os.path.join(dataset_path,f'd{drivers_count+1}'))
            filename = os.path.join(dataset_path,f'd{drivers_count+1}', f"frame_{i}.jpg")
            cv2.imwrite(filename, f)
            # print(f"Saved {filename}")

        for i, f in enumerate(faces_list):
            embeddings_db.append(f)
            names_db.append(str(f'd{drivers_count+1}'))

                # training.create_driver_embeddings(dataset_path="/home/adlytic/Yasir Adlytic/Dataset/Driver_Faces_Updated_New/")

            # Draw result
            # box = face.bbox.astype(int)
            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # cv2.putText(img, f"{name} ({best_score:.2f})", (box[0], box[1]-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # cv2.imwrite("retinaresult.jpg", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def select_best_frame(sheet, video_path, every_n_frames=5, cnnflag=False):

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
        sheet.append([date, alert_folder, "No face detected"])
        return "No face detected"


parentFolder = "/home/adlytic/Yasir Adlytic/Dataset/New_Alerts/"
# Create a new Excel workbook
wb = Workbook()
sheet = wb.active
newFlag = True

for category in os.listdir(parentFolder):
    category_path = os.path.join(parentFolder, category)
    print(f"Working on Category: {category}")
    if os.path.isdir(category_path):
        # First sheet (created by default)
        if newFlag:
            sheet.title = category
            newFlag = False
        else: 
            sheet = wb.create_sheet(title=category)

        for date in os.listdir(category_path):
            if "Processed" not in str(date):
                print(f"Working on Date: {date}")
                date_path = os.path.join(category_path, date)
                if os.path.isdir(date_path):
                    for alert_folder in os.listdir(date_path):
                        alert_folder_path = os.path.join(category_path, date,alert_folder)
                        if os.path.isdir(alert_folder_path):
                            alerts = os.listdir(alert_folder_path)
                            if alerts!=[]:
                                alert_path = os.path.join(category_path, date, alert_folder, alerts[0])
                                print("Processing File: ", alert_path)
                        
                                frame_faces_list = select_best_frame(sheet, alert_path)

                                if frame_faces_list:
                                    recognize_face(alert_folder, date, sheet, frame_faces_list, "best_frame.jpg")
            else:
                print(f"Skipping {date_path} as it is already processed.")
                # Optionally, you can rename or move the processed folder
                # to avoid reprocessing in the future
                # e.g.,
        # os.rename(date_path, date_path + " Processed")
    else:
        print(f"Skipping {category_path} as it is not a directory.")

save_embeddings()
# Save the workbook
output_excel_path = os.path.join("Face_Recognition_Results.xlsx")
wb.save(output_excel_path)
print(f"Results saved to {output_excel_path}")
        