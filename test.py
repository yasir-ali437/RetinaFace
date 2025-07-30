import cv2

img_path = '/home/adlytic/Yasir Adlytic/Dataset/Drivers_Dataset_24_July/d159/frame_0.jpg'

img = cv2.imread(img_path)
print(img.shape)