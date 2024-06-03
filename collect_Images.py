import cv2
import os

Data_Dir = "Desktop/HandSigns/data"
base_dir = os.path.expanduser("~")
full_path = os.path.join(base_dir, Data_Dir)
if not os.path.exists(full_path):
    os.makedirs(full_path)

cam = cv2.VideoCapture(0)

image_count = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    if key == ord("c"):
        image_name = "captured image{}.jpg".format(image_count)
        image_save = os.path.join(full_path, image_name)
        cv2.imwrite(image_save, frame)
        image_count += 1
        
cam.release()  
cv2.destroyAllWindows()