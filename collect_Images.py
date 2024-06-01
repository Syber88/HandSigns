import cv2
import os

cam = cv2.VideoCapture(0)

image_count = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    if key == ord("c"):
        pass
        
    
    
cam.release()  
cv2.destroyAllWindows()