import cv2
import os


cam = cv2.VideoCapture(0)
instructions = "Press Q to quit and C to capture images."
font = cv2.FONT_ITALIC
position = (5, 30)  # Position for the text
font_scale = 0.9
color = (0, 0, 0)  
thickness = 1

image_classes = 3

for i in range(image_classes):
    while True:
        ret, frame = cam.read()
        flip = cv2.flip(frame,1)

        cv2.putText(flip, instructions, position, font, font_scale, color, thickness)
        cv2.imshow("webcam", flip)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        if key == ord("c"):
            for j in range (100):
                image_name = f"captured image {j}.jpg"
                Data_Dir = f"Desktop/HandSigns/data/{i}"
                base_dir = os.path.expanduser("~")
                full_path = os.path.join(base_dir, Data_Dir)
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                image_save = os.path.join(full_path, image_name)
                cv2.imwrite(image_save, frame)
            break
        
cam.release()  
cv2.destroyAllWindows()