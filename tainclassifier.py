import cv2 
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

while True:
    
    ret, frame = video.read()
    flip = cv2.flip(frame,1)
    rgb_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    
    processed = hands.process(rgb_flip)
        
    if processed.multi_hand_landmarks:
    
    cv2.imshow("frame", rgb_flip)
    cv2.waitKey(25)
    
video.release()
cv2.destroyAllWindows()