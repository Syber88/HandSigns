import cv2 
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

while True:
    
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_flip = cv2.flip(rgb_frame,1)
    
    processed = hands.process( rgb_frame)
        
    if processed.multi_hand_landmarks:
        for landmarks in processed.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                frame,  landmarks, 
                mp_hands.HAND_CONNECTIONS,  
                mp_drawingUtils.get_default_hand_landmarks_style(),
                mp_drawingUtils.get_default_hand_connections_style()
            )
    
    cv2.imshow("frame",  frame)
    cv2.waitKey(25)
    
    video.release()
    cv2.destroyAllWindows()