import cv2 
import mediapipe as mp
import pickle
import numpy as np

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = False, min_detection_confidence = 0.3)

model_dict = pickle.load(open("./model.pickle", "rb"))
model = model_dict["model"]

labels = {}
while True:
    
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_flip = cv2.flip(rgb_frame,1)
    
    processed = hands.process(frame_flip)
        
    if processed.multi_hand_landmarks:
        for landmarks in processed.multi_hand_landmarks:
            mp_drawings.draw_landmarks(
                frame_flip,  landmarks, 
                mp_hands.HAND_CONNECTIONS,  
                mp_drawingUtils.get_default_hand_landmarks_style(),
                mp_drawingUtils.get_default_hand_connections_style()
            )
            
            for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                
                    x_.append(x)
                    y_.append(x)
                    
            for i in range(len(landmarks.landmark)):    
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            
        prediction = model.predict([np.asarray(data_aux)])
            
                    
    bgr_frame = cv2.cvtColor(frame_flip, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", bgr_frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()