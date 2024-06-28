import cv2 
import mediapipe as mp
import pickle
import numpy as np

video = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

model_dict = pickle.load(open("./model.pickle", "rb"))
model = model_dict["model"]

labels_dict = {1:"C", 2:"L", 3:"B"}
while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = video.read()
    H, W, _ = frame.shape
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
            
        for landmarks in processed.multi_hand_landmarks:
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
            
                x_.append(x)
                y_.append(y)
                    
            for i in range(len(landmarks.landmark)):    
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
            
        prediction = model.predict([np.asarray(data_aux)])
        predicted_letter = labels_dict[int(prediction[0])]
        print(predicted_letter)
        
        
        bgr_frame = cv2.cvtColor(frame_flip, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(bgr_frame, predicted_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, 
                    cv2.LINE_AA)
            
                    
    cv2.imshow("frame", bgr_frame)
    if cv2.waitKey(1) == ord('q'):
        break
     - 10
video.release()
cv2.destroyAllWindows()