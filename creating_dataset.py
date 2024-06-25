import cv2
import pickle
import os 
import  mediapipe as mp

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.4)

Data_dir = "./data"

data = []
labels = []

for sign in os.listdir(Data_dir):
    for image_path in os.listdir(os.path.join(Data_dir, sign)):
        data_aux = []
                
        x_ = []
        y_ = []

        
        image = cv2.imread(os.path.join(Data_dir,sign,image_path))
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        processed = hands.process(rgb_image)
        
        if processed.multi_hand_landmarks:
            for landmarks in processed.multi_hand_landmarks:
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
                    
            
            data.append(data_aux)
            labels.append(sign)
            
            
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
