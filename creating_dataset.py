import cv2
import os 
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg or another interactive backend

import pickle
import matplotlib.pyplot as plt 
import  mediapipe as mp

mp_hands = mp.solutions.hands 
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

Data_dir = "./data"
if not os.path.exists(Data_dir):
    os.mkdir(Data_dir)

data = []
labels = []

for sign in os.listdir(Data_dir):
    sign_path = os.path.join(Data_dir,sign)
    if not os.path.isdir(sign_path):
        continue
    
    for image_name in os.listdir(sign_path):
        image_path = os.path.join(sign_path,image_name)
        data_aux = []
                
        x_ = []
        y_ = []

        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        processed = hands.process(rgb_image)
        
        if processed.multi_hand_landmarks:
            for landmarks in processed.multi_hand_landmarks:
                # mp_drawings.draw_landmarks(
                #     rgb_image,  landmarks, 
                #     mp_hands.HAND_CONNECTIONS,  mp_drawingUtils.get_default_hand_landmarks_style(),
                #     mp_drawingUtils.get_default_hand_connections_style()
                # )
                
                # plt.figure()
                # plt.imshow(rgb_image)
                # plt.title(f"Processed image of {sign}")
                # plt.savefig(f"processed_{sign}_{image_name}.png")
                # plt.show()
                
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
