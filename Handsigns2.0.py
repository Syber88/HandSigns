import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import mediapipe as mp 
import os 

# mp.solutions.mediapipe.
mp_holistic = mp.solutions.mediapipe.solutions.holistic #holistic model
# mp_drawing = mp.solutions.drawing_utils #drawign utilities
mp_drawing = mp.solutions.mediapipe.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results 

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,213,90), thickness=1, circle_radius=1))
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(53,56,38), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(100,245,78), thickness=1, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100,64,90), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(150,80,75), thickness=1, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(150,245,75), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(220,76,56), thickness=1, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmarks]).flatten() if results.pose_landmarks.landmark else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmarks]).flatten() if results.left_hand_landmaeks.landmark else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmarks]).flatten() if results.right_hand_landmaeks.landmark else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmarks]).flatten() if results.face_landmaeks.landmark else np.zeros(1404)
    return np.concatenate([pose, face, lh, rh])

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        # print(len(results.left_hand_landmarks.landmark))
        cv2.imshow("feed", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()