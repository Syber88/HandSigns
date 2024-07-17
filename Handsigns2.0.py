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