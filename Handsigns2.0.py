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