import cv2
import os
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg or another interactive backend for plotting

import pickle
import matplotlib.pyplot as plt
import mediapipe as mp

# Initialize MediaPipe Hands solution components
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

# Create a Hands object for static image mode with a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory to store hand sign data
Data_dir = "./data"
if not os.path.exists(Data_dir):
    os.mkdir(Data_dir)

# Initialize lists to hold the data and labels
data = []
labels = []

# Iterate through each sign class directory in the data directory
for sign in os.listdir(Data_dir):
    sign_path = os.path.join(Data_dir, sign)
    if not os.path.isdir(sign_path):
        continue
    
    # Iterate through each image in the sign class directory
    for image_name in os.listdir(sign_path):
        image_path = os.path.join(sign_path, image_name)
        data_aux = []  # List to hold processed landmark data
        
        x_ = []  # List to collect x-coordinates of landmarks
        y_ = []  # List to collect y-coordinates of landmarks
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Convert the image from BGR to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hand landmarks
        processed = hands.process(rgb_image)
        
        if processed.multi_hand_landmarks:
            for landmarks in processed.multi_hand_landmarks:
                # Uncomment the following lines to draw and visualize landmarks
                
                # mp_drawings.draw_landmarks(
                #     rgb_image, landmarks, 
                #     mp_hands.HAND_CONNECTIONS, 
                #     mp_drawingUtils.get_default_hand_landmarks_style(),
                #     mp_drawingUtils.get_default_hand_connections_style()
                # )
                
                # plt.figure()
                # plt.imshow(rgb_image)
                # plt.title(f"Processed image of {sign}")
                # plt.savefig(f"processed_{sign}_{image_name}.png")
                # plt.show()
                
                # Extract and normalize landmark coordinates
                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                
                    x_.append(x)
                    y_.append(y)
                    
                for i in range(len(landmarks.landmark)):    
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                    
                    # Normalize coordinates based on the minimum value
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    
            data.append(data_aux)
            labels.append(sign)

# Save the processed data and labels to a pickle file
try:
    with open("./data.pickle", "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
except FileNotFoundError:
    print(f"Error: The file {f} does not exist.")
except pickle.UnpicklingError as e:
    print(f"Error: Failed to unpickle the file: {e}")
except Exception as e:
    print(f"Error: An unexpected error occurred: {e}")
