import cv2
import mediapipe as mp
import pickle
import numpy as np

# Initialize video capture from the default camera
video = cv2.VideoCapture(0)

# Initialize MediaPipe Hands solution components
mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawingUtils = mp.solutions.drawing_styles

# Create a Hands object for static image mode with a minimum detection confidence
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load the pre-trained model and labels dictionary from a pickle file
model_dict = pickle.load(open("./model.pickle", "rb"))
model = model_dict["model"]

# Dictionary mapping model output to corresponding hand sign labels
labels_dict = {1: "C", 2: "L", 3: "B"}

# Start the video capture and processing loop
while True:
    data_aux = []  # List to hold normalized landmark coordinates
    x_ = []  # List to collect x-coordinates of landmarks
    y_ = []  # List to collect y-coordinates of landmarks
    
    # Capture a frame from the video
    ret, frame = video.read()
    H, W, _ = frame.shape  # Get the height and width of the frame
    
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Flip the frame horizontally for a mirrored effect
    frame_flip = cv2.flip(rgb_frame, 1)
    
    # Process the flipped frame to detect hand landmarks
    processed = hands.process(frame_flip)
        
    if processed.multi_hand_landmarks:
        for landmarks in processed.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawings.draw_landmarks(
                frame_flip, landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawingUtils.get_default_hand_landmarks_style(),
                mp_drawingUtils.get_default_hand_connections_style()
            )
            
            # Extract and normalize landmark coordinates
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
                
            # Define bounding box coordinates for the hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
                
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            
            # Predict the hand sign using the pre-trained model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_letter = labels_dict[int(prediction[0])]
            
            # Convert the frame back to BGR for OpenCV display
            bgr_frame = cv2.cvtColor(frame_flip, cv2.COLOR_RGB2BGR)
            
            # Draw the bounding box and predicted label on the frame
            cv2.rectangle(frame_flip, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame_flip, predicted_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, 
                        cv2.LINE_AA)
        
    # Display the processed frame in a window
    cv2.imshow("frame", frame_flip)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
