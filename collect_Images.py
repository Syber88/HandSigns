import cv2
import os

# Initialize webcam capture
cam = cv2.VideoCapture(0)

# Instructions to be displayed on the video feed
instructions = "Press Q to quit and C to capture images."

# Font settings for the instructions text
font = cv2.FONT_ITALIC
position = (5, 30)  # Position for the text
font_scale = 0.9
color = (0, 0, 0)  
thickness = 1

# Number of image classes currently
image_classes =4

# Loop through each image class
for i in range(1, image_classes):
    while True:
        # Capture frame from webcam
        ret, frame = cam.read()
        
        # Flip the frame horizontally
        flip = cv2.flip(frame, 1)

        # Add instructions text to the flipped frame
        cv2.putText(flip, instructions, position, font, font_scale, color, thickness)

        # Display the frame with instructions in a window
        cv2.imshow("webcam", flip)

        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' key is pressed, exit the loop
        if key == ord("q"):
            break

        # If 'c' key is pressed, start capturing images
        if key == ord("c"):
            for j in range(100):
                # Generate a name for the image
                image_name = f"captured image {j}.jpg"
                
                # Define the directory to save images
                Data_Dir = f"Desktop/HandSigns/data/{i}"
                base_dir = os.path.expanduser("~")  # Get the user's home directory
                full_path = os.path.join(base_dir, Data_Dir)  # Construct the full path

                # Create the directory if it doesn't exist
                if not os.path.exists(full_path):
                    os.makedirs(full_path)

                # Save the captured image in the defined path
                image_save = os.path.join(full_path, image_name)
                cv2.imwrite(image_save, frame)
            
            # Break the inner loop after capturing images
            break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
