import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

# Set up MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Initialize camera capture
cap = cv2.VideoCapture(0)

# Ensure the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame dimensions from the actual camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback if FPS is unavailable

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as virtual_camera:
    print(f'Using virtual camera: {virtual_camera.device}')
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a selfie view
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            # Convert the image color back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create a mask based on segmentation results and apply Gaussian Blur to the background
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = cv2.GaussianBlur(image, (55, 55), 0)
            output_image = np.where(condition, image, bg_image)

            # Send the output to the virtual camera
            virtual_camera.send(output_image)
            virtual_camera.sleep_until_next_frame()

            # Optionally display the result in an OpenCV window
            cv2.imshow('MediaPipe Selfie Segmentation', output_image)
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

# Release resources
cap.release()
cv2.destroyAllWindows()
