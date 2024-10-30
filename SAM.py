# Using fastsam for Person Detection
import cv2
import numpy as np
from ultralytics import FastSAM

# Load the FastSAM model
model = FastSAM("FastSAM-s.pt")  # Load model (ensure "FastSAM-s.pt" is in the working directory)

def apply_background_blur(frame, mask):
    # Create a blurred version of the frame
    blurred_background = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # Apply mask: only keep the original frame for the person, blur the rest
    mask_inv = cv2.bitwise_not(mask)  # Invert mask for background
    person_only = cv2.bitwise_and(frame, frame, mask=mask)  # Person area
    background_only = cv2.bitwise_and(blurred_background, blurred_background, mask=mask_inv)  # Background area

    # Combine the two
    return cv2.add(person_only, background_only)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

fps_counter = 0
fps_start_time = cv2.getTickCount()  # Start time for FPS calculation

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

    # Check if results contain masks
    if results and results[0].masks is not None:
        # Extract masks
        masks = results[0].masks.data  # Assuming the first mask corresponds to the person
    else:
        masks = None

    if masks is not None and masks.shape[0] > 0:
        # Assuming the first mask corresponds to the person
        person_mask = (masks[0].cpu().numpy() * 255).astype(np.uint8)
    else:
        person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Blank mask if no person detected

    # Apply background blur effect
    output_frame = apply_background_blur(frame, person_mask)

    # Increment the FPS counter
    fps_counter += 1
    elapsed_time = (cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency()
    
    # Calculate FPS every second
    if elapsed_time >= 1.0:
        fps = fps_counter / elapsed_time
        fps_counter = 0
        fps_start_time = cv2.getTickCount()  # Reset start time

        # Display the FPS on the frame
        cv2.putText(output_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Virtual Background Blur", output_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Using YOLOv3 for Person Detection
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv3 model
model = YOLO("yolov3u.pt")  # Change to "yolov3n.pt" if needed

def apply_background_blur(frame, mask):
    blurred_background = cv2.GaussianBlur(frame, (11, 11), 0)
    mask_inv = cv2.bitwise_not(mask)
    person_only = cv2.bitwise_and(frame, frame, mask=mask)
    background_only = cv2.bitwise_and(blurred_background, blurred_background, mask=mask_inv)
    return cv2.add(person_only, background_only)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame, device='cpu')


    # Initialize mask
    height, width = frame.shape[:2]
    person_mask = np.zeros((height, width), dtype=np.uint8)

    # Process results to create a mask for detected persons
    for result in results:
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get the class ID
                if class_id == 0:  # Class ID 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

    # Apply background blur effect
    output_frame = apply_background_blur(frame, person_mask)

    # Display frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Virtual Background Blur", output_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
