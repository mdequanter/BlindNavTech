import cv2
import numpy as np
from roboflow import Roboflow

# Initialize the Roboflow model
rf = Roboflow(api_key="WgLYEMfa0WjwZWHJhwlO")
project = rf.workspace().project("personalobjects")
model = project.version(1).model

# Open webcam (0 = default webcam, change to 1 or 2 if using external webcams)
cap = cv2.VideoCapture(0)

# Set frame dimensions (adjust for performance)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Save the frame to a temporary file for prediction
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Get predictions
    result = model.predict(temp_image_path, confidence=50).json()
    predictions = result.get('predictions', [])

    # Create an empty mask with the same size as the frame
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Define a color for the floor area
    floor_color = (0, 255, 0)  # Green

    # Process predictions
    for pred in predictions:
        if pred['class'] == 'floor':  # Ensure we're processing floor predictions
            if 'points' in pred:  # Check if polygon points are available
                # Extract x and y values from the dictionary list
                polygon = np.array([(point['x'], point['y']) for point in pred['points']], dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))  # Reshape for OpenCV fillPoly
                
                # Fill the floor area with color
                cv2.fillPoly(mask, [polygon], floor_color)
            else:
                # Fallback to bounding box if no polygon data is provided
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)

                bbox_polygon = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]], dtype=np.int32)
                cv2.fillPoly(mask, [bbox_polygon], floor_color)

    # Blend the mask with the original frame
    alpha = 0.5  # Transparency factor
    blended_frame = cv2.addWeighted(frame, 1, mask, alpha, 0)

    # Show the processed frame
    cv2.imshow("Floor Detection", blended_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


