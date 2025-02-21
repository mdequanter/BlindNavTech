import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
import supervision as sv
import json

# Initialize the Roboflow model
rf = Roboflow(api_key="WgLYEMfa0WjwZWHJhwlO")
project = rf.workspace().project("personalobjects")
model = project.version(1).model

# Load image
image_path = "Pictures/floor2.jpg"
image = cv2.imread(image_path)

# Get predictions
result = model.predict(image_path, confidence=20).json()
predictions = result.get('predictions', [])

# Create an empty mask with the same size as the image
mask = np.zeros_like(image, dtype=np.uint8)

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

# Blend the mask with the original image
alpha = 0.5  # Transparency factor
colored_image = cv2.addWeighted(image, 1, mask, alpha, 0)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
