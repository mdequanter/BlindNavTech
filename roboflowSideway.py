import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow
import supervision as sv
import json

rf = Roboflow(api_key="3piPof3nd2m82YCCLDD7")
project = rf.workspace().project("netrasahaya")
model = project.version(6).model

image_path = "Pictures/DenHam1.jpg"

image = cv2.imread(image_path)

result = model.predict(image_path, confidence=55).json()

predictions = result.get('predictions', [])

# Draw bounding boxes and labels on the image
for pred in predictions:
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)
    label = f"{pred['class']} {pred['confidence']:.2f}"

    # Draw rectangle
    color = (255, 0, 0)  # Red for bounding boxes
    #cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle (image, (int(pred['x']), int(pred['y'])), 5, color, -1)

    # Put label
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()