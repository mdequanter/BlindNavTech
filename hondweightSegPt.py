import cv2
import numpy as np
from ultralytics import YOLO


# Laad je YOLOv8-segmentatiemodel
# model = YOLO('segPersonalObjects.pt', verbose=False)
model = YOLO('hondWeights.pt', verbose=True)

# Open de webcam (gebruik source=0 voor de standaard webcam, 1 voor een externe)
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow


# Controleer of de webcam correct is geopend
if not cap.isOpened():
    print("Fout bij het openen van de webcam")
    exit()


while True:
    # Lees een frame van de webcam
    ret, frame = cap.read()
    if not ret:
        print("Kan frame niet lezen, sluit af...")
        break

    # Voer YOLOv8 segmentatie uit
    results = model(frame, conf=0.5, verbose=False)

  

    # Teken de resultaten op het frame
    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                # FIXED: Converteer mask-coördinaten naar een integer NumPy-array
                points = np.array(mask, dtype=np.int32)

                # Teken de segmentatiemaskers
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.fillPoly(frame, [points], color=(0, 255, 0, 50))  # Semi-transparant groen masker

        '''
        # Teken bounding boxes en labels
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Verkrijg bounding box coördinaten
            label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"  # Objectlabel + vertrouwen

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        '''
    # Toon het bewerkte frame
    cv2.imshow("YOLOv8 Segmentatie", frame)

    # Stoppen met 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de webcam en vensters
cap.release()
cv2.destroyAllWindows()
