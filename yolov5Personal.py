import torch
import cv2
import numpy as np
import supervision as sv

# YOLOv5-segmentatiemodel laden via Torch Hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='segPersonalObjects.pt', force_reload=True)

model.eval()
model.conf = 0.25  # Zorg ervoor dat het wordt toegepast
model.iou = 0.45  # Standaard IoU threshold (optioneel)model.max_det = 100  # Zorg ervoor dat alle objecten worden verwerkt

# Debug info printen
print(f"Model settings: conf={model.conf}, iou={model.iou}, max_det={model.max_det}")


# Model in evaluatiemodus zetten
model.eval()

# Webcam starten
cap = cv2.VideoCapture(1)

# Supervision Annotator voor segmentatie
# Maak supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
polygon_annotator = sv.PolygonAnnotator()
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)  # Set the mask color to Green

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converteren naar RGB (YOLOv5 verwacht RGB invoer)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detectie
    results = model(frame_rgb)

    # Supervision Detections object maken uit YOLOv5 resultaten
    detections = sv.Detections.from_yolov5(results)
 
     # Annoteer het frame met detectieresultaten
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    # Toon het frame
    cv2.imshow("YOLOv5 Segmentation", annotated_frame)
    
    # Stop als 'q' wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
