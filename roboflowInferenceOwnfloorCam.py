from inference import get_model
import supervision as sv
import cv2

# Open de camera (0 voor de standaard camera, wijzig indien nodig)
capture = cv2.VideoCapture(1)

# Controleer of de camera correct is geopend
if not capture.isOpened():
    print("Fout bij openen van de camera.")
    exit()

# Laad een pre-trained YOLOv8n model
model = get_model(model_id="personalobjects/1")

# Maak supervision annotators
# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()
# polygon_annotator = sv.PolygonAnnotator()
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)  # Set the mask color to Green

while True:
    # Lees een frame van de camera
    ret, frame = capture.read()
    if not ret:
        print("Kon geen frame lezen.")
        break

    # Voer objectdetectie uit op het frame
    results = model.infer(frame)[0]
    
    # Laad de resultaten in de supervision Detections API
    detections = sv.Detections.from_inference(results)
    
    # Annoteer het frame met detectieresultaten
    # annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)
    
    # Toon het geannoteerde frame
    cv2.imshow("YOLOv8 Camera Feed", annotated_frame)
    
    # Druk op 'q' om te stoppen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de camera en sluit de vensters
capture.release()
cv2.destroyAllWindows()
