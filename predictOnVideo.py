from inference import get_model
import supervision as sv
import cv2

# Open de video in plaats van de camera
video_path = "paris.mp4"
output_path = "paris_result.mp4"
capture = cv2.VideoCapture(video_path)

# Controleer of de video correct is geopend
if not capture.isOpened():
    print(f"Fout bij openen van de video: {video_path}")
    exit()

# Haal de FPS en resolutie van de originele video op
fps = int(capture.get(cv2.CAP_PROP_FPS)) * 2  # Verdubbel de snelheid
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definieer de video writer (codec: mp4v voor MP4-bestanden)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Laad een pre-trained YOLOv8n model
model = get_model(model_id="personalobjects/1")

# Maak supervision annotators
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)  # Set the mask color to Green

# Stel een hogere confidence threshold in (bijv. 0.6)
CONFIDENCE_THRESHOLD = 0.2
FLOOR_CLASS_ID = 2  # Class ID voor de vloer

frame_counter = 0  # Teller voor frames

while True:
    # Lees een frame van de video
    ret, frame = capture.read()
    if not ret:
        print("Einde van de video bereikt of fout bij lezen frame.")
        break

    # Verhoog de teller en sla om de beurt frames over
    frame_counter += 1
    if frame_counter % 2 == 1:  # Skip every second frame
        continue

    # Voer objectdetectie uit op het frame
    results = model.infer(frame)[0]
    
    # Laad de resultaten in de supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Filter de detecties op confidence score en klasse-ID (alleen vloer)
    floor_detections = detections[
        (detections.confidence > CONFIDENCE_THRESHOLD) & (detections.class_id == FLOOR_CLASS_ID)
    ]

    # Annoteer het frame met alleen vloer-detecties
    annotated_frame = mask_annotator.annotate(scene=frame, detections=floor_detections)

    # Schrijf het geannoteerde frame naar het uitvoervideo-bestand
    out.write(annotated_frame)
    
    # Toon het geannoteerde frame
    cv2.imshow("YOLOv8 Video - Detecting Only Floor (Double Speed)", annotated_frame)
    
    # Druk op 'q' om te stoppen
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Sluit de video en sluit de vensters
capture.release()
out.release()  # Sla de video op
cv2.destroyAllWindows()

print(f"Video opgeslagen als {output_path}")
