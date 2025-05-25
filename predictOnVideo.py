from inference import get_model
import supervision as sv
import cv2
import time  # Import time module to measure inference time

# Open de video in plaats van de camera
video_path = "videos/botopiaCitycam.mp4"
output_path = "videos/botopiaCitycam.mp4_result.mp4"
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
model = get_model(model_id="botopiarobotcam/1")

# Maak supervision annotators
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)  # Set the mask color to Green

# Stel een hogere confidence threshold in (bijv. 0.6)
CONFIDENCE_THRESHOLD = 0.2
FLOOR_CLASS_ID = 2  # Class ID voor de vloer

frame_counter = 0  # Teller voor frames

# Variables to calculate total inference time
total_inference_time = 0
total_frames = 0

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

    # Start time measurement for inference
    start_time = time.time()

    # Voer objectdetectie uit op het frame
    results = model.infer(frame)[0]
    
    # End time measurement after inference
    end_time = time.time()

    # Calculate and accumulate inference time for the frame
    inference_time = end_time - start_time
    total_inference_time += inference_time
    total_frames += 1

    # Laad de resultaten in de supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Filter de detecties op confidence score en klasse-ID (alleen vloer)
    floor_detections = detections[
        (detections.confidence > CONFIDENCE_THRESHOLD) & (detections.class_id == FLOOR_CLASS_ID)
    ]

    # Annoteer het frame met alleen vloer-detecties
    annotated_frame = mask_annotator.annotate(scene=frame, detections=floor_detections)

    # Overlay inference time on the frame
    inference_time_text = f"Inference Time: {inference_time * 1000:.2f} ms"  # Convert to milliseconds
    cv2.putText(annotated_frame, inference_time_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Schrijf het geannoteerde frame naar het uitvoervideo-bestand
    out.write(annotated_frame)
    
    # Toon het geannoteerde frame
    cv2.imshow("YOLOv8 Video - Detecting Only Floor (Double Speed)", annotated_frame)
    
    # Druk op 'q' om te stoppen
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Bereken gemiddelde inference tijd per frame
average_inference_time = total_inference_time / total_frames if total_frames > 0 else 0

# Sluit de video en sluit de vensters
capture.release()
out.release()  # Sla de video op
cv2.destroyAllWindows()

print(f"Video opgeslagen als {output_path}")
print(f"Gemiddelde inferentie tijd per frame: {average_inference_time:.4f} seconden")
