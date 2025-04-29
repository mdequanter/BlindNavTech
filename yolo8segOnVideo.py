import cv2
import numpy as np
import os
import time  # <-- Nieuw toegevoegd
from ultralytics import YOLO

# --- Model laden ---
model = YOLO('models\\pathfinderYolo8Seg.pt', verbose=True)

# --- Videobestand openen ---
video_path = r'videos\\G0110729.MP4'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Fout bij het openen van de video")
    exit()

# --- Video-eigenschappen ophalen ---
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Outputpad opbouwen ---
dirname, basename = os.path.split(video_path)
name, ext = os.path.splitext(basename)
output_path = os.path.join(dirname, f"{name}_result.mp4")

# --- VideoWriter instellen ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Gebruik 'XVID' voor AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Resultaatvideo wordt opgeslagen als: {output_path}")

# --- Voor FPS-berekening ---
prev_time = time.time()  # <-- Nieuw toegevoegd

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame niet beschikbaar, probeer volgende frame")
        continue

    results = model(frame, conf=0.8, verbose=False)

    overlay = frame.copy()
    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                points = np.array(mask, dtype=np.int32)
                cv2.fillPoly(overlay, [points], color=(0, 255, 0))
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # --- FPS berekenen en weergeven ---
    current_time = time.time()
    fps_text = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps_text:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # <-- Nieuw toegevoegd

    cv2.imshow("YOLOv8 Segmentatie", frame)
    out.write(frame)  # Frame wegschrijven naar outputbestand

# --- Alles afsluiten ---
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video verwerking voltooid.")