import cv2
import numpy as np
import os
import time
import onnxruntime as ort  # <-- Nieuw: ONNXRuntime gebruiken

# --- Model laden (ONNX) ---
model_path = 'models\\pathfinderYolo8Seg.onnx'  # <-- Zorg dat je model eerst geëxporteerd is!
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # bijv. [1, 3, 640, 640]

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Resultaatvideo wordt opgeslagen als: {output_path}")

# --- Voor FPS-berekening ---
prev_time = time.time()

# --- Video loop ---
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame niet beschikbaar, probeer volgende frame")
        continue

    # --- Preprocessing ---
    input_frame = cv2.resize(frame, (640, 640))  # pas aan naar input_shape indien nodig
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.transpose(2, 0, 1)  # HWC -> CHW
    input_frame = np.expand_dims(input_frame, axis=0)  # batch dimension
    input_frame = input_frame.astype(np.float32) / 255.0

    # --- Inference ---
    outputs = session.run(None, {input_name: input_frame})

    # --- Postprocessing ---
    # outputs is een lijst van numpy arrays
    # hier hangt het af van je model: bbox, klasse, maskers, ...
    # Voor segmentatie, stel dat maskers in outputs[0] zitten (moet je even checken bij je export)

    # Je moet hier je eigen mask processing invullen afhankelijk van hoe jouw onnx model output is gestructureerd.
    # Ik zet hieronder placeholder code:

    overlay = frame.copy()
    # Placeholder voor maskers: (vervang dit met je eigen masker-processing)
    # Bijvoorbeeld dummy mask:
    mask = np.zeros((height, width), dtype=np.uint8)  # zwart masker

    # Voeg het masker toe als transparante overlay
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 1] = mask  # groen kanaal
    alpha = 0.3
    frame = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)

    # --- FPS berekenen en weergeven ---
    current_time = time.time()
    fps_text = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps_text:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 ONNX Segmentatie", frame)
    out.write(frame)

# --- Alles afsluiten ---
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video verwerking voltooid.")
