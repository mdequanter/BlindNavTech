import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from PIL import Image

# --- Configuratie ---
MODEL_PATH = "test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
IMAGE_PATH = "test_data/floor1.jpg"  # Gebruik een testafbeelding
NUM_RUNS = 50  # Aantal inferentie-runs voor FPS-meting

delegate_path = "edgetpu.dll"

# --- TPU Interpreter laden ---
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH, experimental_delegates=[tflite.load_delegate(delegate_path)]
)
interpreter.allocate_tensors()

# Input en output details ophalen
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Dynamische inputresolutie ophalen ---
input_shape = tuple(input_details[0]['shape'][1:3])  # Converteer NumPy array naar tuple
print(f"✅ Model verwacht invoerresolutie: {input_shape}")

# --- Preprocessing functie ---
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_shape)  # Correcte resize zonder ANTIALIAS
    return np.array(image, dtype=np.uint8)

# --- Benchmarking ---
print(f"\n🔹 Benchmarking voor resolutie: {input_shape}")

# Laad en preprocess afbeelding
image = preprocess_image(IMAGE_PATH, input_shape)
image = np.expand_dims(image, axis=0)  # Voeg batch dimensie toe
interpreter.set_tensor(input_details[0]['index'], image)

# Meting inferentie snelheid
start_time = time.time()
for _ in range(NUM_RUNS):
    interpreter.invoke()  # Voer inferentie uit
end_time = time.time()

# FPS berekening
total_time = end_time - start_time
fps = NUM_RUNS / total_time
print(f"✅ FPS: {fps:.2f}")

# Gemiddelde inferentietijd per frame
avg_time_per_frame = (total_time / NUM_RUNS) * 1000  # in ms
print(f"🕒 Gemiddelde inferentietijd: {avg_time_per_frame:.2f} ms")

print("\n🔹 Benchmark voltooid!")
