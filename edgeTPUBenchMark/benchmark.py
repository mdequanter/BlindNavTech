"""
To get started make sure you have the following installed:
Coral Edge software: https://coral.ai/docs/accelerator/get-started/
And also PyCoral as mentioned in the above link.

Code has been tested on Raspberry Pi 4 with Coral USB Accelerator.
Model: Raspberry Pi 4 Model B Rev 1.5 and Coral USB Accelerator(Google Edge TPU ASIC)

Place following files in the examples folder:  /coral/pycoral/examples and the models in the folder /coral/pycoral/test_data


"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run object detection and measure FPS.")
parser.add_argument("--duration", type=int, default=10, help="Duration of the benchmark in seconds (default: 10)")
parser.add_argument("--imshow", type=str, default="yes", choices=["yes", "no"], help="Display video output (default: yes)")
args = parser.parse_args()

# Set model and labels
CPU_MODEL_PATH = "test_data/ssd_mobilenet_v2_coco.tflite"
TPU_MODEL_PATH = "test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite"
LABELS_PATH = "test_data/coco_labels.txt"

# Read labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Function to load a TFLite model
def load_model(model_path, use_tpu=False):
    if use_tpu:
        interpreter = tflite.Interpreter(model_path=model_path,
                                         experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    else:
        interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Perform object detection and measure FPS
def benchmark_detection(interpreter, duration, use_tpu=False, imshow=True):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    frame_count = 0
    start_time = time.time()

    test_mode = "Testing with TPU" if use_tpu else "Testing with CPU"

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
    
        # Preprocessing
        img = cv2.resize(frame, (input_shape[1], input_shape[2]))
        img = np.expand_dims(img, axis=0).astype(np.uint8)

        if use_tpu:
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
        else:
            img = img.astype('float32')  # Convert to FLOAT32
            img = img / 255.0  # Normalize to [0, 1] if required by the model
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            scores = interpreter.get_tensor(output_details[1]['index'])[0]
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        if imshow:
            # Display test mode and FPS on the frame
            cv2.putText(frame, f"{test_mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return fps

# Run benchmark
duration = args.duration
imshow = args.imshow == "yes"

print("\nLoading TPU model...")
tpu_interpreter = load_model(TPU_MODEL_PATH, use_tpu=True)
print("Starting benchmark for TPU...")
tpu_fps = benchmark_detection(tpu_interpreter, duration=duration, use_tpu=True, imshow=imshow)
print(f"FPS with TPU: {tpu_fps:.2f}")

print("Loading CPU model...")
cpu_interpreter = load_model(CPU_MODEL_PATH, use_tpu=False)
print("Starting benchmark for CPU...")
cpu_fps = benchmark_detection(cpu_interpreter, duration=duration, use_tpu=False, imshow=imshow)
print(f"FPS without TPU: {cpu_fps:.2f}")

# Print final results
print("\n--- Results ---")
print(f"FPS without TPU: {cpu_fps:.2f}")
print(f"FPS with TPU: {tpu_fps:.2f}")
print(f"TPU acceleration: {tpu_fps / cpu_fps:.2f}x faster")
