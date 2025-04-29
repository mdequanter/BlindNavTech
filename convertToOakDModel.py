from ultralytics import YOLO



# Step 1: Convert YOLOv8 Model to ONNX

# Load YOLOv8 model

model = YOLO("models\pathfinderYolo8Seg.pt")

# Export to ONNX format

# model.export(format="onnx")
model.export(format="onnx", opset=10)



# Step 2: Convert ONNX Model to OpenVINO

# Convert to blob format via online tool:
# https://blobconverter.luxonis.com/

# Step 3: Run OpenVINO Model on DepthAI

