# !python3 -m pip install depthai opencv-python numpy

import os
import numpy as np
import cv2
import depthai as dai
import time

# Model and output file paths
YOLOV8N_MODEL = "segPersonalObjects.blob"  # Adjust path accordingly
OUTPUT_VIDEO = "oak-d-live_video.mp4"  # Adjust path accordingly


# Camera settings
FRAME_WIDTH, FRAME_HEIGHT = 640, 640
LABELS = ["Pot-hole"]

# Hardcoded YOLO model parameters
NN_CONFIG = {
    "classes": 1,
    "coordinates": 4,
    "anchors": [],  # Example values, adjust for your model
    "anchor_masks": {},  # Example values
    "iou_threshold": 0.5,
    "confidence_threshold": 0.5
}

def create_camera_pipeline(model_path):
    pipeline = dai.Pipeline()

    # Create the OAK-D camera node
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    camRgb.setVideoSize(FRAME_WIDTH, FRAME_HEIGHT)
    camRgb.setInterleaved(False)

    # Create neural network node
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork.setConfidenceThreshold(NN_CONFIG["confidence_threshold"])
    detectionNetwork.setNumClasses(NN_CONFIG["classes"])
    detectionNetwork.setCoordinateSize(NN_CONFIG["coordinates"])
    detectionNetwork.setAnchors(NN_CONFIG["anchors"])
    detectionNetwork.setAnchorMasks(NN_CONFIG["anchor_masks"])
    detectionNetwork.setIouThreshold(NN_CONFIG["iou_threshold"])
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Create output queues
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("video")
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("nn")

    # Linking nodes
    camRgb.video.link(detectionNetwork.input)
    camRgb.video.link(xoutRgb.input)
    detectionNetwork.out.link(xoutNN.input)

    return pipeline

def annotate_frame(frame, detections, fps):
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Create and start pipeline
pipeline = create_camera_pipeline(YOLOV8N_MODEL)

# Ensure output directory exists
# os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

with dai.Device(pipeline) as device:
    videoQ = device.getOutputQueue("video", maxSize=4, blocking=False)
    detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)

    fps = 30
    frame_count = 0
    start_time = time.time()

    # Video writer for saving the output
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

    while True:
        frame = videoQ.get().getCvFrame()  # Get frame from the camera

        frame = cv2.resize(frame, (640, 640))

        inDet = detectionNN.get()  # Get detection results



        detections = []
        if inDet is not None:
            detections = inDet.detections
            print("Detections:", detections)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        frame = annotate_frame(frame, detections, fps)

        cv2.imshow("OAK-D Live Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cv2.destroyAllWindows()

print(f"[INFO] Processed live stream and saved to {OUTPUT_VIDEO}")
