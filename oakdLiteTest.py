import cv2
import depthai as dai
import numpy as np
import time

# Instellen van de DepthAI pipeline
pipeline = dai.Pipeline()

# Creëer de stereo diepte module
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Creér de RGB camera
colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setFps(15)

# Schakel autofocus uit en stel de focusafstand in op 1.3 meter
#focus_distance = 60  # 1.3 meter in millimeters (DepthAI gebruikt een schaal van 0-255)
#colorCam.initialControl.setManualFocus(focus_distance)

# Stel de camera resolutie en frames in
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configuratie van stereo diepte
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

# Link camera's naar stereo diepte module
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Uitvoer van de dieptekaart
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# RGB output stream
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")
colorCam.video.link(xoutVideo.input)

scaled_x = 320
scaled_y = 240

# Function to scale values
def scale_value(value, source_min, source_max, target_min, target_max):
    return int((target_max - target_min) / (source_max - source_min) * (value - source_min) + target_min)


with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=5, blocking=False)
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        inDepth = depthQueue.get()
        depth_map = inDepth.getFrame()

        inVideo = videoQueue.get()
        frame = inVideo.getCvFrame()
        frame = cv2.resize(frame, (640, 400))

        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        x, y = scaled_x, scaled_y 
        depth_value = depth_map[scaled_y, scaled_x]

        box_size = 5  
        cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)

        label = f"{depth_value:.2f}"
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("RGB Camera Stream", frame)        

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()