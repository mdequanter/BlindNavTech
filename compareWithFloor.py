import depthai as dai
import numpy as np
import cv2
import json
from collections import deque

# Number of frames to average

NUM_FRAMES = 3
THRESHOLD = 1

# Initialize a rolling buffer for depth frames
depth_buffer = deque(maxlen=NUM_FRAMES)

# Instellen van de DepthAI pipeline
pipeline = dai.Pipeline()

# Creëer de stereo diepte module
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

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


# Creëer de RGB camera
colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setFps(30)


# Schakel autofocus uit en stel de focusafstand in op 1.3 meter
focus_distance = 60  # 1.3 meter in millimeters (DepthAI gebruikt een schaal van 0-255)
colorCam.initialControl.setManualFocus(focus_distance)

# RGB output stream
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")
colorCam.video.link(xoutVideo.input)



# Start het apparaat en de pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    
    # Laad de opgeslagen gemiddelde dieptekaart
    try:
        with open("depth_map.json", "r") as f:
            saved_data = json.load(f)
            saved_depth_map = np.array(saved_data["average_depth_map"], dtype=np.float32)
    except FileNotFoundError:
        print("Geen opgeslagen dieptekaart gevonden.")
        saved_depth_map = None
    
    while True:
        # Haal de huidige dieptekaart op
        inDepth = depthQueue.get()
        depth_map = inDepth.getFrame().astype(np.float32)

        inVideo = videoQueue.get()
        frame = inVideo.getCvFrame()
        frame = cv2.resize(frame, (640, 400))
        
        # Voeg de huidige dieptekaart toe aan de buffer
        depth_buffer.append(depth_map)
        
        if len(depth_buffer) == NUM_FRAMES:
            # Bereken de gemiddelde dieptekaart over de opgeslagen frames
            avg_depth_map = np.max(depth_buffer, axis=0)
        else:
            continue  # Wacht tot de buffer vol is
        
        if saved_depth_map is not None:
            # Bereken het verschil tussen de gemiddelde en opgeslagen dieptekaart


            diff_map = avg_depth_map - saved_depth_map
            
            diff_map[np.abs(diff_map) < THRESHOLD] = 0

            # Maak een kleurenmap: Groen = gelijk, Rood = verder weg, Blauw = dichterbij
            color_map = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
            color_map[..., 2] = np.clip(-diff_map*2, 0, 255).astype(np.uint8)  # Blauw voor dichterbij
            color_map[..., 1] = np.clip(diff_map*2, 0, 255).astype(np.uint8)  # Rood voor verder weg
            
            # Toon de kleurenmap

            # Threshold to isolate red areas
            # Adjust threshold values based on your image
            lower_red = np.array([0, 0, 50])   # Lower bound of red in BGR
            upper_red = np.array([50, 50, 255]) # Upper bound of red in BGR
            mask = cv2.inRange(color_map, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Define minimum area threshold
            min_area = 600  # Adjust as needed
            output = np.zeros_like(color_map)
            #output = frame.copy()

            # Draw only contours with area greater than the threshold
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    frame = cv2.drawContours(output, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

            #cv2.imshow("Depth Difference Map", color_map)
            cv2.imshow("Detections Contours", output)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
