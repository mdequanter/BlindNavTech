import cv2
import depthai as dai
import numpy as np
import json
import time
from collections import deque

import serial
import time


#ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
ser = serial.Serial(port='COM4', baudrate=9600, timeout=1)

time.sleep(2)

# disable avoidqance with ultrasonic sensor
ser.write(b'O')



# Number of frames to average
NUM_FRAMES = 5
THRESHOLD = 1
fps=0
lastDirection = 'F'
scaled_x = 320
scaled_y = 240

lastDepthServo = 0

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
#monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
#monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configuratie van stereo diepte
# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
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
# colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setFps(30)

# Schakel autofocus uit en stel de focusafstand in op 1.3 meter
focus_distance = 60  # 1.3 meter in millimeters (DepthAI gebruikt een schaal van 0-255)
colorCam.initialControl.setManualFocus(focus_distance)

# RGB output stream
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")
colorCam.video.link(xoutVideo.input)


# Function to scale values
def scale_value(value, source_min, source_max, target_min, target_max):
    return int((target_max - target_min) / (source_max - source_min) * (value - source_min) + target_min)


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

    # FPS meting
    prev_time = time.time()
    frame_count = 0

    while True:
        # Haal de huidige dieptekaart op
        # inDepth = depthQueue.get()

        try:
                inDepth = depthQueue.get()
                frame = inDepth.getFrame()

        except RuntimeError as e:
            print(f"⚠️ Warning: {e} (Skipping frame)")
            time.sleep(0.1)  # Small delay to avoid excessive errors


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

        if ser.in_waiting > 0:
            # Read the available data
            data = ser.readline().decode('utf-8').strip()  # Read a line and decode
            # print(f"Received: {data}")
        
            if data:  # Ensure data is not empty
                try:
                    parsed_data = json.loads(data)  # Parse JSON string
                    x = parsed_data.get("X", 0)
                    y = parsed_data.get("Y", 0)
                    
                    # Scale X and Y to 640x480
                    scaled_x = scale_value(x, 0, 1023, 639, 0)
                    scaled_y = scale_value(y, 0, 1023, 0, 479)
                    
                    # print(f"Original: ({x}, {y}) -> Scaled: ({scaled_x}, {scaled_y})")
                    depth_value = depth_map[scaled_y, scaled_x]  # Remember: NumPy uses (row, column) -> (y, x)
                    depthToServo = scale_value(depth_value, 0,2000, 0, 9)
                    if (lastDepthServo!=depthToServo and depthToServo != 0) :
                        ser.write(b'{depthToServo}')
                        lastDepthServo = depthToServo
                        print(f"Depth at ({x}, {y}): {depth_value}")            # Maak een kleurenmap: Groen = gelijk, Rood = verder weg, Blauw = dichterbij
                        print (f"Depth to servo: {depthToServo}")
    

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {data}")




        if saved_depth_map is not None:
            # Bereken het verschil tussen de gemiddelde en opgeslagen dieptekaart
            diff_map = avg_depth_map - saved_depth_map
            diff_map[np.abs(diff_map) < THRESHOLD] = 0

            color_map = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
            color_map[..., 2] = np.clip(-diff_map * 2, 0, 255).astype(np.uint8)  # Blauw voor dichterbij
            color_map[..., 1] = np.clip(diff_map * 2, 0, 255).astype(np.uint8)  # Rood voor verder weg

            # Threshold to isolate red areas
            lower_red = np.array([0, 0, 50])   # Ondergrens van rood in BGR
            upper_red = np.array([50, 50, 255]) # Bovengrens van rood in BGR
            mask = cv2.inRange(color_map, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Definieer minimum oppervlak threshold
            min_area = 600  # Pas aan indien nodig
            output = np.zeros_like(color_map)

            # Teken enkel contouren met een groot genoeg oppervlak
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.drawContours(output, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

            # Zoek het meest vrije pad
            free_space_map = cv2.bitwise_not(mask)  # Inverteer masker (vrije ruimte = wit)
            moments = cv2.moments(free_space_map)

            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])  # Zwaartepunt x-coördinaat
                cy = int(moments["m01"] / moments["m00"])  # Zwaartepunt y-coördinaat
            else:
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2  # Midden als fallback

            if (cx < 300 ) :
                if (lastDirection != 'L'):
                    print("ga links")
                    lastDirection = 'L'
                    ser.write(b'6')
            if (cx > 340 ) :
                if (lastDirection != 'R'):
                    print("ga rechts")
                    lastDirection = 'R'
                    ser.write(b'8')

            if (cx >= 300 and cx <= 340) :
                if (lastDirection != 'F'):
                    print("ga rechtdoor")
                    lastDirection = 'F'
                    ser.write(b'7')

            # Pijl start van onderaan in het midden en wijst naar vrije zone
            start_point = (frame.shape[1] // 2, frame.shape[0] - 10)
            end_point = (cx, cy)

            cv2.arrowedLine(output, start_point, end_point, (0, 255, 0), 3, tipLength=0.3)

            # Bereken FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time > 1.0:  # Update elke seconde
                fps = frame_count / elapsed_time
                frame_count = 0
                prev_time = current_time

            cv2.circle(output, (scaled_x, scaled_y), 10, (255, 100, 100), -1)
            # Toon de FPS op het beeld
            cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Toon de output met de pijl en FPS
            cv2.imshow("Detections Contours & Navigation Arrow", output)

        if cv2.waitKey(1) == ord("q"):
            ser.close()
            break

    cv2.destroyAllWindows()
ser.close()