import cv2
import depthai as dai
import numpy as np

import serial
import time
import json


# Instellen van de DepthAI pipeline
pipeline = dai.Pipeline()

# Creëer de stereo diepte module
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Creëer de RGB camera
colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setFps(30)

# Schakel autofocus uit en stel de focusafstand in op 1.3 meter
focus_distance = 60  # 1.3 meter in millimeters (DepthAI gebruikt een schaal van 0-255)
colorCam.initialControl.setManualFocus(focus_distance)

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


ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
time.sleep(2)
# disable avoidqance with ultrasonic sensor
ser.write(b'O')
time.sleep(1)
ser.write(b'L')
time.sleep(1)
scaled_x = 320
scaled_y = 240
lastDepthServo = 99

# Function to scale values
def scale_value(value, source_min, source_max, target_min, target_max):
    return int((target_max - target_min) / (source_max - source_min) * (value - source_min) + target_min)

def get_min_depth_value(depth_map, x, y):
    # Define valid bounds while ensuring we stay within array limits
    x_min = max(0, x - 10)
    x_max = min(639, x + 10)
    y_min = max(0, y - 10)
    y_max = min(399, y + 10)

    # Extract the region of interest (ROI)
    roi = depth_map[y_min:y_max+1, x_min:x_max+1]

    # Filter out zero values
    nonzero_values = roi[roi > 0]

    # Return the minimum nonzero value or None if all values are zero
    return np.min(nonzero_values) if nonzero_values.size > 0 else 0



# Start het apparaat en de pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    def detect_object_simple(x, y, depth_map, camera_height=1250):
        """Detecteert objecten op basis van de dieptekaart van de OAK-D Lite."""
        
        # Controleer of de pixel binnen het beeld valt
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            a = depth_map[y, x]   # Omzetten van mm naar meters
        else:
            return 0, 0, "Geen geldige coördinaten"

        # Interpoleer de y-afstand
        yDistance = np.interp(y, [0, depth_map.shape[0]], [0, 1350])

        # Bereken de theoretische afstand met de stelling van Pythagoras
        pythagoras_distance = np.sqrt(camera_height**2 + yDistance**2)

        yDistance =  (depth_map[y, x])

        aValue = (a - pythagoras_distance)

        # Controleer of de gemeten afstand significant afwijkt
        if aValue < 0:
            return yDistance, aValue, "Object"
        elif aValue > 0:
            return yDistance, aValue, "Kuil"
        else:
            return 0, 0, "Geen object"
        





    while True:
        # Haal de dieptekaart op
        inDepth = depthQueue.get()
        depth_map = inDepth.getFrame()

        inVideo = videoQueue.get()
        frame = inVideo.getCvFrame()
        frame = cv2.resize(frame, (640, 400))


        # Normaliseer de dieptekaart voor betere zichtbaarheid
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)  # Voeg kleur toe

        if ser.in_waiting > 0:
            # Read the available data
            data = ser.readline().decode('utf-8').strip()  # Read a line and decode
            #print(f"Received: {data}")
        
            if data:  # Ensure data is not empty
                try:
                    parsed_data = json.loads(data)  # Parse JSON string
                    x = parsed_data.get("X", 0)
                    y = parsed_data.get("Y", 0)
                    
                    # Scale X and Y to 640x480
                    scaled_x = scale_value(x, 0, 1023, 639, 0)
                    scaled_y = scale_value(y, 0, 1023, 0, 399)
                    
                    depth_value = get_min_depth_value(depth_map, scaled_x, scaled_y)

                    if (depth_value > 0) :
                        #depth_value = depth_map[scaled_y, scaled_x]  # Remember: NumPy uses (row, column) -> (y, x)
                        depthToServo = scale_value(depth_value, 0,5000, 0, 9)
                        if (lastDepthServo != depthToServo and depthToServo > 0) :
                            ser.write(f'{{{depthToServo}}}'.encode())
                            lastDepthServo = depthToServo
                            #print(f"Depth at ({x}, {y}): {depth_value}")  
                            print (f"Depth to servo: {depthToServo}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {data}")

        

        # Coördinaten waar de afstand moet worden gemeten
        x, y = scaled_x, scaled_y 

        # print(f"Original: ({x}, {y}) -> Scaled: ({scaled_x}, {scaled_y})")
        depth_value = depth_map[scaled_y, scaled_x]  # Remember: NumPy uses (row, column) -> (y, x)

        # Teken een klein kadertje rond de gemeten positie
        box_size = 20  # Grootte van het vierkant
        #cv2.rectangle(depth_display, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)
        cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)

        # Voeg tekstoverlay toe met de meting
        label = f"{depth_value:.2f}"
        #cv2.putText(depth_display, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Toon de dieptekaart met de overlays
        # cv2.imshow("Depth Map", depth_display)


        # Toon de RGB-camera stream
        cv2.imshow("RGB Camera Stream", frame)        

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
