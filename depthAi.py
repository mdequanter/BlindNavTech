import depthai as dai
import numpy as np
import cv2

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

        

        # Coördinaten waar de afstand moet worden gemeten
        x, y = 320, 380 
        yDistance, aValue, status = detect_object_simple(x, y, depth_map)

        # Teken een klein kadertje rond de gemeten positie
        box_size = 5  # Grootte van het vierkant
        cv2.rectangle(depth_display, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)
        cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)

        # Voeg tekstoverlay toe met de meting
        label = f"{yDistance:.2f}"
        cv2.putText(depth_display, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Toon de dieptekaart met de overlays
        cv2.imshow("Depth Map", depth_display)


        # Toon de RGB-camera stream
        cv2.imshow("RGB Camera Stream", frame)        

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
