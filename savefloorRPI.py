import cv2
import depthai as dai
import numpy as np
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

# Buffer voor gemiddelde berekening
num_frames = 30
frame_buffer = []

# Start het apparaat en de pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        # Haal de dieptekaart op
        inDepth = depthQueue.get()
        depth_map = inDepth.getFrame()

        inVideo = videoQueue.get()
        frame = inVideo.getCvFrame()
        frame = cv2.resize(frame, (640, 400))

        # Voeg frame toe aan buffer en beperk grootte
        frame_buffer.append(depth_map)
        if len(frame_buffer) > num_frames:
            frame_buffer.pop(0)
        
        # Bereken gemiddelde per pixel
        avg_depth_map = np.mean(frame_buffer, axis=0)
        blurred_depth_map = cv2.GaussianBlur(avg_depth_map, (15, 15), 0)


        # Normaliseer de dieptekaart voor betere zichtbaarheid
        depth_display = cv2.normalize(avg_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)  # Voeg kleur toe

        # Normaliseer de dieptekaart voor betere zichtbaarheid
        depth_display2 = cv2.normalize(blurred_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display2 = cv2.applyColorMap(depth_display2, cv2.COLORMAP_JET)  # Voeg kleur toe


        # Toon de dieptekaart met de overlays
        # cv2.imshow("Depth Map", depth_display)
        # cv2.imshow("Depth Map2", depth_display2)


        # Toon de RGB-camera stream
        # cv2.imshow("RGB Camera Stream", frame)

        # Opslaan van gemiddelde dieptekaart als JSON
        avg_depth_list = avg_depth_map.tolist()
        with open("depth_map.json", "w") as f:
            json.dump({"average_depth_map": avg_depth_list}, f)
            print("Dieptekaart opgeslagen.")
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
