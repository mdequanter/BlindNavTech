import depthai as dai
import cv2

# Instellen van de DepthAI pipeline
pipeline = dai.Pipeline()

# Creëer de RGB camera
colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setFps(30)

# Schakel autofocus uit en stel de focusafstand in op 1.3 meter
focus_distance = 60  # 1.3 meter in millimeters (DepthAI gebruikt een schaal van 0-255)
colorCam.initialControl.setManualFocus(focus_distance)

# Uitvoer van de RGB stream
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")
colorCam.video.link(xoutVideo.input)

# Start het apparaat en de pipeline
with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        inVideo = videoQueue.get()
        frame = inVideo.getCvFrame()

        # Toon de RGB-camera stream
        cv2.imshow("RGB Camera Stream", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()