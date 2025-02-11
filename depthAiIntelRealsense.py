import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import time
import json

# === Initialize RealSense Pipeline ===
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline
pipeline.start(config)

# Configure depth sensor for better accuracy
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)  # High accuracy mode

# Initialize serial communication
ser = serial.Serial(port='COM4', baudrate=9600, timeout=1)
time.sleep(2)
ser.write(b'O')  # Disable avoidance with ultrasonic sensor
time.sleep(1)
ser.write(b'L')
time.sleep(1)

scaled_x = 320
scaled_y = 240
lastDepthServo = 99
depthToServo = 0

# Function to scale values
def scale_value(value, source_min, source_max, target_min, target_max):
    return int((target_max - target_min) / (source_max - source_min) * (value - source_min) + target_min)

def get_min_depth_value(depth_frame, x, y):
    """ Get the minimum depth value around the (x, y) region """
    x_min = max(0, x - 10)
    x_max = min(639, x + 10)
    y_min = max(0, y - 10)
    y_max = min(479, y + 10)

    roi = depth_frame[y_min:y_max+1, x_min:x_max+1]
    nonzero_values = roi[roi > 0]

    return np.min(nonzero_values) if nonzero_values.size > 0 else 0

# Main loop
try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalize depth map for visualization
        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Read from serial (joystick or external input)
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            if data:
                try:
                    parsed_data = json.loads(data)
                    x = parsed_data.get("X", 0)
                    y = parsed_data.get("Y", 0)

                    # Scale X and Y to 640x480
                    scaled_x = scale_value(x, 0, 1023, 639, 0)
                    scaled_y = scale_value(y, 0, 1023, 0, 479)

                    depth_value = get_min_depth_value(depth_image, scaled_x, scaled_y)

                    if depth_value > 0:
                        depthToServo = scale_value(depth_value, 0, 5000, 0, 9)
                        depthToServo = min(9, max(0, depthToServo))  # Clamp value between 0 and 9

                        if lastDepthServo != depthToServo:
                            ser.write(str(depthToServo).encode())
                            lastDepthServo = depthToServo
                            print(f"Depth to servo: {depthToServo}")

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {data}")

        # Draw overlay on the RGB image
        box_size = 5
        cv2.rectangle(color_image, (scaled_x - box_size, scaled_y - box_size), 
                      (scaled_x + box_size, scaled_y + box_size), (0, 255, 0), 2)
        cv2.putText(color_image, f"{depthToServo:.2f}", (scaled_x + 10, scaled_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the RGB image and depth map
        cv2.imshow("RGB Camera Stream", color_image)
        cv2.imshow("Depth Map", depth_display)

        if cv2.waitKey(1) == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
