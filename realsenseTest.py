import cv2
import pyrealsense2 as rs
import numpy as np
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

fps=0

# Function to get the minimum depth value in a small region
def get_min_depth_value(depth_frame, x, y):
    x_min = max(0, x - 10)
    x_max = min(639, x + 10)
    y_min = max(0, y - 10)
    y_max = min(479, y + 10)
    
    roi = depth_frame[y_min:y_max+1, x_min:x_max+1]
    nonzero_values = roi[roi > 0]
    return np.min(nonzero_values) if nonzero_values.size > 0 else 0

# Initialize FPS calculation
fps_start_time = time.time()
fps_counter = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Normalize depth data for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Define test coordinates (center of frame)
        x, y = 320, 240
        depth_value = get_min_depth_value(depth_image, x, y)
        
        # Draw a rectangle around the measured position
        box_size = 5
        cv2.rectangle(color_image, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)
        label = f"{depth_value:.2f} mm"
        cv2.putText(color_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS Calculation
        fps_counter += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = fps_counter / elapsed_time
            fps_start_time = time.time()
            fps_counter = 0
        
        # Display FPS
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(color_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show the RGB stream with overlay
        cv2.imshow("RealSense RGB Stream", color_image)
        
        # Exit loop on 'q' key press
        if cv2.waitKey(1) == ord("q"):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
