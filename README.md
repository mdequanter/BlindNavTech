# BlindNavTech
Code and research for a master's thesis on advanced obstacle detection for the visually impaired. Uses 3D LiDAR (Unitree Go2 PRO) and depth cameras (OAK-D, RealSense D435i) to detect low obstacles, curbs, and stairs. Aims to enhance mobility and safety with an affordable, AI-driven solution for real-world use.


# savefloor.py
The script initializes a DepthAI pipeline to capture and process stereo depth data and RGB video. It calculates an average depth map over multiple frames, normalizes it for visualization, and displays both the depth map and RGB video stream in real-time. Additionally, it saves the averaged depth map as a JSON file for further analysis. To ensure proper calibration, the OAK-D-Lite should be aimed at an angle of about 30 degrees downward so that the flat floor is visible without obstructions. This allows the system to calibrate the ground floor and use it as a reference for detecting obstacles.

# compareWithFloor.py
This script extends the previous pipeline by using the saved JSON depth map for real-time object detection. It initializes a DepthAI pipeline to capture stereo depth data and an RGB video stream, storing recent depth frames in a rolling buffer. The system compares the current depth map with the pre-recorded reference depth map, highlighting deviations as potential obstructions. Differences are visualized using a color-coded depth difference map, where blue indicates objects closer than before, and red indicates objects further away. A contour detection algorithm is applied to isolate and highlight significant changes, helping identify obstacles in real-time. The OAK-D-Lite should be aimed at a 30-degree downward angle to ensure a clear floor calibration for accurate obstacle detection.

# determinFreeDirection.py
This script builds upon the previous version by adding real-time navigation assistance for a blind user. It initializes a DepthAI pipeline to capture stereo depth data and an RGB video stream while using a stored reference depth map for obstacle detection. The script processes a rolling buffer of recent depth frames to detect deviations from the calibrated floor. Identified obstacles are highlighted using a color-coded depth difference map, where red indicates objects further away and blue indicates objects closer.

To guide the user, the script analyzes free space by inverting the detected obstacle mask and computing the centroid of the largest open area. A green arrow is dynamically drawn from the bottom-center of the frame toward the optimal free path, helping the user navigate safely. The OAK-D-Lite should be positioned at a 30-degree downward angle to ensure an accurate floor calibration, allowing for reliable obstacle detection and navigation assistance.

![image](https://github.com/user-attachments/assets/af8f7baa-f68f-4e48-bd3d-ee57cea6369a)
