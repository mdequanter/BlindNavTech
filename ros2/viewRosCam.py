'''
Records and displays video from a ROS2 camera topic.
keyboard controls:
#         r: start recording
#         p: pause recording
#         s: save snapshot   
#         q: quit program
'''


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
import os
from datetime import datetime

class ROS2VideoRecorder(Node):
    def __init__(self):
        super().__init__('ros2_video_recorder')

        self.fps = 30
        self.frame_size = (640, 480)
        self.output_path = 'ros2/recording/ros2_video_output.avi'
        self.out = None
        self.is_recording = False

        self.prev_time = time.time()
        self.current_frame = None  # For snapshot use

        os.makedirs('ros2/recording', exist_ok=True)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        self.get_logger().info(f"Recording started: {self.output_path}")
        self.is_recording = True

    def pause_recording(self):
        self.get_logger().info("Recording paused.")
        self.is_recording = False

    def stop_recording(self):
        if self.out:
            self.out.release()
            self.get_logger().info(f"Recording saved to: {self.output_path}")
        self.is_recording = False

    def save_snapshot(self):
        if self.current_frame is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'ros2/recording/snapshot_{timestamp}.jpg'
            cv2.imwrite(filename, self.current_frame)
            self.get_logger().info(f"Snapshot saved: {filename}")
        else:
            self.get_logger().warn("No frame available to save.")

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Received empty frame.")
            return

        frame = cv2.resize(frame, self.frame_size)
        self.current_frame = frame  # Update current frame for snapshot

        # Display
        display_frame = frame.copy()
        current_time = time.time()
        fps_text = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        cv2.putText(display_frame, f"FPS: {fps_text:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Live Camera Feed", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quitting program.")
            self.stop_recording()
            rclpy.shutdown()
        elif key == ord('r') and not self.is_recording:
            self.start_recording()
        elif key == ord('p') and self.is_recording:
            self.pause_recording()
        elif key == ord('s'):
            self.save_snapshot()

        if self.is_recording and self.out is not None:
            self.out.write(frame)

def main(args=None):
    rclpy.init(args=args)
    node = ROS2VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("Program ended.")

if __name__ == '__main__':
    main()
