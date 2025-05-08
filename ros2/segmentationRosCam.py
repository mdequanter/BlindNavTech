import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from ultralytics import YOLO

class ROS2YOLOSegmenter(Node):
    def __init__(self):
        super().__init__('ros2_yolo_segmenter')

        # Load YOLOv8 segmentation model
        self.model = YOLO('ros2/models/botopiasim.pt', verbose=True)

        # Define colors per class (BGR format)
        self.classColors = {    
            'street': (150, 150, 150),       # Gray
            'crosswalk': (0, 255, 255),      # Yellow
            'atomium': (180, 105, 255),     # Pink
            'house': (0, 0, 255),              # Red
            'trafficlight': (0, 255, 0),           # Green
            'parking': (255, 0, 0),                # Blue
            # Add more class names and colors as needed
        }

        # Video writer setup
        self.output_path = 'ros2_yolo_output.mp4'
        self.out = None
        self.fps = 30
        self.frame_size = (640, 480)

        self.prev_time = time.time()

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Received empty frame.")
            return

        frame = cv2.resize(frame, self.frame_size)

        if self.out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            self.get_logger().info(f"Output video: {self.output_path}")

        # --- Measure inference time ---
        inference_start = time.time()
        results = self.model(frame, conf=0.5, verbose=False)
        inference_end = time.time()
        inference_time_ms = (inference_end - inference_start) * 1000  # milliseconds

        masked_frame = np.copy(frame)

        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i, mask_xy in enumerate(result.masks.xy):
                    class_id = int(result.boxes.cls[i].item())
                    class_name = self.model.names[class_id]

                    if class_name in self.classColors:
                        color = self.classColors[class_name]
                        points = np.array(mask_xy, dtype=np.int32)
                        cv2.fillPoly(masked_frame, [points], color=color)
                        cv2.polylines(masked_frame, [points], isClosed=True, color=(255, 255, 255), thickness=2)

                        if points.shape[0] > 0:
                            text_pos = tuple(points[0])
                            cv2.putText(
                                masked_frame,
                                class_name,
                                text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 255),
                                2,
                                lineType=cv2.LINE_AA
                            )

        # Display FPS and inference time
        display_frame = masked_frame.copy()
        current_time = time.time()
        fps_text = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        cv2.putText(display_frame, f"FPS: {fps_text:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Inference: {inference_time_ms:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Segmentation View", display_frame)
        self.out.write(masked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("User exit requested.")
            cv2.destroyAllWindows()
            rclpy.shutdown()
            exit()

def main(args=None):
    rclpy.init(args=args)
    node = ROS2YOLOSegmenter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.out:
            node.out.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("Video processing complete.")

if __name__ == '__main__':
    main()
