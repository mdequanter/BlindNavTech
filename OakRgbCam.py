import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import depthai as dai
import cv2
from cv_bridge import CvBridge

class OakRgbPublisher(Node):
    def __init__(self):
        super().__init__('oak_rgb_publisher')
        self.publisher_ = self.create_publisher(Image, 'oakRgb', 10)
        self.bridge = CvBridge()

        # Setup DepthAI pipeline
        pipeline = dai.Pipeline()
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setPreviewSize(640, 480)

        xoutRgb.setStreamName("rgb")
        camRgb.preview.link(xoutRgb.input)

        self.device = dai.Device(pipeline)
        self.rgbQueue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30 FPS

    def timer_callback(self):
        inRgb = self.rgbQueue.get()
        frame = inRgb.getCvFrame()

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'oak_rgb_frame'

        self.publisher_.publish(img_msg)
        self.get_logger().info('Published RGB frame')

def main(args=None):
    rclpy.init(args=args)
    node = OakRgbPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
main()
