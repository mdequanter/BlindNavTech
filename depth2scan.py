''''
Uses the Oak D lite for transferring 3D depth to /scan topic

Launch rviz wit following command :  rviz2 -d dept2scan.rviz
'''



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import depthai as dai
import numpy as np
import time

class DepthToScan(Node):
    def __init__(self):
        super().__init__('depth_to_scan')
        self.publisher_ = self.create_publisher(LaserScan, 'scan', 10)

        # DepthAI pipeline
        pipeline = dai.Pipeline()
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)

        self.device = dai.Device(pipeline)
        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=5, blocking=False)

        self.timer = self.create_timer(1.0 / 10.0, self.timer_callback)  # 10 Hz

    def timer_callback(self):
        inDepth = self.depthQueue.get()
        depth_map = inDepth.getFrame()

        middle_row = depth_map[240]  # Neem de middelste horizontale rij
        ranges = []

        # Hoeken in graden
        angle_min = -np.pi / 2  # -90 graden
        angle_max = np.pi / 2   # +90 graden
        angle_increment = (angle_max - angle_min) / 121.0  # 121 waarden (60 + 1 + 60)

        # Rechts: pixels 0–59
        for i in range(60):
            dist = middle_row[i]
            ranges.append(self.convert_to_meters(dist))

        # Midden: pixel 320
        ranges.append(self.convert_to_meters(middle_row[320]))

        # Links: pixels 300–359
        for i in range(300, 360):
            dist = middle_row[i]
            ranges.append(self.convert_to_meters(dist))

        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'oakd_frame'
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = angle_increment
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / 10.0
        scan_msg.range_min = 0.2  # minimaal detecteerbare afstand in meter
        scan_msg.range_max = 4.0  # maximaal detecteerbare afstand in meter
        scan_msg.ranges = ranges

        self.publisher_.publish(scan_msg)
        self.get_logger().info('Published LaserScan')

    def convert_to_meters(self, value):
        if value == 0:
            return float('inf')  # onbekend
        return value / 1000.0  # DepthAI meet in millimeters

def main(args=None):
    rclpy.init(args=args)
    node = DepthToScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
main()
