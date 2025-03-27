import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import depthai as dai
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud')
        self.publisher_ = self.create_publisher(PointCloud2, 'points', 10)

        # DepthAI pipeline setup
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

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("depth")
        stereo.depth.link(xout.input)

        self.device = dai.Device(pipeline)
        self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=5, blocking=False)

        self.timer = self.create_timer(1.0 / 10.0, self.timer_callback)  # 10 Hz

    def timer_callback(self):
        inDepth = self.depthQueue.get()
        depth_frame = inDepth.getFrame()
        height, width = depth_frame.shape

        # Fictieve camera parameters
        fx = 220  # brandpuntsafstand in pixels
        fy = 220
        cx = width // 2
        cy = height // 2

        # Maak pointcloud data
        points = []
        for y in range(0, height, 4):       # subsample elke 4 rijen
            for x in range(0, width, 4):    # subsample elke 4 kolommen
                z = depth_frame[y, x] / 1000.0  # mm naar m
                if z == 0:
                    continue
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                points.append((X, Y, z))

        # Header + sensor_msgs PointCloud2 aanmaak
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'oakd_frame'

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.publisher_.publish(cloud_msg)
        self.get_logger().info(f'Published PointCloud2 with {len(points)} points')

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
main()
