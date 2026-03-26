import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge  ###
import numpy as np
import cv2
class HeightEstimator(Node):
    def __init__(self):
        super().__init__('height_estimator')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.listener_callback,
            10)
        self.subscription2 = self.create_subscription(
            Image,
            '/camera/image',
            self.imageCallBack,
            10)
    def imageCallBack(self,msg):
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow("Depth View", depth_display)
        cv2.waitKey(1)


    def listener_callback(self, msg):
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        if len(points) == 0:
            return

        y_values = points['y'] #y is the downward
        
        current_height = np.percentile(y_values, 95) 

        self.get_logger().info(f'Estimated Height: {current_height:.3f} meters')

def main(args=None):
    rclpy.init(args=args)
    node = HeightEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()