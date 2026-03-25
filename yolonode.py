import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        # Change this topic to match your Gazebo camera output
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # Downloads automatically on first run

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run YOLO inference
        results = self.model(cv_image, conf=0.5)
        
        # Visualize the results in a window
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()