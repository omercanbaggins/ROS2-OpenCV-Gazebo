import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist 
import cv2
from rclpy.qos import qos_profile_sensor_data
import numpy as np


class CameraListener(Node):
    def __init__(self):
        super().__init__('camera_listener_node')
        #self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Subscribing to the bridged topic using the high-bandwidth QoS profile
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)
            
        self.bridge = CvBridge()
        self.get_logger().info("Communication established. Processing visual matrix...")
    def getGrayScale(self,img):
        if img is not None:     
            return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  
        else: return img
    def processImage(self,img):
        blurred = cv2.GaussianBlur(self.getGrayScale(img),(5,15),5)
        _,threshImg = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
        if(_):
            CannyImg = cv2.Canny(threshImg,50,150)
            return CannyImg

    def image_callback(self, msg):
        try:
            # Reconstructing the 1D byte array into a 3D OpenCV matrix (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = self.processImage(cv_image)
            # Rendering the matrix to the screen
            lines = cv2.HoughLinesP(cv_image,1,np.pi/180,20,minLineLength=5, maxLineGap=10)
            blankImage = np.zeros_like(cv_image)
            steering = 0 
            throttle = 0
        
            for l in lines:
                if l is not None and cv_image is not None:
                    x,y,x1,y1 = l[0]
                    cv2.line(blankImage,(x,y),(x1,y1),(255,255,255))
                    tan = x1-x/y1-y
                    degree = np.atan(tan)*180/np.pi
                    steering = degree
                    throttle = 0.2
                    
                
            #cmd = Twist()

            # Example Logic: Move forward and turn slightly left
            #cmd.linear.x = throttle  # Linear velocity in m/s
            #cmd.angular.z = steering # Angular velocity in rad/s

            # 3. PUBLISH THE COMMAND
            #self.vel_pub.publish(cmd)
            cv2.imshow("TurtleBot3 Visual Feed", cv_image)

            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Matrix transformation error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraListener()
    print("zaa")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
