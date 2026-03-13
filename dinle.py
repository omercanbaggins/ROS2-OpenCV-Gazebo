#!/usr/bin/env python3

import rclpy
import cv2
from rclpy.node import Node
from std_msgs.msg import String
from cv_bridge import CvBridge  # 1. Make sure this is imported!mport cv_bridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import os
import math
import numpy as np 
class dinle(Node):
        def __init__(self,name):
            super().__init__(name)
            self.get_logger().info("sss")
            self.listener = self.create_subscription(Image, '/camera/image_raw', self.imageCallBack, 10)
            self.listener2 = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

            self.thresh = 0
            self.windowName = "dff"
            self.bridge = CvBridge()
            print("asdsssssssss")

        def scan_callback(self, msg):
                closest_distance = float('inf')
                closest_angle_deg = 0.0
                blankImage = np.zeros((640,480,3),np.uint8)

                # Loop through all LiDAR measurements to find the closest object
                for i, distance in enumerate(msg.ranges):
                    # Ignore 0.0 or 'inf' values (which happen when the laser hits nothing or errors)
                    if 0.0 < distance < float('inf'):
                        
                            # Calculate the exact angle using the ROS 2 parameters
                            angle_rad = msg.angle_min + (i * msg.angle_increment)
                            closest_angle_deg = math.degrees(angle_rad)
                            x=60*distance*np.cos(angle_rad)
                            y= 60*distance*np.sin(angle_rad)
                            print(distance)

                            cv2.circle(blankImage, (250+int(x), 250+int(y)), 5, (255, 255, 255), 5)
                cv2.imshow("radar",blankImage)


                

           
        def imageCallBack(self,msg):
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow(self.windowName,cv_image)
            cv_image = self.processImage(cv_image)
            lines = cv2.HoughLinesP(cv_image,1,np.pi/180,20,minLineLength=5, maxLineGap=10)
            blankImage = np.zeros_like(cv_image)
            steering = 0 
            throttle = 0
            #print("asddddd")
            if lines is not None:
                for l in lines:
                    if l is not None and cv_image is not None:
                        x,y,x1,y1 = l[0]
                        cv2.line(blankImage,(x,y),(x1,y1),(255,255,255),thickness=5)
                        tan = x1-x/y1-y
                        degree = math.atan(tan)*180/np.pi
                        steering = degree
                        #print(steering)
            cv2.imshow("lines",blankImage)
            cv2.waitKey(10)
             
        def listener_callback(self, msg):
            self.get_logger().info('Gelen mesaj: %s' % msg.data)
            self.windowName = msg.data

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


def main():

    rclpy.init(args=None)   
    node = dinle("asdaaa")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
     main()


