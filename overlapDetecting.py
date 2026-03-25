#!/usr/bin/env python3

import rclpy
import cv2
from rclpy.node import Node
from std_msgs.msg import String
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
import os
import math
import numpy as np 

class dinle(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("Node started: overlap detecting with ORB + RANSAC")
        
        # ROS 2 Setup
        self.listener = self.create_subscription(Image, '/camera/image_raw', self.imageCallBack, 10)
        self.bridge = CvBridge()
        
        # State Variables
        self.previousImage = None
        self.current = None
        self.windowName = "overlapDetect"
        
        # Initialize ORB Detector (Industry Standard)
        # We do this once here to save CPU power
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Timer for reference frame update (every 5 seconds)
        self.create_timer(5, self.timerfunc)

    def timerfunc(self):
        self.previousImage = self.current

    def imageCallBack(self, msg):
        # 1. Convert and Resize
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (640, 480))
        
        # 2. Show Main Stream
        cv2.imshow(self.windowName, cv_image)

        # 3. Logic: Compare Bitwise (Your original logic)
        self.compareBitWise(self.previousImage, cv_image)
        
        # 4. Logic: Find Overlap with Homography (New industrial method)
        
        # 5. Update State and Process
        self.current = cv_image
        processed = self.processImage(cv_image)
        if processed is not None:
            cv2.imshow("CannyEdges", processed)
            
        cv2.waitKey(10)

    def findOverlapHomography(self, i1, i2):
        # Find keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(i1, None)
        kp2, des2 = self.orb.detectAndCompute(i2, None)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, m in enumerate(matches):
                    points1[i, :] = kp1[m.queryIdx].pt
                    points2[i, :] = kp2[m.trainIdx].pt

                # Calculate Homography with RANSAC
                H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

                if H is not None:
                    draw_params = dict(matchColor=(0, 255, 0),
                                     singlePointColor=None,
                                     matchesMask=mask.ravel().tolist(),
                                     flags=2)
                    
                    match_vis = cv2.drawMatches(i1, kp1, i2, kp2, matches[:50], None, **draw_params)
                    cv2.imshow("IndustrialAlignment", match_vis)

    def compareBitWise(self, i1, i2):
        if i1 is not None and i2 is not None:
            resultImg = cv2.bitwise_and(i1, i2)
            diff = cv2.absdiff(i1, i2)
            cv2.imshow("BitwiseAND", resultImg)
            cv2.imshow("DifferenceMap", diff)
            cv2.imshow("old",i1)

    def getGrayScale(self, img):
        if img is not None:     
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        return img

    def processImage(self, img):
        gray = self.getGrayScale(img)
        if gray is not None:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshImg = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            cannyImg = cv2.Canny(threshImg, 50, 150)
            return cannyImg
        return None

def main():
    rclpy.init(args=None)   
    node = dinle("overlap_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()