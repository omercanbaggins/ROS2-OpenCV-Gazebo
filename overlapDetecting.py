#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

class OverlapDetector(Node):
    def __init__(self):
        super().__init__('overlap_detector_node')
        self.get_logger().info("Node started: Overlap detecting with KNN + Ratio Test")
        
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # FIX: For KNN Ratio test, we turn off crossCheck
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.previous_image = None
        self.window_name = "Current Stream"
        self.match_window = "Overlap Alignment"

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_frame = cv2.resize(current_frame, (640, 480))

            if self.previous_image is None:
                self.previous_image = current_frame
                return

            # Process Overlap
            self.find_overlap_homography(self.previous_image, current_frame)
            
            cv2.imshow(self.window_name, current_frame)
            cv2.imshow("Reference Frame (Old)", self.previous_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {str(e)}")

    def find_overlap_homography(self, i1, i2):
        # 1. Enhance Images with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        gray1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        
        # Use the enhanced versions!
        enhanced1 = clahe.apply(gray1)
        enhanced2 = clahe.apply(gray2)

        # 2. Detect and Compute
        kp1, des1 = self.orb.detectAndCompute(enhanced1, None)
        kp2, des2 = self.orb.detectAndCompute(enhanced2, None)

        if des1 is not None and des2 is not None:
            # 3. Use KNN Match (k=2) for Ratio Test
            knn_matches = self.bf.knnMatch(des1, des2, k=2)

            # 4. Apply Lowe's Ratio Test
            # This kills the 'crazy' lines connecting handles to wheels
            good_matches = []
            for m_n in knn_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            self.get_logger().info(f"Good Matches: {len(good_matches)}")

            # 5. Threshold Logic
            if len(good_matches) > 7:
                points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)

                if H is not None and mask is not None:
                    # Only draw the inliers identified by RANSAC
                    num_to_draw = min(len(good_matches), 50)
                    draw_mask = mask.ravel().tolist()[:num_to_draw]

                    match_vis = cv2.drawMatches(
                        i1, kp1, i2, kp2, 
                        good_matches[:num_to_draw], 
                        None, 
                        matchesMask=draw_mask,
                        flags=2
                    )
                    cv2.imshow(self.match_window, match_vis)
            else:
                # 6. Scene Change: If matches are too low, the camera has moved 
                # significantly. Update the reference image to the current view.
                self.get_logger().warn("Overlap lost. Updating reference frame.")
                self.previous_image = i2

def main(args=None):
    rclpy.init(args=args)
    node = OverlapDetector()
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