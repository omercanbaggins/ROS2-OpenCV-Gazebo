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
        self.get_logger().info("Node started: Overlap detecting (Index Fixed)")
        
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.previous_image = None
        self.match_window = "Overlap Alignment"

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_frame = cv2.resize(current_frame, (640, 480))
            
            if self.previous_image is None:
                self.previous_image = current_frame
                return

            self.find_overlap_homography(self.previous_image, current_frame)
            cv2.imshow("Current Stream", current_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {str(e)}")

    def find_overlap_homography(self, i1, i2):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # 1. Hazırlık ve Kırpma
        gray1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gray2_full = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        gray2_cropped = gray2_full[:, 200:400] # Sağ %25'lik dilim
        
        enhanced1 = clahe.apply(gray1)
        enhanced2 = clahe.apply(gray2_cropped)
        cv2.imshow("cropped",gray2_cropped)
        # 2. Keypoint ve Descriptor
        kp1, des1 = self.orb.detectAndCompute(enhanced1, None)
        kp2, des2 = self.orb.detectAndCompute(enhanced2, None)

        if des1 is not None and des2 is not None:
            # SIRALAMA DEĞİŞTİ: des1 (Query), des2 (Train)
            knn_matches = self.bf.knnMatch(des1, des2, k=2)

            # 3. Ratio Test
            good_matches = []
            for m_n in knn_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # 4. Homografi ve Çizim
            if len(good_matches) > 5:
                # m.queryIdx -> des1 (kp1), m.trainIdx -> des2 (kp2)
                points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

                # Görselleştirme için i2'nin de kırpılmış halini kullan
                i2_visual_crop = i2[:, 480:]
                
                # ÇİZİM: i1 -> kp1 (query), i2_visual_crop -> kp2 (train)
                # Sıralama knnMatch ile aynı olduğu için hata vermez
                match_vis = cv2.drawMatches(
                    i1, kp1, i2_visual_crop, kp2, 
                    good_matches[:50], 
                    None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow(self.match_window, match_vis)
            else:
                self.get_logger().warn("Eşleşme az, referans güncelleniyor...")
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