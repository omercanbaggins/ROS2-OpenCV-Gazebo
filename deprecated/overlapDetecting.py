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
        self.get_logger().info("Node started: Overlap detecting (Advanced RANSAC & Filter)")
        
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        
        self.bridge = CvBridge()
        # 1. GÜÇLENDİRME: Nokta sayısını 2500'e çıkardık (daha fazla detay)
        self.orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.previous_image = None
        self.match_window = "Advanced Overlap Alignment"
        self.numberofImagesSaved = 0 

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            current_frame = current_frame[:,:400].copy()
            if self.previous_image is None:
                self.previous_image = current_frame
                cv2.imwrite(str(self.numberofImagesSaved)+".png",self.previous_image)

                return

            self.find_overlap_homography(self.previous_image, current_frame)
            cv2.imshow("Original Stream", current_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

    def find_overlap_homography(self, i1, i2):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # Hazırlık
        gray1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        gray2_full = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        # Örtüşme beklenen bölgeyi kırp (Sol %30'luk dilim)
        crop_width = int(i2.shape[1] * 0.55)
        gray2_cropped = gray2_full[:, :crop_width]
        cv2.imshow("croppedCurrent", gray2_cropped)
        cv2.imshow("old", gray1)

        # 2. GÜÇLENDİRME: CLAHE ile kontrast artırma (eşleşme kalitesini artırır)
        enhanced1 = clahe.apply(gray1)
        enhanced2 = clahe.apply(gray2_cropped)

        kp1, des1 = self.orb.detectAndCompute(enhanced1, None)
        kp2, des2 = self.orb.detectAndCompute(enhanced2, None)

        if des1 is not None and des2 is not None:
            # 3. GÜÇLENDİRME: KNN Match + Lowe's Ratio Test
            matches = self.bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    # 0.7 oranı daha sıkı ve güvenilirdir
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # 4. GÜÇLENDİRME: USAC_MAGSAC (En modern Homografi algoritması)
            if len(good_matches) > 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # USAC_MAGSAC, RANSAC'tan çok daha hızlı ve hatasızdır
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)

                if H is not None:
                    # Sadece doğru (inlier) noktaları filtrele
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                    
                    # Ortalama X kaymasını (offset) hesapla ve yayınla
                    offset_x = H[0, 2]
                    self.overlap_pub.publish(Int32(data=int(abs(offset_x))))

                    # Görselleştirme (Sadece doğru eşleşenleri çiz)
                    vis_img = cv2.drawMatches(i1, kp1, i2[:, :crop_width], kp2, 
                                            inlier_matches[:50], None, 
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow(self.match_window, vis_img)
                else:
                    self.get_logger().warn("Homografi hesaplanamadı.")
            else:
                self.get_logger().warn("Eşleşme sayısı yetersiz.")
                # Referansı güncelle (Takılmaları önler)
                self.previous_image = i2
                self.numberofImagesSaved+=1
                p= str(self.numberofImagesSaved)
                cv2.imwrite(p+".png", i2)

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
