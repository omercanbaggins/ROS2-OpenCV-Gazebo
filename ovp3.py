#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

class PhaseCorrelationDetector(Node):
    def __init__(self):
        super().__init__('phase_overlap_detector')
        self.get_logger().info("Node started: Precise Panorama Tracker (Current Left -> Previous Right)")
        
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        self.debug_img_pub = self.create_publisher(Image, '/overlap_debug_image', 10)
        self.diff_pub = self.create_publisher(Image, '/overlap_difference', 10)
        
        self.bridge = CvBridge()
        
        self.prev_gray_f32 = None
        self.cumulative_dx = 0.0
        self.numberofImagesSaved = 0
        
        # State tracking
        self.last_saved_gray = None 
        
        self.s = 40 # The small area 'S' (width of the template on the left edge)
        self.search_window = 100 # How much of the previous frame's right edge we check
        
    def save_new_reference(self, current_frame, gray_frame, reason_msg):
        """Saves the FULL uncropped image and resets the tracking math."""
        self.numberofImagesSaved += 1
        
        # Save the full, uncropped main image (e.g., '1.png', '2.png')
        cv2.imwrite(f"{self.numberofImagesSaved}.png", current_frame.copy())
        self.get_logger().info(f"{reason_msg} Ana kare kaydedildi: {self.numberofImagesSaved}.png")
        
        # Reset trackers and save the new reference for the next match
        self.cumulative_dx = 0.0
        self.last_saved_gray = gray_frame.copy()

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_f32 = np.float32(gray)
            
            h, w = current_frame.shape[:2]
            
            # 1. Initialization (First Frame)
            if self.prev_gray_f32 is None:
                self.prev_gray_f32 = curr_gray_f32
                # For the very first frame, there is no "overlap" or "cropped" version, just the main shot
                self.save_new_reference(current_frame, gray, "İlk Kurulum.")
                return

            # Difference Map (Visual Debugging)
            #diff_img = cv2.absdiff(self.last_saved_gray, gray)
            #self.diff_pub.publish(self.bridge.cv2_to_imgmsg(diff_img, encoding="mono8"))

            # 2. Global Phase Shift (Runs constantly as backup)
            shift, response = cv2.phaseCorrelate(self.prev_gray_f32, curr_gray_f32)
            dx, dy = shift
            self.cumulative_dx += abs(dx)
            self.overlap_pub.publish(Int32(data=int(self.cumulative_dx)))

            shot_taken = False
            debug_img = current_frame.copy()

            # Only start checking the seam if we've panned at least 70% of the screen width
            if self.cumulative_dx > (w * 0.70) and self.last_saved_gray is not None:
                
                # TEMPLATE: Leftmost 's' pixels of the CURRENT frame
                template = gray[:, :self.s]
                
                # SEARCH AREA: Rightmost pixels of the PREVIOUS frame
                search_area = self.last_saved_gray[:, w - self.search_window : w]
                
                # Search for current left inside previous right
                res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                template_x = max_loc[0]
                confidence = max_val
                
                # Calculate exactly how many overlapping pixels remain between the two frames
                remaining_overlap = self.search_window - template_x
                
                if confidence > 0.6:
                    # Draw visual debug on the left side showing the matched area
                    cv2.rectangle(debug_img, (0, 0), (self.s, h), (255, 0, 0), 2)
                    cv2.putText(debug_img, f"Overlap: {remaining_overlap}px", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # If the remaining overlap has shrunk to just the template width (plus a tiny 2px buffer)
                    if remaining_overlap <= self.s + 2:
                        
                        next_id = self.numberofImagesSaved + 1
                        
                        # ---> A. SAVE THE COLORED OVERLAP AREA TO DISK <---
                        overlap_crop_color = current_frame[:, :remaining_overlap].copy()
                        cv2.imwrite(f"overlap_area_{next_id}.png", overlap_crop_color)
                        
                        # ---> B. SAVE THE CROPPED MAIN IMAGE <---
                        cropped_main_image = current_frame[:, remaining_overlap:].copy()
                        cv2.imwrite(f"cropped_{next_id}.png", cropped_main_image)
                        
                        self.get_logger().info(f"Ekstra dosyalar kaydedildi: overlap_area_{next_id}.png ve cropped_{next_id}.png")
                        
                        # ---> C. SAVE THE FULL IMAGE & RESET <---
                        self.save_new_reference(current_frame, gray, "Kusursuz Kenar Eşleşmesi!")
                        shot_taken = True

            # 4. Phase Math Failsafe
            if not shot_taken and self.cumulative_dx > (w * 0.95):
                # If template fails completely, we still save a shot to prevent getting lost,
                # but we don't save cropped/overlap files because we don't have exact pixel data.
                self.save_new_reference(current_frame, gray, "Template Bulunamadı, Phase Sınırı Aşıldı.")

            # Debug Overlay
            shift_int = int(self.cumulative_dx)
            cv2.rectangle(debug_img, (0, 0), (shift_int, 20), (0, 255, 0), -1)
            cv2.putText(debug_img, f"Phase Shift: {shift_int}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("debug", debug_img)
            cv2.waitKey(1)
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))

            self.prev_gray_f32 = curr_gray_f32

        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PhaseCorrelationDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cleanly...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()