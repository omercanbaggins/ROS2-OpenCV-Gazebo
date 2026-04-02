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
        self.get_logger().info("Node started: Phase Tracker with Difference Output")
        
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        self.debug_img_pub = self.create_publisher(Image, '/overlap_debug_image', 10)
        
        # NEW: Publisher for the difference image
        self.diff_pub = self.create_publisher(Image, '/overlap_difference', 10)
        
        self.bridge = CvBridge()
        
        self.prev_gray_f32 = None
        self.cumulative_dx = 0.0
        self.numberofImagesSaved = 0
        
        # NEW: Keep a copy of the exact frame we last saved to disk
        self.last_saved_gray = None 
        
    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_f32 = np.float32(gray)
            
            # 1. Initialization (First Frame)
            if self.prev_gray_f32 is None:
                self.prev_gray_f32 = curr_gray_f32
                self.last_saved_gray = gray.copy() # Save the reference for the difference check
                
                self.numberofImagesSaved += 1
                cv2.imwrite(f"{self.numberofImagesSaved}.png", current_frame.copy())
                self.get_logger().info(f"İlk referans kaydedildi: {self.numberofImagesSaved}.png")
                return

            # --- NEW: CALCULATE AND PUBLISH THE DIFFERENCE ---
            # This subtracts the current live frame from the last picture we saved
            diff_img = cv2.absdiff(self.last_saved_gray, gray)
            
            # Publish as a grayscale (mono8) image so you can view it in rqt_image_view
            diff_msg = self.bridge.cv2_to_imgmsg(diff_img, encoding="mono8")
            self.diff_pub.publish(diff_msg)
            # -------------------------------------------------

            # 2. Calculate Shift (Frame to Frame)
            shift, response = cv2.phaseCorrelate(self.prev_gray_f32, curr_gray_f32)
            dx, dy = shift
            
            self.cumulative_dx += abs(dx)
            self.overlap_pub.publish(Int32(data=int(self.cumulative_dx)))
            
            # 3. Check if we need a new shot
            image_width = current_frame.shape[1]
            max_allowed_shift = image_width * 0.85 
            
            if self.cumulative_dx > max_allowed_shift:
                self.numberofImagesSaved += 1
                cv2.imwrite(f"{self.numberofImagesSaved}.png", current_frame.copy())
                
                self.get_logger().info(
                    f"Örtüşme bitti. Yeni kare kaydedildi: {self.numberofImagesSaved}.png "
                    f"(Toplam Kayma: {int(self.cumulative_dx)}px)"
                )
                
                self.cumulative_dx = 0.0
                
                # NEW: Update the reference image so the difference goes back to black
                self.last_saved_gray = gray.copy() 

            # 4. Visual Debugging (Progress Bar)
            debug_img = current_frame.copy()
            shift_int = int(self.cumulative_dx)
            cv2.rectangle(debug_img, (0, 0), (shift_int, 20), (0, 255, 0), -1)
            cv2.putText(debug_img, f"Shift Accumulation: {shift_int}/{int(max_allowed_shift)}px", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))

            # 5. Update previous frame for tracking
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()