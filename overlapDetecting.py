#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

class PhaseCorrelationDetector(Node):
    def __init__(self):
        super().__init__('phase_overlap_detector')
        self.get_logger().info("Node started: Precise Panorama Tracker (Current Left -> Previous Right)")
        
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.overlap_pub = self.create_publisher(Int32, '/overlap_value', 10)
        self.debug_img_pub = self.create_publisher(Image, '/overlap_debug_image', 10)
        self.diff_pub = self.create_publisher(Image, '/overlap_difference', 10)
        
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.turn_speed = 0.15 

        self.bridge = CvBridge()
        
        self.prev_gray_f32 = None
        self.cumulative_dx = 0.0
        self.numberofImagesSaved = 0
        
        self.last_saved_gray = None 
        
        self.is_paused = False
        self.pause_duration = 2.0 
        self.pause_timer = None   
        
        self.s = 40 
        self.search_window = 100 

        self.is_first_frame = True

        self.create_timer(0.1, self.cmd_vel_loop)

    def cmd_vel_loop(self):
        """Continuously publishes velocity commands to keep the motor watchdog happy."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        if self.is_paused:
            # If paused, constantly tell the wheels to stay at 0
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
        else:
            # If active, constantly tell the wheels to spin
            msg.twist.linear.x = 0.3
            
        self.cmd_vel_pub.publish(msg)

    def start_pause(self):
        self.get_logger().info(f"Taking a {self.pause_duration}-second pause... (Stopping motors)")
        self.is_paused = True
        self.pause_timer = self.create_timer(self.pause_duration, self.end_pause)

    def end_pause(self):    ###this timer destroys itself after 2 seconds (it can be adjusted by modifying value of self.pauseduration)
        self.get_logger().info("Pause finished. Resuming tracking... (Starting motors)")
        self.is_paused = False
        
        self.destroy_timer(self.pause_timer)
        self.pause_timer = None
        self.prev_gray_f32 = None

    def save_new_reference(self, current_frame, gray_frame, reason_msg):
        self.numberofImagesSaved += 1
        
        cv2.imwrite(f"{self.numberofImagesSaved}.png", current_frame.copy())
        self.get_logger().info(f"{reason_msg} Ana kare kaydedildi: {self.numberofImagesSaved}.png")
        
        self.cumulative_dx = 0.0
        self.last_saved_gray = gray_frame.copy()
        
        self.start_pause()

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.is_paused:
                cv2.putText(current_frame, "PAUSED / WAITING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("debug", current_frame)
                cv2.waitKey(1)
                return

            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            curr_gray_f32 = np.float32(gray)
            h, w = current_frame.shape[:2]
            
            if self.is_first_frame:
                self.is_first_frame = False
                self.prev_gray_f32 = curr_gray_f32
                self.save_new_reference(current_frame, gray, "İlk Kurulum.")
                return

            if self.prev_gray_f32 is None:
                self.prev_gray_f32 = curr_gray_f32
                return

            shift, response = cv2.phaseCorrelate(self.prev_gray_f32, curr_gray_f32)
            dx, dy = shift  ## we do not care y axis so dy is not used 
            self.cumulative_dx += abs(dx)  ## we need total shift
            self.overlap_pub.publish(Int32(data=int(self.cumulative_dx)))

            shot_taken = False
            debug_img = current_frame.copy()

            if self.cumulative_dx > (w * 0.65) and self.last_saved_gray is not None:
                template = gray[:, :self.s]  ##after calculating shift, small overlapping area may left so we 
                                            ## we add one more layer to optimize image,  ##template matching can detect small overlaping areas 
                search_area = self.last_saved_gray[:, w - self.search_window : w]
                                                                 
                res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                template_x = max_loc[0]
                confidence = max_val
                
                remaining_overlap = self.search_window - template_x
                
                if confidence > 0.6:
                    cv2.rectangle(debug_img, (0, 0), (self.s, h), (255, 0, 0), 2)
                    cv2.putText(debug_img, f"Overlap: {remaining_overlap}px", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    if remaining_overlap <= self.s + 2:
                        next_id = self.numberofImagesSaved + 1
                        
                        overlap_crop_color = current_frame[:, :remaining_overlap].copy()
                        cv2.imwrite(f"overlap_area_{next_id}.png", overlap_crop_color)
                        
                        cropped_main_image = current_frame[:, remaining_overlap:].copy()
                        cv2.imwrite(f"cropped_{next_id}.png", cropped_main_image)
                        
                        self.get_logger().info(f"Ekstra dosyalar kaydedildi: overlap_area_{next_id}.png ve cropped_{next_id}.png")
                        
                        self.save_new_reference(current_frame, gray, "Kusursuz Kenar Eşleşmesi!")
                        shot_taken = True

            if not shot_taken and self.cumulative_dx > (w * 0.95):
                self.save_new_reference(current_frame, gray, "Template Bulunamadı, Phase Sınırı Aşıldı.")

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
        # Send one last stop command before fully dying
        msg = TwistStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        node.cmd_vel_pub.publish(msg)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()