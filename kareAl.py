#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from geometry_msgs.msg import TwistStamped

class kareAL(Node):
        def __init__(self,name):
            super().__init__(name)
            self.get_logger().info("hiz yolla")
            self.publisher = self.create_publisher(TwistStamped, '/cmd_vel',10)
            self.timer = self.create_timer(0.1,self.velocitySend)
            self.msg = TwistStamped()

        def velocitySend(self):
             # 3. TwistStamped requires a header with a timestamp and a frame ID
            self.msg.header.stamp = self.get_clock().now().to_msg()
            self.msg.header.frame_id = 'base_link' 
            
            # 4. The velocities are now nested inside the 'twist' attribute
            self.msg.twist.linear.x = 0.0
            self.msg.twist.angular.z = 0.0
            self.publisher.publish(self.msg)
        def stop_robot(self):
            self.get_logger().info("Kapatiliyor... Robot durduruluyor!")
            
            # Create a fresh, empty message (all values default to 0.0)
            stop_msg = TwistStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link' 
            
            # Explicitly set velocities to zero
            stop_msg.twist.linear.x = 0.0
            stop_msg.twist.angular.z = 0.0
            
            # Publish the stop command
            self.publisher.publish(stop_msg)
             

def main():

    rclpy.init(args=None)   
    node = kareAL("kareAlici")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
     main()