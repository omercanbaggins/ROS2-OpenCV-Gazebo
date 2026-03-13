#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
import random
class denemeNode(Node):
        def __init__(self,name):
            super().__init__(name)
            self.get_logger().info("deneme")
            self.create_timer(1.0,self.callback)
            self.publisher_ = self.create_publisher(String, 'chatter', 10)
        def callback(self):
            msg = String()
            
            msg.data = str(random.randint(5,11))
            self.publisher_.publish(msg)
            self.get_logger().info("yolladim:"+str(msg.data))

def main():

    rclpy.init(args=None)   
    node = denemeNode("asds") # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()
    

if __name__ == "__main__":
     main()