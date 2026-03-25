import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
import cv2
import math
class OrbitController(Node):
    def __init__(self):
        super().__init__('orbit_controller')
        self.closest_cluster = None
        self.objAngle = 0 
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Orbit Parameters
        self.target_distance = 0.5  # sabit uzaklık
        self.forward_speed = 0.2    # Constant forward speed in m/s
        self.orbit_direction = 1    # 1 for Counter-Clockwise (left side), -1 for Clockwise (right side)
        self.k_orbit_dist = 2.0     # 
        self.k_angular_p = 1.5      
        
        self.get_logger().info("Orbit Controller Node Started! Waiting for scan data...")

    def detectHoughLines(self,img):
        blankImage = np.zeros_like(img)

        if img is not None:

            lines = cv2.HoughLinesP(img,1,np.pi/180,20,minLineLength=5, maxLineGap=35)
            if lines is not None:

                for l in lines:
                    if l is not None:

                        x,y,x1,y1 = l[0]
                        cv2.line(blankImage,(x,y),(x1,y1),(255,255,255),thickness=5)
                        tan = (y1 - y) / (x1 - x)
                        degree = math.atan(tan)*180/np.pi
                        steering = degree
                        print(str("angle between AGV and Object:")+str(steering))
                        return steering
                    
            else:
                return None
        cv2.imshow("..",blankImage)

        
    def drawClosestObject(self):
        blank = np.zeros((480, 640), np.uint8)

        if self.closest_cluster is not None:

            scale = 100   # 1 meter = 100 pixels
            cx, cy = 320, 240  # image center

            for x, y in self.closest_cluster:

                px = int(x * scale) + cx
                py = cy - int(y * scale)   # <-- IMPORTANT (flip y)

                if 0 <= px < 640 and 0 <= py < 480:
                    blank[py, px] = 255


        cv2.imshow("closest", blank)
        self.objAngle = self.detectHoughLines(blank)

        cv2.waitKey(1)

        
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        
        # 1. Generate an array of angles corresponding to each range
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # 2. Filter out invalid ranges (inf, nan, or out of bounds)
        valid_max = 10.0 # Define a reasonable max range for your sensor
        valid_idx = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges <= valid_max)
        
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]

        if len(valid_ranges) == 0:
            return  # Exit if no valid data is seen

        # 3. poları kartezyene ceviriyoruz (x,y) = rcon(a),rsin(a)
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))

        # 4. Sequential Euclidean Clustering
        cluster_tolerance = 0.3  # Max distance between points to be in the same object (meters)
        min_cluster_size = 3     # Minimum points required to form a valid object

        diffs = np.linalg.norm(points[1:] - points[:-1], axis=1)
        split_indices = np.where(diffs > cluster_tolerance)[0] + 1
        clusters = np.split(points, split_indices)

        centroids = []
        valid_clusters = []
        
        for cluster in clusters:                                   
            if len(cluster) >= min_cluster_size:
                centroid = np.mean(cluster, axis=0) #merkez konum icin önemli
                centroids.append(centroid)
                valid_clusters.append(cluster)

        if not centroids:
            return  

        # 6. Find the closest object (centroid)
        centroids = np.array(centroids)
        distances_to_centroids = np.linalg.norm(centroids, axis=1)
        

        
        closest_idx = np.argmin(distances_to_centroids)
        closest_centroid = centroids[closest_idx]
        self.closest_cluster = valid_clusters[closest_idx]
        self.drawClosestObject()
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg() 
        cmd.header.frame_id = "base_link" 
        
        if(self.objAngle is not None):
            if(self.objAngle<5):

                cmd.twist.linear.x = 0.5
                cmd.twist.angular.z = 0.0
            

        else:


        
            # 7. Convert centroid back to polar for the control logic
            min_dist = distances_to_centroids[closest_idx]
            
            angle = np.arctan2(closest_centroid[1], closest_centroid[0])

            # ---------------------------------------------------------
            # 8. Execute Continuous Orbit Control Logic
            # ---------------------------------------------------------
            
            # Populate header (Required for TwistStamped, ignore if using standard Twist)
            
            # Calculate distance error
            error_dist = min_dist - self.target_distance

            # Calculate desired angle to the object (90 degrees / pi/2)
            base_angle = self.orbit_direction * (np.pi / 2.0)
            
            # Angle correction based on distance error
            angle_correction = np.arctan(self.k_orbit_dist * error_dist)
            
            # Adjust desired angle based on whether we are too close or too far
            desired_angle = base_angle - (self.orbit_direction * angle_correction)

            # Calculate the angular error 
            error_angle = angle - desired_angle
            
            # Normalize the error between -pi and pi to avoid wrap-around jumps
            error_angle = np.arctan2(np.sin(error_angle), np.cos(error_angle))

            # Constant linear velocity forces the robot to move
            cmd.twist.linear.x = self.forward_speed
            
            # Proportional control for steering to maintain the desired angle
            raw_angular_z = self.k_angular_p * error_angle
            
            # Clamp angular velocity between -1.0 and 1.0 rad/s for safety
            cmd.twist.angular.z = float(np.clip(raw_angular_z, -1.0, 1.0))

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = OrbitController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Orbit Controller...")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()