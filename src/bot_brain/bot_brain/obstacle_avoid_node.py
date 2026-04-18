#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class obstacleavoid(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        self.subscription=self.create_subscription(LaserScan,'scan',self.lidar_callback,10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.safe_dist=0.8
    
    def lidar_callback(self,msg):
        
        front_ranges = (msg.ranges[135:225])
        new_f_r=[]
        for i in front_ranges:
            if(i>0.0 and math.isfinite(i)):
                new_f_r.append(i)
        

        if len(new_f_r)==0:
            temp=10
            new_f_r.append(temp)

        
        min_dis=min(new_f_r)
        twist_msg=Twist()
        
        if(min_dis<=self.safe_dist):
            twist_msg.linear.x=0.0
            twist_msg.angular.z=0.25
        else:
            twist_msg.linear.x=0.2
            twist_msg.angular.z=0.0

        self.publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = obstacleavoid()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()


