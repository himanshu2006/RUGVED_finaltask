#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import time

class obstacleavoid(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        self.subscription=self.create_subscription(LaserScan,'scan',self.lidar_callback,10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.safe_dist=1.2
        self.kp_a=0.5
        self.ki_a=0.001
        self.kd_a=0.07
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = self.get_clock().now()

    def turn_new(self,l,r):
        
        error=l-r
        if(l==r):
            error=0.1
        now=self.get_clock().now()
        diff=now-self.last_time
        dt=diff.nanoseconds/1e9
        if dt<=0.0:
            return 0.0

        prop=self.kp_a*error
        self.integral+=error*dt
        inte=self.ki_a*self.integral
        deriv=self.kd_a* (error-self.prev_error)/dt

        self.prev_error = error
        self.last_time = now
        hh=prop + inte +deriv

        return hh
    
    def speed_new(self,current):
        max_w=0.5
        drop_ratio = 0.4
        turn_percentage = min(abs(current) / max_w, 1.0)
        self.new_speed = 0.4* (1 - (drop_ratio * turn_percentage))

        return max(self.new_speed, 0.1)

    def lidar_callback(self,msg):
        
        front_ranges = (msg.ranges[150:210])
        new_f=[]
        new_r=[]
        new_l=[]
        left_ranges=msg.ranges[210:270]
        right_ranges=msg.ranges[90:150]
        for i in front_ranges:
            if(i>0.0 and math.isfinite(i)):
                new_f.append(i)
        
        for i in left_ranges:
            if(i>0.0 and math.isfinite(i)):
                new_l.append(i)

        for i in right_ranges:
            if(i>0.0 and math.isfinite(i)):
                new_r.append(i)


        if len(new_f)==0:
            temp=10
            new_f.append(temp)
        if len(new_l)==0:
            temp=10
            new_l.append(temp)
        if len(new_r)==0:
            temp=10
            new_r.append(temp)
        
        min_dis=min(new_f)
        twist_msg=Twist()
        
        if(min_dis<=self.safe_dist):
            angle=self.turn_new(min(new_l),min(new_r))
            speed=self.speed_new(angle)
            twist_msg.linear.x=speed
            twist_msg.angular.z=max(min(angle,0.5),-0.5)
        else:
            twist_msg.linear.x=0.4
            twist_msg.angular.z=0.0
            self.integral = 0.0
            self.prev_error = 0.0
            self.last_time = self.get_clock().now()

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


