#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String  
import math
import time

class obstacleavoid(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_node')
        
        
        self.subscription = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        
        self.sign_sub = self.create_subscription(String, 'detected_sign', self.sign_callback, 10)
        
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.safe_dist = 1.4
        self.kp_a = 0.5
        self.ki_a = 0.001
        self.kd_a = 0.07
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = self.get_clock().now()
        

        self.pending_turn = None


    def sign_callback(self, msg):
        self.get_logger().info(f"Received Sign: {msg.data} - Saving to memory.")
        self.pending_turn = msg.data

    def turn_new(self, l, r):
        error = l - r
        if abs(error) < 0.15:
            error = 1.5
        now = self.get_clock().now()
        diff = now - self.last_time
        dt = diff.nanoseconds / 1e9
        if dt <= 0.0:
            return 0.0

        prop = self.kp_a * error
        self.integral += error * dt
        inte = self.ki_a * self.integral
        deriv = self.kd_a * (error - self.prev_error) / dt

        self.prev_error = error
        self.last_time = now
        hh = prop + inte + deriv

        return hh
    
    def speed_new(self, current):
        max_w = 0.5
        drop_ratio = 0.4
        turn_percentage = min(abs(current) / max_w, 1.0)
        self.new_speed = 0.4 * (1 - (drop_ratio * turn_percentage))
        return max(self.new_speed, 0.1)


    def right_turn(self):
        self.get_logger().info("Executing RIGHT turn from memory!")
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = -0.5 
        self.publisher.publish(msg)
        time.sleep(3.0) 
        
    def left_turn(self):
        self.get_logger().info("Executing LEFT turn from memory!")
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.7
        self.publisher.publish(msg)
        time.sleep(3.0)

    def stop_robot(self):
        self.get_logger().info("Stopping at STOP sign!")
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)
        time.sleep(5.0) 

    def lidar_callback(self, msg):
        front_ranges = (msg.ranges[150:210])
        new_f = []
        new_r = []
        new_l = []
        left_ranges = msg.ranges[210:270]
        right_ranges = msg.ranges[90:150]
        
        for i in front_ranges:
            if(i > 0.0 and math.isfinite(i)):
                new_f.append(i)
        
        for i in left_ranges:
            if(i > 0.0 and math.isfinite(i)):
                new_l.append(i)

        for i in right_ranges:
            if(i > 0.0 and math.isfinite(i)):
                new_r.append(i)

        if len(new_f) == 0:
            new_f.append(10.0)
        if len(new_l) == 0:
            new_l.append(10.0)
        if len(new_r) == 0:
            new_r.append(10.0)
        
        min_dis = min(new_f)
        twist_msg = Twist()
        
    
        if(min_dis <= self.safe_dist):
            angle = self.turn_new(min(new_l), min(new_r))
            speed = self.speed_new(angle)
            twist_msg.linear.x = speed
            twist_msg.angular.z = max(min(angle, 0.5), -0.5)
            self.publisher.publish(twist_msg)
        
        
        elif self.pending_turn is not None:
            if self.pending_turn == "RIGHT":
                self.right_turn()
            elif self.pending_turn == "LEFT":
                self.left_turn()
            elif self.pending_turn == "STOP":
                self.stop_robot()
                
            self.pending_turn = None 
            return
            
        else:
            twist_msg.linear.x = 0.4
            twist_msg.angular.z = 0.0
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