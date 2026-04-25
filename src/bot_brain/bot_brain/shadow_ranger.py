#!/usr/bin/env python3

import os
import cv2
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import Odometry

def relu(x): return np.maximum(0, x)

def fast_conv(x, k, b):
    windows = sliding_window_view(x, (k.shape[2], k.shape[3]), axis=(2, 3))
    return np.einsum('bchwuv,kcuv->bkhw', windows, k) + b.reshape(1, -1, 1, 1)

def fast_maxpool(x, size=2):
    B, K, H, W = x.shape
    oH, oW = H // size, W // size
    return x[:, :, :oH*size, :oW*size].reshape(B, K, oH, size, oW, size).max(axis=(3, 5))

def predict(img, params):
    z1 = fast_maxpool(relu(fast_conv(img, params['k1'], params['b1_c'])))
    z2 = fast_maxpool(relu(fast_conv(z1, params['k2'], params['b2_c'])))
    
    z3 = relu(np.dot(z2.reshape(1, -1), params['w1']) + params['b1'])
    logits = np.dot(z3, params['w2']) + params['b2']
    
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = (e / np.sum(e, axis=1, keepdims=True))[0]
    
    best_idx = np.argmax(probs)
    gap = probs[best_idx] - np.partition(probs, -2)[-2] # Difference between top 2
    return best_idx, probs[best_idx], gap

class VisionController(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("Initializing Pose-Driven Vision Node...")
        
        try:
            self.weights = dict(np.load("/home/kanot/RUGVED_finaltask/src/bot_brain/bot_brain/model.npz"))
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}"); return

        self.class_map = {0: "LEFT", 1: "RIGHT", 2: "STOP", 3: "UTURN"}
        
        # ROS Communications
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sign_pub = self.create_publisher(String, 'detected_sign', 10)
        
        # --- STRICT VOTING CONFIG ---
        self.buffer_size = 30       # Collect 30 frames
        self.vote_threshold = 24    # Require 80% consensus to trigger a lock
        self.prediction_buffer = deque(maxlen=self.buffer_size)
        
        # --- STATE MACHINE & POSE ---
        self.state = "SEARCHING"    # SEARCHING -> TRACKING_DISTANCE -> COOLDOWN
        self.locked_sign = None
        self.current_pose = (0.0, 0.0)
        self.lock_pose = (0.0, 0.0)
        
        # TUNABLE: Meters to drive forward after locking the sign
        self.distance_to_turn = 2.0 
        self.cooldown_frames = 0    

    def odom_callback(self, msg):
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def extract_arrow_shape(self, img_gray):
        edges = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 0), 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: 
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50: 
            return None
            
        x, y, w, h = cv2.boundingRect(largest)
        return edges[y:y+h, x:x+w]

    def camera_callback(self, msg):
        # 1. Handle Cooldown (Ignore camera while turning)
        if self.state == "COOLDOWN":
            self.cooldown_frames -= 1
            if self.cooldown_frames <= 0:
                self.state = "SEARCHING"
                self.prediction_buffer.clear()
            return

        # 2. Track Pose Distance
        if self.state == "TRACKING_DISTANCE":
            # Calculate Euclidean distance from where we saw the sign to where we are now
            dist_traveled = math.hypot(self.current_pose[0] - self.lock_pose[0], 
                                       self.current_pose[1] - self.lock_pose[1])
            
            if dist_traveled >= self.distance_to_turn:
                self.get_logger().info(f"⚠️ REACHED TURN DISTANCE! Executing: {self.locked_sign}")
                msg_out = String(); msg_out.data = self.locked_sign
                self.sign_pub.publish(msg_out)
                
                self.state = "COOLDOWN"
                self.cooldown_frames = 90 # Pause vision for ~3 seconds while turning
            return

        # 3. Process Camera (SEARCHING State)
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            gray = 0.299*raw[:,:,2] + 0.587*raw[:,:,1] + 0.114*raw[:,:,0] if msg.encoding in ['bgr8','bgra8'] else \
                   0.299*raw[:,:,0] + 0.587*raw[:,:,1] + 0.114*raw[:,:,2]
        except Exception: 
            self.get_logger().error("Error processing camera image")
            return

        h, w = gray.shape
        roi = gray[int(h*0.15):int(h*0.45), int(w*0.25):int(w*0.75)].astype(np.uint8)
        shape_edges = self.extract_arrow_shape(roi)
        
        if shape_edges is None: 
            self.prediction_buffer.append(-1)
        else:
            img_input = (cv2.resize(shape_edges, (64, 64)).astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
            idx, conf, gap = predict(img_input, self.weights)
            
            # --- NEW: SAVE THE PROCESSED IMAGE TO DISK ---
            self.save_counter += 1
            if self.save_counter % 5 == 0:  # Save 1 out of every 5 frames to prevent lag
                # Format: "0015_saw_LEFT_conf_0.85.png"
                img_name = f"{self.save_counter:04d}_saw_{self.class_map[idx]}_conf_{conf:.2f}.png"
                img_path = os.path.join(self.debug_dir, img_name)
                # Multiply img_input back to 255 so OpenCV can save it as a visible image
                img_to_save = (img_input[0, 0] * 255.0).astype(np.uint8)
                cv2.imwrite(img_path, img_to_save)
            # ----------------------------------------------
            
            is_valid = (conf > 0.95 and gap > 0.45) if idx == 2 else (conf > 0.70 and gap > 0.20)
            self.prediction_buffer.append(idx if is_valid else -1)
        # 4. Voting Logic
        if len(self.prediction_buffer) == self.buffer_size:
            most_common_pred, count = Counter(self.prediction_buffer).most_common(1)[0]

            if count >= self.vote_threshold and most_common_pred != -1:
                self.locked_sign = self.class_map[most_common_pred]
                self.lock_pose = self.current_pose # Record exact physical location
                self.state = "TRACKING_DISTANCE"
                self.get_logger().info(f"🎯 LOCKED ON: {self.locked_sign}. Driving {self.distance_to_turn}m before turning...")

def main(args=None):
    rclpy.init(args=args)
    node = VisionController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
