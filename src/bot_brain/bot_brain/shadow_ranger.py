import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ==========================================
# 5x5 DEEP BRAIN ARCHITECTURE (PURE NUMPY)
# ==========================================
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def conv_forward(input_data, filters):
    b, h, w, _ = input_data.shape
    f_size = filters.shape[1]
    oh, ow = h - f_size + 1, w - f_size + 1
    windows = sliding_window_view(input_data, (f_size, f_size), axis=(1, 2)) 
    out = np.einsum('bxycij,kij->bxyk', windows, filters)
    return out

def maxpool_forward(x, pool_size=2):
    b, h, w, c = x.shape
    oh, ow = h // pool_size, w // pool_size
    reshaped = x.reshape(b, oh, pool_size, ow, pool_size, c)
    return reshaped.max(axis=(2, 4))

def predict(img, weights):
    # 1. Conv Pipeline
    x = conv_forward(img, weights['conv'])
    x = leaky_relu(x)
    x = maxpool_forward(x) 
    
    # 2. Dense Layers
    x = x.reshape(1, -1)
    x = leaky_relu(np.dot(x, weights['d1w']) + weights['d1b'])
    x = leaky_relu(np.dot(x, weights['d2w']) + weights['d2b'])
    
    # 3. Probabilities & Gap Analysis
    logits = np.dot(x, weights['ow']) + weights['ob']
    probs = softmax(logits)[0]
    
    sorted_indices = np.argsort(probs)[::-1]
    best_idx = sorted_indices[0]
    gap = probs[best_idx] - probs[sorted_indices[1]]
    
    return best_idx, probs[best_idx], gap

# ==========================================
# UPDATED VISION NODE
# ==========================================
class VisionController(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("Initializing Shadow Ranger (Ultra-Strict Logic)...")
        
        model_path = "/home/himanshu/ws_final/src/model_weights.npz"
        try:
            self.weights = np.load(model_path)
            self.get_logger().info("Deep Weights Loaded ✅")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            return

        self.class_map = {0: "LEFT", 1: "RIGHT", 2: "STOP", 3: "UTURN"}
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.sign_pub = self.create_publisher(String, 'detected_sign', 10)
        
        self.buffer_size = 15 # Increased for smoother detection
        self.vote_threshold = 9 # Requires 60% agreement
        self.prediction_buffer = deque(maxlen=self.buffer_size)
        self.current_state = None

    def resize_numpy(self, img, nh, nw):
        h, w = img.shape
        row_indices = (np.arange(nh) * (h / nh)).astype(int)
        col_indices = (np.arange(nw) * (w / nw)).astype(int)
        return img[np.ix_(row_indices, col_indices)]

    def camera_callback(self, msg):
        try:
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            channels = msg.step // msg.width
            img_array = raw_data.reshape((msg.height, msg.width, channels))
            
            if msg.encoding in ['bgr8', 'bgra8']:
                gray = 0.299 * img_array[:,:,2] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,0]
            else:
                gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        except Exception: return

        # 1. CENTER-FOCUS ROI
        # By looking at 15% to 45%, we stay above the floor but catch the signs earlier
        h, w = gray.shape
        roi = gray[int(h*0.15):int(h*0.45), int(w*0.25):int(w*0.75)]
        
        # 2. ACTIVITY GATE (Based on your S=70 logs)
        activity_score = np.std(roi)
        if activity_score < 60.0: 
            self.prediction_buffer.append(-1)
            self.process_voting() 
            return

        # 3. Predict
        img_32 = self.resize_numpy(roi, 32, 32)
        img_norm = (img_32 - np.mean(img_32)) / (np.std(img_32) + 1e-7)
        img_input = img_norm.reshape(1, 32, 32, 1)

        idx, conf, gap = predict(img_input, self.weights)
        
        # --- THE BIAS NERF ---
        # If the model thinks it's a STOP (index 2), we make the gate EXTREMELY high.
        # If it's a LEFT/RIGHT (index 0 or 1), we keep the gate accessible.
        
        if idx == 2: # STOP Class
            # The model must be nearly 95% sure to call a STOP
            is_valid = (conf > 0.95 and gap > 0.45)
        else: # LEFT, RIGHT, UTURN
            # Standard gates for the round signs
            is_valid = (conf > 0.70 and gap > 0.20)

        if is_valid: 
            self.prediction_buffer.append(idx)
        else:
            self.prediction_buffer.append(-1)

        self.process_voting()

    def process_voting(self):
        if len(self.prediction_buffer) == self.buffer_size:
            counts = Counter(self.prediction_buffer)
            most_common_pred, most_common_count = counts.most_common(1)[0]

            if most_common_count >= self.vote_threshold and most_common_pred != -1:
                detected_sign = self.class_map[most_common_pred]
                if self.current_state != detected_sign:
                    self.get_logger().info(f"🟢 SIGN LOCKED: {detected_sign}")
                    self.current_state = detected_sign
                
                msg_out = String(); msg_out.data = detected_sign
                self.sign_pub.publish(msg_out)

            elif most_common_pred == -1 and most_common_count >= self.vote_threshold:
                if self.current_state is not None:
                    self.get_logger().info("⚪ Path Clear")
                    self.current_state = None
            else:
                if self.current_state:
                    msg_out = String(); msg_out.data = self.current_state
                    self.sign_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = VisionController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()