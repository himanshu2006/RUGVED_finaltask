import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ==========================================
# NEW ARCHITECTURE FORWARD PASS (PURE NUMPY)
# ==========================================
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def conv_forward(input_data, filters):
    # input_data: (batch, h, w, 1), filters: (32, 5, 5)
    b, h, w, _ = input_data.shape
    f_size = filters.shape[1]
    oh, ow = h - f_size + 1, w - f_size + 1
    
    # Use sliding window for speed
    windows = sliding_window_view(input_data, (f_size, f_size), axis=(1, 2)) # (b, oh, ow, 1, 5, 5)
    # Einstein summation to mimic the training script's region * weight logic
    # b:batch, x:oh, y:ow, c:channel(1), i:f_h, j:f_w, k:filters
    out = np.einsum('bxycij,kij->bxyk', windows, filters)
    return out

def maxpool_forward(x, pool_size=2):
    b, h, w, c = x.shape
    oh, ow = h // pool_size, w // pool_size
    reshaped = x.reshape(b, oh, pool_size, ow, pool_size, c)
    return reshaped.max(axis=(2, 4))

def predict(img, weights):
    # 1. Conv + LeakyReLU + MaxPool
    # Input img must be (1, 32, 32, 1)
    x = conv_forward(img, weights['conv'])
    x = leaky_relu(x)
    x = maxpool_forward(x) # Becomes (1, 14, 14, 32)
    
    # 2. Flatten
    x = x.reshape(1, -1)
    
    # 3. Dense 1 + LeakyReLU
    x = np.dot(x, weights['d1w']) + weights['d1b']
    x = leaky_relu(x)
    
    # 4. Dense 2 + LeakyReLU
    x = np.dot(x, weights['d2w']) + weights['d2b']
    x = leaky_relu(x)
    
    # 5. Output Layer + Softmax
    logits = np.dot(x, weights['ow']) + weights['ob']
    probs = softmax(logits)
    
    return np.argmax(probs), np.max(probs)

# ==========================================
# VISION NODE
# ==========================================
class VisionController(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.get_logger().info("Initializing Shadow Ranger (New 5x5 Deep Brain)...")
        
        # Load the new weight keys
        model_path = "/home/himanshu/ws_final/src/model_weights.npz"
        try:
            self.weights = np.load(model_path)
            self.get_logger().info("Deep Weights Loaded Successfully ✅")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            return

        self.class_map = {0: "LEFT", 1: "RIGHT", 2: "STOP", 3: "UTURN"}
        
        # ROS Setup
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.sign_pub = self.create_publisher(String, 'detected_sign', 10)
        
        # --- STABLE VOTING (10 frames / 60% threshold) ---
        self.buffer_size = 10
        self.vote_threshold = 6 
        self.prediction_buffer = deque(maxlen=self.buffer_size)
        self.current_state = None

    def camera_callback(self, msg):
        try:
            # 1. Convert to Grayscale NumPy (matches training cv2.IMREAD_GRAYSCALE)
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            channels = msg.step // msg.width
            img_array = raw_data.reshape((msg.height, msg.width, channels))
            
            # Convert BGR/RGB to Grayscale mathematically
            if msg.encoding in ['bgr8', 'bgra8']:
                gray = 0.299 * img_array[:,:,2] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,0]
            else:
                gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        except Exception:
            return

        # 2. Aggressive ROI (Top-Middle focus)
        h, w = gray.shape
        roi = gray[int(h*0.15):int(h*0.55), int(w*0.1):int(w*0.9)]

        # 3. Resize to 32x32 (matches new CNN input)
        rh, rw = roi.shape
        row_indices = (np.arange(32) * (rh / 32)).astype(int)
        col_indices = (np.arange(32) * (rw / 32)).astype(int)
        img_32 = roi[np.ix_(row_indices, col_indices)]
        
        # 4. Standardize (matches training script)
        img_norm = (img_32 - np.mean(img_32)) / (np.std(img_32) + 1e-7)
        img_input = img_norm.reshape(1, 32, 32, 1)

        # 5. Predict
        idx, conf = predict(img_input, self.weights)
        
        # Debugging
        # self.get_logger().info(f"Guess: {self.class_map[idx]} | Conf: {conf:.2f}")

        # 6. Logic Gate (Only vote if reasonably sure)
        if conf > 0.65: # High gate to avoid floor noise
            self.prediction_buffer.append(idx)
        else:
            self.prediction_buffer.append(-1)

        # 7. Stable Voting
        if len(self.prediction_buffer) == self.buffer_size:
            counts = Counter(self.prediction_buffer)
            most_common_pred, most_common_count = counts.most_common(1)[0]

            if most_common_count >= self.vote_threshold and most_common_pred != -1:
                detected_sign = self.class_map[most_common_pred]
                if self.current_state != detected_sign:
                    self.get_logger().info(f"🟢 LOCKED: {detected_sign}")
                    self.current_state = detected_sign
                
                # Send to Obstacle Node
                msg_out = String()
                msg_out.data = detected_sign
                self.sign_pub.publish(msg_out)

            elif most_common_pred == -1 and most_common_count >= self.vote_threshold:
                if self.current_state is not None:
                    self.get_logger().info("⚪ Path Clear")
                    self.current_state = None
            else:
                # Stubbornly keep the previous state if consensus is low
                if self.current_state:
                    msg_out = String()
                    msg_out.data = self.current_state
                    self.sign_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = VisionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()