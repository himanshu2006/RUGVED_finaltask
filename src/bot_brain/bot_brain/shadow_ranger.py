import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque, Counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String # Added for node-to-node talk

# ==========================================
# CONFIGURATION
# ==========================================
class_map = {0: "LEFT", 1: "RIGHT", 2: "STOP", 3: "UTURN"}

def load_model(path):
    data = np.load(path)
    fixed_kernels = data["fixed_kernels"]
    learnable_kernels = data["learnable_kernels"]
    kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)
    return kernels, data["conv_bias"], data["w1"], data["b1"]

# ==========================================
# VECTORIZED CNN LAYERS (PURE NUMPY)
# ==========================================
def conv_layer_forward(images, kernels, biases):
    kh, kw = kernels.shape[1], kernels.shape[2]
    windows = sliding_window_view(images, (kh, kw), axis=(1, 2))
    output = np.einsum('bxyhw,khw->bkxy', windows, kernels)
    output += biases[None, :, None, None]
    return output

def maxpool2d_forward(x):
    B, K, H, W = x.shape
    out = x.reshape(B, K, H//2, 2, W//2, 2).max(axis=(3, 5))
    return out

def relu(x):
    return np.maximum(0, x)

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(image, kernels, conv_bias, w1, b1):
    image = image[np.newaxis, ...]   
    x = conv_layer_forward(image, kernels, conv_bias)
    x = relu(x)
    x = maxpool2d_forward(x)
    x_flat = x.reshape(1, -1)
    pred = dense(x_flat, w1, b1)
    pred = softmax(pred)
    return np.argmax(pred), np.max(pred)

# ==========================================
# UPDATED VISION NODE
# ==========================================
class VisionController(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.get_logger().info("Initializing Shadow Ranger (10-Frame Logic)...")
        
        # Adjust this path if needed
        model_path = "/home/himanshu/ws_final/model_weights.npz"
          
        try:
            self.kernels, self.conv_bias, self.w1, self.b1 = load_model(model_path)
            self.get_logger().info("CNN Weights Loaded ✅")
        except Exception as e:
            self.get_logger().error(f"Weights Load Failed: {e}")
            return
    
        # Subscriptions & Publishers
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.sign_pub = self.create_publisher(String, 'detected_sign', 10)
        
        # --- TUNED VOTING SETUP ---
        self.buffer_size = 10     # Now checks 10 frames
        self.vote_threshold = 6   # Requires 60% consensus (6 out of 10)
        self.prediction_buffer = deque(maxlen=self.buffer_size)
        
        self.current_state = None

    def camera_callback(self, msg):
        try:
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            channels = msg.step // msg.width
            img_array = raw_data.reshape((msg.height, msg.width, channels))
        except Exception as e:
            return

        # 1. ROI CROP: Focusing on the horizon/walls
        h, w, _ = img_array.shape
        start_row, end_row = int(h * 0.15), int(h * 0.55)
        start_col, end_col = int(w * 0.1), int(w * 0.9) 
        img_cropped = img_array[start_row:end_row, start_col:end_col]

        # 2. Pure NumPy Preprocessing (Red Channel Fix)
        if msg.encoding in ['bgr8', 'bgra8']:
            red_index = 2
        else:
            red_index = 0
            
        img_red_full = img_cropped[:, :, red_index]

        # 3. Pure NumPy Resize (64x64)
        old_h, old_w = img_red_full.shape
        row_indices = (np.arange(64) * (old_h / 64)).astype(int)
        col_indices = (np.arange(64) * (old_w / 64)).astype(int)
        img_red = img_red_full[np.ix_(row_indices, col_indices)]
        
        img_normalized = (img_red - np.mean(img_red)) / (np.std(img_red) + 1e-7)

        # 4. Predict
        pred_idx, confidence = predict(img_normalized, self.kernels, self.conv_bias, self.w1, self.b1)
        
        # Use 0.50 as a strict confidence gate for individual frames
        if confidence > 0.85: 
            self.prediction_buffer.append(pred_idx)
        else:
            self.prediction_buffer.append(-1)

        # 5. Stable Voting Logic
        if len(self.prediction_buffer) == self.buffer_size:
            counts = Counter(self.prediction_buffer)
            most_common_pred, most_common_count = counts.most_common(1)[0]

            # OPTION A: Consensus met for a sign
            if most_common_count >= self.vote_threshold and most_common_pred != -1:
                detected_sign = class_map[most_common_pred]
                
                if self.current_state != detected_sign:
                    self.get_logger().info(f"🟢 SIGN DETECTED: {detected_sign} ({most_common_count}/{self.buffer_size} votes)")
                    self.current_state = detected_sign
                
                # Continuously publish the detected sign
                sign_msg = String()
                sign_msg.data = detected_sign
                self.sign_pub.publish(sign_msg)
                    
            # OPTION B: Consensus met that NO sign is visible
            elif most_common_pred == -1 and most_common_count >= self.vote_threshold:
                if self.current_state is not None:
                    self.get_logger().info("⚪ Clear Path: Resuming Navigation.")
                    self.current_state = None
            
            # OPTION C: No clear consensus, keep the last known state
            else:
                if self.current_state is not None:
                    # Keep the brain "locked" on the sign to prevent flickering
                    sign_msg = String()
                    sign_msg.data = self.current_state
                    self.sign_pub.publish(sign_msg)

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionController()
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()