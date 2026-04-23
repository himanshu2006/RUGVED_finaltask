import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque, Counter
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class_map = {0: "LEFT", 1: "RIGHT", 2: "STOP", 3: "UTURN"}

def load_model(path):
    data = np.load(path)
    fixed_kernels = data["fixed_kernels"]
    learnable_kernels = data["learnable_kernels"]
    kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)
    return kernels, data["conv_bias"], data["w1"], data["b1"]


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
    image = image[np.newaxis, ...]   # Add batch dimension: (1,64,64)
    x = conv_layer_forward(image, kernels, conv_bias)
    x = relu(x)
    x = maxpool2d_forward(x)
    x_flat = x.reshape(1, -1)
    pred = dense(x_flat, w1, b1)
    pred = softmax(pred)
    return np.argmax(pred), np.max(pred)


class VisionController(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.get_logger().info("Initializing CNN Brain...")
        
        model_path = "/home/himanshu/ws_final/src/model_weights.npz"
          
        self.kernels, self.conv_bias, self.w1, self.b1 = load_model(model_path)
        self.get_logger().info("Model weights loaded successfully ✅")
    
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        
        # --- TEMPORAL VOTING SETUP ---
        self.buffer_size = 5
        self.vote_threshold = 4  # Needs 4 out of 5 frames to agree
        self.prediction_buffer = deque(maxlen=self.buffer_size)
        
        # State tracker to prevent terminal spam
        self.current_state = None

    def camera_callback(self, msg):
        # 1. Convert raw ROS Image bytes to NumPy Array
        try:
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            channels = msg.step // msg.width
            img_array = raw_data.reshape((msg.height, msg.width, channels))
        except Exception as e:
            self.get_logger().error(f"Image Parse Error: {e}")
            return

        
        # Grab the correct red index based on ROS encoding
        if msg.encoding in ['bgr8', 'bgra8']:
            red_index = 2
        elif msg.encoding in ['rgb8', 'rgba8']:
            red_index = 0
        else:
            red_index = 0 # Fallback
            
        img_red_full = img_array[:, :, red_index]

        # NumPy Nearest-Neighbor Resize to 64x64
        old_h, old_w = img_red_full.shape
        row_indices = (np.arange(64) * (old_h / 64)).astype(int)
        col_indices = (np.arange(64) * (old_w / 64)).astype(int)
        
        img_red = img_red_full[np.ix_(row_indices, col_indices)]
        
        # Normalize
        img_normalized = (img_red - np.mean(img_red)) / (np.std(img_red) + 1e-7)

        # 3. Predict
        pred_idx, confidence = predict(img_normalized, self.kernels, self.conv_bias, self.w1, self.b1)
        
        # 4. Temporal Voting Logic
        
        if confidence > 0.60:
            self.prediction_buffer.append(pred_idx)
        else:
            self.prediction_buffer.append(-1) # -1 represents "unsure/no sign"

        # Check if we have enough frames to vote
        if len(self.prediction_buffer) == self.buffer_size:
            # Count the occurrences of each prediction
            counts = Counter(self.prediction_buffer)
            most_common_pred, most_common_count = counts.most_common(1)[0]

            # If the most common prediction meets our threshold, execute it
            if most_common_count >= self.vote_threshold and most_common_pred != -1:
                detected_sign = class_map[most_common_pred]
                
                # Only print/trigger if it's a NEW detection to avoid terminal spam
                if self.current_state != detected_sign:
                    self.current_state = detected_sign
                    
            else:
                if self.current_state is not None:
                    self.get_logger().info("⚪ Sign lost or unsure. Resuming normal navigation.")
                    self.current_state = None

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionController()
    
    rclpy.spin(vision_node)
        
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()