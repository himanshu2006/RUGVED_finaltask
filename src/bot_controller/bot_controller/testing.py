import os
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# CONFIGURATION
# ==========================================
class_map = {
    "left": 0,
    "right": 1,
    "stop": 2,
    "uturn": 3  
}

labels = ["left", "right", "stop", "uturn"]

def load_model(path="model_weights.npz"):
    data = np.load(path)
    fixed_kernels = data["fixed_kernels"]
    learnable_kernels = data["learnable_kernels"]
    kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)
    return kernels, data["conv_bias"], data["w1"], data["b1"]

# ==========================================
# VECTORIZED LAYERS (MATCHING TRAINING)
# ==========================================

def conv_layer_forward(images, kernels, biases):
    kh, kw = kernels.shape[1], kernels.shape[2]
    windows = sliding_window_view(images, (kh, kw), axis=(1, 2))
    output = np.einsum('bxyhw,khw->bkxy', windows, kernels)
    output += biases[None, :, None, None]
    return output

def maxpool2d_forward(x):
    B, K, H, W = x.shape
    x_reshaped = x.reshape(B, K, H//2, 2, W//2, 2)
    out = x_reshaped.max(axis=(3, 5))
    return out

def relu(x):
    return np.maximum(0, x)

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(image, kernels, conv_bias, w1, b1):
    # image: (64,64)
    image = image[np.newaxis, ...]   # (1,64,64)

    # 1. Convolution (Fast Vectorized)
    x = conv_layer_forward(image, kernels, conv_bias)
    
    # 2. Activation
    x = relu(x)
    
    # 3. Max Pooling (Reduces from 62x62 to 31x31)
    x = maxpool2d_forward(x)
    
    # 4. Flatten & Dense
    x_flat = x.reshape(1, -1)
    pred = dense(x_flat, w1, b1)
    pred = softmax(pred)

    return np.argmax(pred), pred

if __name__ == "__main__":
    # Load model
    print("Loading model weights...")
    kernels, conv_bias, w1, b1 = load_model()

    # PATHS
    test_dir = "/home/himanshu/ws_final/src/bot_controller/master_dataset/data/test"

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        exit()

    # Initialize tracking metrics
    total_images = 0
    total_correct = 0
    class_metrics = {class_name: {"correct": 0, "total": 0} for class_name in class_map.keys()}
    confusion_matrix = np.zeros((4, 4), dtype=int)

    print(f"\nScanning test directory: {test_dir}")
    print("-" * 40)

    # Iterate through each class folder
    for class_name, true_label_idx in class_map.items():
        class_path = os.path.join(test_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Folder for class '{class_name}' not found. Skipping.")
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            original_img = cv2.imread(img_path)
            
            if original_img is None: 
                continue
            
            # --- STRICT PREPROCESSING MATCHING TRAINING ---
            img_resized = cv2.resize(original_img, (64, 64))
            
            # Extract ONLY the Red channel (Index 2 in BGR)
            img_red = img_resized[:, :, 2]
            
            # The exact array fed to the network
            img_normalized = (img_red - np.mean(img_red)) / (np.std(img_red) + 1e-7)

            # Predict
            pred_class, probs = predict(img_normalized, kernels, conv_bias, w1, b1)
            
            # Update confusion matrix
            confusion_matrix[true_label_idx, pred_class] += 1

            # Record metrics
            total_images += 1
            class_metrics[class_name]["total"] += 1
            
            if pred_class == true_label_idx:
                total_correct += 1
                class_metrics[class_name]["correct"] += 1

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    print("\n" + "=" * 40)
    print("🚦 TESTING EVALUATION RESULTS 🚦")
    print("=" * 40)

    if total_images == 0:
        print("No valid images found in the test directories.")
    else:
        overall_acc = (total_correct / total_images) * 100
        print(f"OVERALL ACCURACY: {overall_acc:.2f}% ({total_correct}/{total_images} images)")
        print("-" * 40)
        
        # Display breakdown per class
        for class_name in class_map.keys():
            metrics = class_metrics[class_name]
            total = metrics["total"]
            correct = metrics["correct"]
            
            if total > 0:
                acc = (correct / total) * 100
                print(f"Class '{class_name.upper():<5}': {acc:>6.2f}%  ({correct}/{total})")
            else:
                print(f"Class '{class_name.upper():<5}':    N/A  (0 images found)")
        print("=" * 40)

    