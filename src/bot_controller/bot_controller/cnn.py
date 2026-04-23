import os
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class_map = {
    "left": 0,
    "right": 1,
    "stop": 2,
    "uturn": 3
}

def load_dataset(split_path):
    img_size = 64
    images = []
    labels = []

    for class_name in ["left", "right", "stop", "uturn"]:
        class_path = os.path.join(split_path, class_name)
        if not os.path.exists(class_path): continue
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # --- THE RED CHANNEL FIX ---
            img = cv2.resize(img, (img_size, img_size))
            
            # Extract ONLY the Red channel (Index 2 in BGR)
            # Red signs become bright white, green backgrounds become dark.
            img_red = img[:, :, 2] 
            
            # Normalize based on the red channel
            img_normalized = (img_red - np.mean(img_red)) / (np.std(img_red) + 1e-7)  
            
            images.append(img_normalized)
            labels.append(class_map[class_name])
            
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    return np.array(images), np.array(labels)

# ==========================================
# VECTORIZED LAYERS (NO LOOPS)
# ==========================================

def conv_layer_forward(images, kernels, biases):
    """
    Highly optimized Convolution using sliding windows and Einstein Summation.
    """
    kh, kw = kernels.shape[1], kernels.shape[2]
    
    # Extract all possible 3x3 windows from the image at once
    # Shape becomes: (B, out_h, out_w, kh, kw)
    windows = sliding_window_view(images, (kh, kw), axis=(1, 2))
    
    # Multiply and sum windows with kernels instantly
    # b=batch, x=out_h, y=out_w, k=filters, h=kernel_h, w=kernel_w
    output = np.einsum('bxyhw,khw->bkxy', windows, kernels)
    
    # Add bias to each filter channel
    output += biases[None, :, None, None]
    return output, windows

def maxpool2d_forward(x):
    """
    Clever reshape trick for pure NumPy 2x2 Max Pooling without loops.
    Assumes dimensions are divisible by 2.
    """
    B, K, H, W = x.shape
    # Reshape to isolate 2x2 blocks, then take the max over those blocks
    x_reshaped = x.reshape(B, K, H//2, 2, W//2, 2)
    out = x_reshaped.max(axis=(3, 5))
    return out

def maxpool2d_backward(dx_pool, x_pre_pool, x_post_pool):
    """
    Routes the gradient only to the pixel that had the maximum value.
    """
    # Scale up the pooled output and gradients back to the original size
    x_post_up = x_post_pool.repeat(2, axis=2).repeat(2, axis=3)
    dx_pool_up = dx_pool.repeat(2, axis=2).repeat(2, axis=3)
    
    # Create a binary mask of where the maximums were
    mask = (x_pre_pool == x_post_up)
    
    # Apply mask to gradients
    return mask * dx_pool_up

def relu(x):
    return np.maximum(0, x)

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def softmax(x):
    # Stabilized Softmax
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def backward_dense(x, pred, labels, w1):
    B = len(labels)
    grad = pred.copy()
    grad[np.arange(B), labels] -= 1
    grad /= B

    dw = np.dot(x.T, grad)   
    db = np.sum(grad, axis=0)
    dx = np.dot(grad, w1.T)

    return dw, db, dx

# ==========================================
# TRAINING LOOP
# ==========================================

def train(images, labels, fixed_kernels, learnable_kernels, conv_bias, w1, b1, lr=0.01, epochs=40, batch_size=32):
    n = len(images)
    
    # Momentum variables
    v_w1, v_b1 = np.zeros_like(w1), np.zeros_like(b1)
    v_k, v_cb = np.zeros_like(learnable_kernels), np.zeros_like(conv_bias)
    momentum = 0.9

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        images = images[indices]
        labels = labels[indices]
        
        total_loss, correct = 0, 0

        for i in range(0, n, batch_size):
            batch_images = images[i:i+batch_size]   
            batch_labels = labels[i:i+batch_size]
            B_curr = len(batch_images)

            kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)

            # --- FORWARD PASS ---
            # 1. Convolution (64x64 -> 62x62)
            x_conv, windows = conv_layer_forward(batch_images, kernels, conv_bias)
            
            # 2. Activation
            x_relu = relu(x_conv)
            
            # 3. Max Pooling (62x62 -> 31x31)
            x_pool = maxpool2d_forward(x_relu)

            # 4. Flatten & Dense
            x_flat = x_pool.reshape(B_curr, -1)
            pred = dense(x_flat, w1, b1)
            pred = softmax(pred)

            # --- METRICS ---
            loss = -np.mean(np.log(pred[np.arange(B_curr), batch_labels] + 1e-9))
            total_loss += loss * B_curr
            correct += np.sum(np.argmax(pred, axis=1) == batch_labels)

            # --- BACKWARD PASS ---
            # 1. Dense Gradients
            dw, db, dx_flat = backward_dense(x_flat, pred, batch_labels, w1)
            
            # 2. Reshape back to pooled dimensions
            dx_pool = dx_flat.reshape(x_pool.shape)
            
            # 3. Max Pooling Backward (31x31 -> 62x62)
            dx_relu = maxpool2d_backward(dx_pool, x_relu, x_pool)
            
            # 4. ReLU Backward
            dx_conv = dx_relu.copy()
            dx_conv[x_conv <= 0] = 0

            # 5. Convolution Backward (Vectorized)
            # Isolate gradients for LEARNABLE kernels only
            dx_learnable = dx_conv[:, len(fixed_kernels):, :, :]
            
            # einsum magic to calculate kernel gradients instantly
            d_kernels = np.einsum('bxyhw,bkxy->khw', windows, dx_learnable)
            d_conv_bias = np.sum(dx_conv, axis=(0, 2, 3))

            # --- UPDATE WEIGHTS (with Momentum) ---
            v_w1 = momentum * v_w1 - lr * dw
            w1 += v_w1
            
            v_b1 = momentum * v_b1 - lr * db
            b1 += v_b1
            
            v_k = momentum * v_k - lr * d_kernels
            learnable_kernels += v_k
            
            v_cb = momentum * v_cb - lr * d_conv_bias
            conv_bias += v_cb

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/n:.4f} | Acc: {correct/n:.2%}")

    return learnable_kernels, conv_bias, w1, b1

# ==========================================
# INITIALIZATION & EXECUTION
# ==========================================

# Fixed Kernels Definition
fixed_kernels = np.array([
    [[ 0,  1,  0], [ 1,  0, -1], [ 0, -1,  0]], # U-turn
    [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], # Sobel X
    [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
    [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
    [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]], # Octagon detector for stop sign
    [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], # Sobel Y
    [[ 0,  1,  1], [-1,  0,  1], [-1, -1,  0]], # Curve 1
    [[ 1,  1,  0], [ 1,  0, -1], [ 0, -1, -1]], # Mirror Curve
    [[ 1, -1,  1], [-1,  1, -1], [ 1, -1,  1]]  # High-freq
], dtype=np.float32)

num_learnable = 32
total_kernels = len(fixed_kernels) + num_learnable

# HE INITIALIZATION: Critical for deep learning convergence
learnable_kernels = np.random.randn(num_learnable, 3, 3) * np.sqrt(2.0 / (3*3))
conv_bias = np.zeros(total_kernels)

# Determine Flatten Size automatically
sample_output_h = (64 - 3 + 1) // 2 # 64 -> 62 -> 31
sample_output_w = (64 - 3 + 1) // 2
flatten_size = total_kernels * sample_output_h * sample_output_w

# HE INITIALIZATION for Dense Layer
w1 = np.random.randn(flatten_size, 4) * np.sqrt(2.0 / flatten_size)
b1 = np.zeros(4)

print("Loading Images...")
images, labels = load_dataset("/home/himanshu/ws_final/src/bot_controller/master_dataset/data/train")

if len(images) > 0:
    print(f"Loaded {len(images)} images. Starting Training...")
    learnable_kernels, conv_bias, w1, b1 = train(images, labels, fixed_kernels, learnable_kernels, conv_bias, w1, b1)

    np.savez("model_weights.npz",
             fixed_kernels=fixed_kernels,
             learnable_kernels=learnable_kernels,
             conv_bias=conv_bias,
             w1=w1,
             b1=b1)
    print("Model saved ✅")
else:
    print("No images found. Check your dataset path.")