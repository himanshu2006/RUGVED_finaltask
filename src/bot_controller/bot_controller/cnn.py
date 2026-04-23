import os
import cv2
import numpy as np

class_map = {
    "left": 0,
    "right": 1,
    "stop": 2,
    "uturn": 3
}

def load_dataset(split_path):
    img_size=64
    images = []
    labels = []

    for class_name in ["left", "right", "stop", "uturn"]:
        class_path = os.path.join(split_path, class_name)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img - np.mean(img)) / (np.std(img) + 1e-7) #Huge diff in real camera input   
            images.append(img)
            labels.append(class_map[class_name])
            #print(img_path, len(images))

    return np.array(images), np.array(labels)

def conv_layer(images, kernels):
    # images: (B, H, W)
    B, H, W = images.shape
    K, kh, kw = kernels.shape

    out_h = H - kh + 1
    out_w = W - kw + 1

    output = np.zeros((B, K, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = images[:, i:i+kh, j:j+kw][:, None, :, :]   # (B,1,kh,kw)
            output[:, :, i, j] = np.sum(region * kernels[None, :, :, :], axis=(2,3))

    return output

def relu(x):
    return np.maximum(0, x)

def dense(x, weights, bias):
    return np.dot(x, weights) + bias

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(image, kernels, w1, b1):
    # image: (64,64)

    image = image[np.newaxis, ...]   # (1,64,64)

    x = conv_layer(image, kernels)
    x = relu(x)

    x_flat = x.reshape(1, -1)

    pred = dense(x_flat, w1, b1)
    pred = softmax(pred)

    return np.argmax(pred), pred

def backward_dense(x, pred, labels, w1):
    B = len(labels)

    grad = pred.copy()
    grad[np.arange(B), labels] -= 1
    grad /= B

    dw = np.dot(x.T, grad)   # (features, classes)
    db = np.sum(grad, axis=0)

    dx = np.dot(grad, w1.T)

    return dw, db, dx

def train(images, labels, fixed_kernels, learnable_kernels, w1, b1, lr=0.001, epochs=55, batch_size=32):
    n = len(images)

    for epoch in range(epochs):

        indices = np.random.permutation(n)
        images = images[indices]
        labels = labels[indices]

        total_loss = 0

        for i in range(0, n, batch_size):

            batch_images = images[i:i+batch_size]   # (B, 64, 64)
            batch_labels = labels[i:i+batch_size]

            kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)

            # --- FORWARD ---
            x = conv_layer(batch_images, kernels)   # (B, K, H, W)
            x = relu(x)
            x_before_pool = x.copy()   # 🔥 STORE THIS                 # (B, K, H/2, W/2)

            B = x.shape[0]
            x_flat = x.reshape(B, -1)

            conv_shape = x.shape

            pred = dense(x_flat, w1, b1)
            pred = softmax(pred)

            loss = -np.mean(np.log(pred[np.arange(B), batch_labels] + 1e-9))
            total_loss += loss * B

            dw, db, dx = backward_dense(x_flat, pred, batch_labels, w1)

            dx = dx.reshape(conv_shape)

            # ReLU backward
            dx[x_before_pool <= 0] = 0
            d_kernels = np.zeros_like(learnable_kernels)

            B, K, H, W = dx.shape
            _, kh, kw = learnable_kernels.shape

            for i_h in range(H):
                for j_w in range(W):
                    region = batch_images[:, i_h:i_h+kh, j_w:j_w+kw][:, None, :, :]
                    grad_slice = dx[:, len(fixed_kernels):, i_h, j_w][:, :, None, None]
                    d_kernels += np.sum(region * grad_slice, axis=0)

            # update ONLY learnable kernels
            learnable_kernels -= lr * d_kernels / B

            w1 -= lr * dw
            b1 -= lr * db

        print(f"Epoch {epoch+1}, Loss: {total_loss/n:.4f}")

    return w1, b1

fixed_kernels = np.array([

    # U-turn hook (critical)
    [[ 0,  1,  0],
    [ 1,  0, -1],
    [ 0, -1,  0]],

    # 1. Sobel X (direction)
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]

    # 2. Sobel Y
    [[-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]]

    # 3. One curve kernel
    [[ 0,  1,  1],
    [-1,  0,  1],
    [-1, -1,  0]]

    # 4. Its mirror
    [[ 1,  1,  0],
    [ 1,  0, -1],
    [ 0, -1, -1]],

    # high-frequency detector (text-like)
    [[ 1, -1,  1],
    [-1,  1, -1],
    [ 1, -1,  1]]
], dtype=np.float32)

learnable_kernels = np.random.randn(16, 3, 3) * 0.1
sample = np.zeros((1, 64, 64))

kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)

x = conv_layer(sample, kernels)
x = relu(x)

flatten_size = x.flatten().shape[0]
w1 = np.random.randn(flatten_size, 4) * 0.1
b1 = np.zeros(4)
images, labels = load_dataset("/home/kanot/Downloads/master_dataset/data/train")
print("Images Loaded")
w1, b1 = train(images, labels, fixed_kernels, learnable_kernels, w1, b1)

np.savez("model_weights.npz",
         fixed_kernels=fixed_kernels,
         learnable_kernels=learnable_kernels,
         w1=w1,
         b1=b1)

print("Model saved ✅")

