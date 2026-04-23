import os
import cv2
import numpy as np

class_map = {
    "left": 0,
    "right": 1,
    "stop": 2,
    "uturn": 3  
}

def load_model(path="model_weights.npz"):
    data = np.load(path)

    fixed_kernels = data["fixed_kernels"]
    learnable_kernels = data["learnable_kernels"]

    kernels = np.concatenate([fixed_kernels, learnable_kernels], axis=0)

    return kernels, data["w1"], data["b1"]

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


# Load model
kernels, w1, b1 = load_model()

# Load ANY real image
img_path = "/path/to/your/test/image.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (64,64))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img / 255.0

pred_class, probs = predict(img, kernels, w1, b1)

labels = ["left", "right", "stop", "uturn"]

print("Prediction:", labels[pred_class])
print("Confidence:", probs)