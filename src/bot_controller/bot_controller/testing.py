import os
import cv2
import numpy as np

TEST_DATA_DIR = "/home/himanshu/ws_final/src/bot_controller/master_dataset/data/val"
WEIGHTS_PATH = "/home/himanshu/ws_final/src/model_weights.npz"
CLASSES = {'left': 0, 'right': 1, 'stop': 2, 'uturn': 3}

# --- MINIMAL INFERENCE ARCHITECTURE ---
class Conv2D:
    def __init__(self, num_filters=32, f_size=5): # Matches new 5x5 size
        self.num_filters = num_filters
        self.f_size = f_size
        self.filters = None 
    def forward(self, input_data):
        b, h, w, _ = input_data.shape
        oh, ow = h - self.f_size + 1, w - self.f_size + 1
        out = np.zeros((b, oh, ow, self.num_filters))
        for i in range(self.f_size):
            for j in range(self.f_size):
                region = input_data[:, i:i+oh, j:j+ow, 0:1] 
                weight = self.filters[:, i, j].reshape(1, 1, 1, -1)
                out += region * weight
        return out

class LeakyReLU:
    def __init__(self, alpha=0.01): self.alpha = alpha
    def forward(self, x): return np.where(x > 0, x, x * self.alpha)

class MaxPooling2D:
    def __init__(self, pool_size=2): self.pool_size = pool_size
    def forward(self, input_data):
        b, h, w, c = input_data.shape
        ph, pw = self.pool_size, self.pool_size
        oh, ow = h // ph, w // pw
        return input_data.reshape(b, oh, ph, ow, pw, c).max(axis=(2, 4))

class Dropout:
    def forward(self, x, training=False): return x

class Dense:
    def __init__(self, in_dim, out_dim):
        self.w = None; self.b = None 
    def forward(self, x): return np.dot(x, self.w) + self.b

class SoftmaxLoss:
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

# --- TEST EXECUTION ---
def run_test():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ ERROR: Weights not found at {WEIGHTS_PATH}.")
        return

    data = np.load(WEIGHTS_PATH)
    
    # Init Layers with 32 filters (5x5) and 6272 inputs
    conv = Conv2D(32, 5); conv.filters = data['conv']
    l_conv = LeakyReLU(); pool = MaxPooling2D()
    d1 = Dense(6272, 128); d1.w, d1.b = data['d1w'], data['d1b']
    l1 = LeakyReLU(); drop1 = Dropout()
    d2 = Dense(128, 64); d2.w, d2.b = data['d2w'], data['d2b']
    l2 = LeakyReLU(); drop2 = Dropout()
    out_layer = Dense(64, 4); out_layer.w, out_layer.b = data['ow'], data['ob']
    softmax_fn = SoftmaxLoss()

    images, labels = [], []
    for name, lbl in CLASSES.items():
        path = os.path.join(TEST_DATA_DIR, name)
        if not os.path.exists(path): continue
        for f in os.listdir(path):
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized = cv2.resize(img, (32, 32))
                std_img = (resized - np.mean(resized)) / (np.std(resized) + 1e-7)
                images.append(std_img)
                labels.append(lbl)

    X_test = np.array(images).reshape(-1, 32, 32, 1)
    y_test = np.array(labels)

    print(f"🔍 Analyzing {len(X_test)} test images...")
    
    pool_out = pool.forward(l_conv.forward(conv.forward(X_test)))
    flat = pool_out.reshape(len(X_test), -1)
    h1 = drop1.forward(l1.forward(d1.forward(flat)), training=False)
    h2 = drop2.forward(l2.forward(d2.forward(h1)), training=False)
    probs = softmax_fn.forward(out_layer.forward(h2))
    
    preds = np.argmax(probs, axis=1)

    print("\n" + "="*40)
    print("       FINAL PER-SIGN PERFORMANCE")
    print("="*40)
    
    for name, lbl in CLASSES.items():
        idx = (y_test == lbl)
        if np.any(idx):
            correct = np.sum(preds[idx] == y_test[idx])
            total = np.sum(idx)
            print(f"{name.upper():<10} | Acc: {correct/total:>6.2%} ({correct}/{total})")
    
    print("-" * 40)
    print(f"TOTAL SYSTEM ACCURACY: {np.mean(preds == y_test):.2%}")
    print("="*40)

if __name__ == "__main__":
    run_test()