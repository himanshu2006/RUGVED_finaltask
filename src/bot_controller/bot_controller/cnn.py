import os
import cv2
import numpy as np

# ==========================================
# 1. SETTINGS & PREPROCESSING
# ==========================================
DATA_DIR = "/home/himanshu/ws_final/src/bot_controller/master_dataset/data/test"
WEIGHTS_PATH = "/home/himanshu/ws_final/src/model_weights.npz"
CLASSES = {'left': 0, 'right': 1, 'stop': 2, 'uturn': 3}

def load_dataset(data_dir, max_samples=200):
    images, labels = [], []
    print(f" Loading and Augmenting dataset from {data_dir}...")
    
    for name, lbl in CLASSES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_samples]
        
        for f in files:
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized = cv2.resize(img, (32, 32))
                
                # 1. Store Original
                images.append(resized)
                labels.append(lbl)
                
                # 2. TARGETED DATA AUGMENTATION
                if name == 'left':
                    images.append(cv2.flip(resized, 1))
                    labels.append(CLASSES['right'])
                elif name == 'right':
                    images.append(cv2.flip(resized, 1))
                    labels.append(CLASSES['left'])
                elif name == 'stop':
                    # Stop signs get Lighting Augmentation (Shadows & Glare)
                    dark = np.clip(resized * 0.6, 0, 255).astype(np.uint8)
                    images.append(dark)
                    labels.append(lbl)
                    
                    bright = np.clip(resized * 1.4, 0, 255).astype(np.uint8)
                    images.append(bright)
                    labels.append(lbl)
                elif name == 'uturn':
                    # U-turns get slight noise
                    noise = np.random.normal(0, 5, resized.shape).astype(np.float32)
                    noisy_img = np.clip(resized + noise, 0, 255).astype(np.uint8)
                    images.append(noisy_img)
                    labels.append(lbl)

    processed_images = []
    for img in images:
        std_img = (img - np.mean(img)) / (np.std(img) + 1e-7)
        processed_images.append(std_img)
                
    X = np.array(processed_images, dtype=np.float32).reshape(-1, 32, 32, 1)
    Y = np.array(labels, dtype=np.int32)
    print(f"📈 Total images after targeted augmentation: {len(X)}")
    return X, Y

# ==========================================
# 2. VECTORIZED LAYERS
# ==========================================
class Conv2D:
    # Changed default f_size to 5 for Macro-vision
    def __init__(self, num_filters=32, f_size=5, l2_reg=0.005):
        self.num_filters = num_filters
        self.f_size = f_size
        self.l2_reg = l2_reg
        self.filters = np.random.randn(num_filters, f_size, f_size) * np.sqrt(2.0 / (f_size * f_size))

    def forward(self, input_data):
        self.last_input = input_data
        b, h, w, _ = input_data.shape
        oh, ow = h - self.f_size + 1, w - self.f_size + 1
        out = np.zeros((b, oh, ow, self.num_filters))
        for i in range(self.f_size):
            for j in range(self.f_size):
                region = input_data[:, i:i+oh, j:j+ow, 0:1] 
                weight = self.filters[:, i, j].reshape(1, 1, 1, -1)
                out += region * weight
        return out

    def backward(self, d_out, lr):
        b, oh, ow, _ = d_out.shape
        d_f = np.zeros_like(self.filters)
        d_in = np.zeros_like(self.last_input)
        for i in range(self.f_size):
            for j in range(self.f_size):
                region = self.last_input[:, i:i+oh, j:j+ow, 0:1]
                d_f[:, i, j] = np.sum(region * d_out, axis=(0, 1, 2))
                weight = self.filters[:, i, j].reshape(1, 1, 1, -1)
                d_in[:, i:i+oh, j:j+ow, 0] += np.sum(d_out * weight, axis=3)
        
        self.filters -= lr * (np.clip(d_f, -1.0, 1.0) + self.l2_reg * self.filters)
        return d_in

class LeakyReLU:
    def __init__(self, alpha=0.01): self.alpha = alpha
    def forward(self, x):
        self.last_x = x
        return np.where(x > 0, x, x * self.alpha)
    def backward(self, d_out):
        dx = d_out.copy()
        dx[self.last_x <= 0] *= self.alpha
        return dx

class MaxPooling2D:
    def __init__(self, pool_size=2): self.pool_size = pool_size
    def forward(self, input_data):
        self.last_input = input_data
        b, h, w, c = input_data.shape
        ph, pw = self.pool_size, self.pool_size
        oh, ow = h // ph, w // pw
        reshaped = input_data.reshape(b, oh, ph, ow, pw, c)
        self.out = reshaped.max(axis=(2, 4))
        return self.out
    def backward(self, d_out):
        b, h, w, c = self.last_input.shape
        ph, pw = self.pool_size, self.pool_size
        oh, ow = h // ph, w // pw
        out_expanded = self.out.reshape(b, oh, 1, ow, 1, c)
        reshaped_in = self.last_input.reshape(b, oh, ph, ow, pw, c)
        mask = (reshaped_in == out_expanded)
        d_out_expanded = d_out.reshape(b, oh, 1, ow, 1, c)
        d_in_reshaped = mask * d_out_expanded
        return d_in_reshaped.reshape(b, h, w, c)

class Dropout:
    def __init__(self, rate=0.3): self.rate = rate
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
            return x * self.mask
        return x
    def backward(self, d_out): return d_out * self.mask

class Dense:
    def __init__(self, in_dim, out_dim, l2_reg=0.005):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.l2_reg = l2_reg
    def forward(self, x):
        self.last_in = x
        return np.dot(x, self.w) + self.b
    def backward(self, d_out, lr):
        dw = np.dot(self.last_in.T, d_out)
        dx = np.dot(d_out, self.w.T)
        self.w -= lr * (np.clip(dw, -1.0, 1.0) + self.l2_reg * self.w)
        self.b -= lr * np.clip(np.sum(d_out, axis=0), -1.0, 1.0)
        return dx

class SoftmaxLoss:
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = e / np.sum(e, axis=1, keepdims=True)
        return self.probs
    def backward(self, y):
        dx = self.probs.copy()
        dx[np.arange(len(y)), y] -= 1
        return dx / len(y)
    def get_loss(self, y):
        return -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-8))

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    X_train, y_train = load_dataset(DATA_DIR)
    if len(X_train) == 0: return

    counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(counts) * count) if count > 0 else 1.0 for i, count in enumerate(counts)}
    print("⚖️ Class Weights Applied:", class_weights)

    # Init Layers: Conv uses 5x5 filter. 
    # Output of 32x32 -> Conv(5x5) -> 28x28 -> Pool(2x2) -> 14x14
    # 14 * 14 * 32 filters = 6272 Flattened inputs
    conv = Conv2D(32, 5, l2_reg=0.005); l_conv = LeakyReLU(); pool = MaxPooling2D()
    d1 = Dense(6272, 128, l2_reg=0.005); l1 = LeakyReLU(); drop1 = Dropout(0.3)
    d2 = Dense(128, 64, l2_reg=0.005); l2 = LeakyReLU(); drop2 = Dropout(0.3)
    out_layer = Dense(64, 4) 
    loss_fn = SoftmaxLoss()

    lr = 0.01
    decay = 0.96 # Slightly slower decay to learn the complex 5x5 shapes
    epochs = 50
    batch_size = 32

    print(f"🚀 Training Brain...")
    for epoch in range(epochs):
        current_lr = lr * (decay ** epoch)
        indices = np.random.permutation(len(X_train))
        X_s, y_s = X_train[indices], y_train[indices]
        
        epoch_loss, correct = 0, 0
        for i in range(0, len(X_train), batch_size):
            xb, yb = X_s[i:i+batch_size], y_s[i:i+batch_size]
            curr_batch = len(xb)
            
            pool_out = pool.forward(l_conv.forward(conv.forward(xb)))
            flat = pool_out.reshape(curr_batch, -1)
            h1 = drop1.forward(l1.forward(d1.forward(flat)), training=True)
            h2 = drop2.forward(l2.forward(d2.forward(h1)), training=True)
            probs = loss_fn.forward(out_layer.forward(h2))
            
            dy = loss_fn.backward(yb)
            sample_weights = np.array([class_weights[y] for y in yb])
            dy = dy * sample_weights[:, None]

            g = out_layer.backward(dy, current_lr)
            g = d2.backward(l2.backward(drop2.backward(g)), current_lr)
            g = d1.backward(l1.backward(drop1.backward(g)), current_lr)
            
            g_reshaped = g.reshape(pool_out.shape)
            conv.backward(l_conv.backward(pool.backward(g_reshaped)), current_lr)

            epoch_loss += loss_fn.get_loss(yb) * curr_batch
            correct += np.sum(np.argmax(probs, axis=1) == yb)

        print(f"Epoch {epoch+1:02d} | Acc: {correct/len(X_train):.2%} | Loss: {epoch_loss/len(X_train):.4f} | LR: {current_lr:.6f}")

    np.savez(WEIGHTS_PATH, conv=conv.filters, d1w=d1.w, d1b=d1.b, d2w=d2.w, d2b=d2.b, ow=out_layer.w, ob=out_layer.b)
    print("✅ Brain Saved Successfully!")

if __name__ == "__main__":
    train()