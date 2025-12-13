"""
Core layer implementations: dense, activations, conv2d, pooling, flatten utilities 
for the custom NumPy-based networks.
"""
import numpy as np
# ---- Base Layer ----
class Layer:
    # Forward pass interface to be implemented by subclasses
    def forward(self, X):
        raise NotImplementedError

    # Backward pass interface to be implemented by subclasses
    def backward(self, grad):
        raise NotImplementedError

    # Parameter update hook (no-op for layers without params)
    def update(self, lr):
        # general no parameters to update
        return
    
class Activation(Layer):
    def __init__(self):
        self.mask = None # Store mask for backward pass

class ReLU(Activation):
    def __init__(self):
        super().__init__()
        
    # Apply ReLU elementwise and store mask for backprop
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    # Backpropagate through ReLU using stored mask
    def backward(self, dA):
        if self.mask is None:
            raise RuntimeError("ReLU backward called before forward")
        return dA * self.mask
    
class Linear(Activation):
    def __init__(self):
        super().__init__()
        
    # Identity activation: passthrough
    def forward(self, X):
        return X

    # Backprop for identity is passthrough
    def backward(self, dA):
        return dA

# ---- High Layers ----
class Dense(Layer):
    def __init__(self, input_dim, output_dim, seed=42, activation=ReLU()):
        # Initialize weights with random initialization, bias to zeros
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        # Store for backward pass
        self.cache = None # Store input
        self.activation = activation 
        self.dW = None
        self.db = None

    # Affine forward + activation
    def forward(self, X):
        self.cache = X
        Z = X.dot(self.W) + self.b
        A = self.activation.forward(Z)
        return A

    # Backprop through activation then affine
    def backward(self, dA):
        dZ = self.activation.backward(dA)
        X = self.cache
        m = X.shape[0]
        self.dW = X.T.dot(dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return dZ.dot(self.W.T)

    # Gradient step on weights and bias
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
# ---- Convolutional and Pooling Layers ----

class Conv2D(Layer):
    def __init__(self, in_channels, num_filters, kernel_size=3, stride=1, padding=0, seed=42):
        np.random.seed(seed)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        self.W = np.random.randn(num_filters, kernel_size, kernel_size, in_channels) * np.sqrt(
            2.0 / (kernel_size * kernel_size * in_channels)
        )
        self.b = np.zeros((num_filters,))
        self.cache = None
        self.dW = None
        self.db = None

    # Zero-pad input if padding > 0
    def pad(self, X):
        if self.padding == 0:
            return X
        return np.pad(
            X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
        )

    # Naive convolution forward (NHWC)
    def forward(self, X):
        m, H, W, C = X.shape
        if C != self.in_channels:
            raise ValueError("Input channels do not match!")
        k = self.kernel_size
        s = self.stride
        Xp = self.pad(X)
        Hp, Wp, _ = Xp.shape[1:]
        out_h = (Hp - k) // s + 1
        out_w = (Wp - k) // s + 1
        Z = np.zeros((m, out_h, out_w, self.num_filters))
        for n in range(m):
            for f in range(self.num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        hs = i * s
                        ws = j * s
                        region = Xp[n, hs:hs+k, ws:ws+k, :]
                        Z[n, i, j, f] = np.sum(region * self.W[f]) + self.b[f]
        self.cache = (X, Xp)
        return Z

    # Naive convolution backward: compute dW, db, and input gradient
    def backward(self, dZ):
        X, Xp = self.cache
        m, H, W, C = X.shape
        k = self.kernel_size
        s = self.stride
        Hp, Wp, _ = Xp.shape[1:]
        out_h = (Hp - k) // s + 1
        out_w = (Wp - k) // s + 1
        dXp = np.zeros_like(Xp)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        for n in range(m):
            for f in range(self.num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        hs = i * s
                        ws = j * s
                        region = Xp[n, hs:hs+k, ws:ws+k, :]
                        dW[f] += region * dZ[n, i, j, f]
                        db[f] += dZ[n, i, j, f]
                        dXp[n, hs:hs+k, ws:ws+k, :] += self.W[f] * dZ[n, i, j, f]
        # remove padding from dXp
        if self.padding > 0:
            dX = dXp[:, self.padding : self.padding + H, self.padding : self.padding + W, :]
        else:
            dX = dXp
        self.dW = dW / m
        self.db = db / m
        return dX

    # Apply gradient step to filters and biases
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class Pool2D(Layer):
    def __init__(self, pool_size=2, stride=2, type='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        if type != 'max': 
            raise NotImplementedError("Not implemented pooling type other than 'max'")
        else:
            self.type = type

    # Forward max pooling (NHWC) with mask for argmax
    def forward(self, X):
        m, H, W, C = X.shape
        k = self.pool_size
        s = self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        out = np.zeros((m, out_h, out_w, C))
        mask = np.zeros_like(X, dtype=bool)
        for n in range(m):
            for i in range(out_h):
                for j in range(out_w):
                    hs = i * s
                    ws = j * s
                    region = X[n, hs:hs+k, ws:ws+k, :]
                    max_vals = np.max(region, axis=(0, 1))
                    out[n, i, j, :] = max_vals
                    for c in range(C):
                        idx = np.unravel_index(np.argmax(region[:, :, c]), (k, k))
                        mask[n, hs + idx[0], ws + idx[1], c] = True
        self.cache = (X, mask)
        return out

    # Backprop through max pooling using stored mask
    def backward(self, dA):
        X, mask = self.cache
        dX = np.zeros_like(X)
        m, H, W, C = X.shape
        k = self.pool_size
        s = self.stride
        out_h = (H - k) // s + 1
        out_w = (W - k) // s + 1
        for n in range(m):
            for i in range(out_h):
                for j in range(out_w):
                    hs = i * s
                    ws = j * s
                    for c in range(C):
                        dX[n, hs:hs+k, ws:ws+k, c] += mask[n, hs:hs+k, ws:ws+k, c] * dA[n, i, j, c]
        return dX

class Flatten(Layer):
    def __init__(self):
        self.orig_shape = None

    # Flatten spatial dims to (batch, -1)
    def forward(self, X):
        self.orig_shape = X.shape
        return X.reshape(X.shape[0], -1)

    # Reshape gradient back to original tensor shape
    def backward(self, dA):
        return dA.reshape(self.orig_shape)

# ---- Model skeleton ----

class NeuralNet:
    def __init__(self, layers, loss_fn, learning_rate=0.01):
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = learning_rate

    # Forward pass through sequential layers
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # Backward pass through layers in reverse
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # Update parameters for all layers that implement update
    def update(self):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(self.lr)

    # One training step: forward, loss/grad, backward, update
    def train_step(self, X, y):
        logits = self.forward(X)
        loss, grad = self.loss_fn.loss_and_grad(logits, y)
        self.backward(grad)
        self.update()
        return loss



