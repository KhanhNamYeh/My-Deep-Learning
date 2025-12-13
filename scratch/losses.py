"""
Loss functions (MSE, CrossEntropy) providing loss and gradient for our handcrafted networks.
"""
import numpy as np

class MSELoss:
    def loss_and_grad(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / (2 * m)
        grad = (y_pred - y_true) / m
        return loss, grad


class CrossEntropyLoss:
    def loss_and_grad(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(probs + 1e-12)) / m
        grad = (probs - y_true) / m
        return loss, grad

