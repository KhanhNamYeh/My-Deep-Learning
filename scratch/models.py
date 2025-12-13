"""
Lightweight model skeletons: defines Module base with forward/backward/update hooks 
and a train_step helper used by custom layers/losses.
"""
class Module:
    def forward(self, X):
        raise NotImplementedError("forward not implemented")

    def backward(self, grad):
        raise NotImplementedError("backward not implemented")

    def update(self):
        raise NotImplementedError("update not implemented")

    def train_step(self, X, y):
        if not hasattr(self, "loss_fn"):
            raise AttributeError("Module needs a loss function to perform train_step")
        logits = self.forward(X)
        loss, grad = self.loss_fn.loss_and_grad(logits, y)
        self.backward(grad)
        self.update()
        return loss