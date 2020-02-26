from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params_or_target, lr):
        super().__init__(params_or_target)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.value -= self.lr * p.grad
