import collections

from .optimizer import Optimizer


class Momentum(Optimizer):

    def __init__(self, params_or_target, lr, beta=0.9):
        super().__init__(params_or_target)
        self.lr = lr
        self.beta = beta
        self.v = collections.defaultdict(lambda: 0)

    def step(self):
        beta = self.beta
        v = self.v
        lr = self.lr
        for p in self.params:
            v[p] = beta * v[p] + (1 - beta) * p.grad
            p.value -= lr * v[p]
