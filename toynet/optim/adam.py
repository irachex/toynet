import collections

from .optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, params_or_target, lr,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, t=2):
        super().__init__(params_or_target)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.v = collections.defaultdict(lambda: 0)
        self.s = collections.defaultdict(lambda: 0)

    def step(self):
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        t = self.t
        v = self.v
        s = self.s
        lr = self.lr

        for p in self.params:
            v[p] = beta1 * v[p] + (1 - beta1) * p.grad
            v_corrected = v[p] / (1 - beta1 ** t)
            s[p] = beta2 * s[p] + (1 - beta2) * (p.grad ** t)
            s_corrected = s[p] / (1 - beta2 ** t)
            p.value -= lr * v_corrected / (s_corrected ** (1/t) + epsilon)
