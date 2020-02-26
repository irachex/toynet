import toynet as tn


class Optimizer:

    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for x in self.params:
            x.grad = 0.

    def step(self):
        raise NotImplementedError
