import numpy as np

from .node import Node
from .param import Param


class FullyConnected(Node):

    r"""
    y = Wx + b
    """

    def __init__(self, num_neurons, name=None):
        super().__init__(name=name)
        self.num_neurons = num_neurons
        self.W = self.b = None

    def __call__(self, x):
        _, num_inputs = x.shape  # n, num_inputs, 1
        self.num_inputs = num_inputs

        self.W = Param(shape=(self.num_neurons, num_inputs),
                       name='{}:W'.format(self.name))
        self.b = Param(shape=(self.num_neurons, 1),
                       name='{}:b'.format(self.name))

        super().__call__(x, self.W, self.b)
        self._set_shape_by(x)
        return self

    def _set_shape_by(self, x):
        assert len(x.shape) == 2, 'shape should be N,M: {}'.format(x.shape)
        _, num_inputs = x.shape
        self.shape = (x.shape[0], self.num_neurons, 1)

    def forward(self):
        x = self.in_edges[0].value  # n, m
        x = x.reshape(x.shape + (1,))
        W, b = self.W.value, self.b.value
        out = np.zeros(self.shape, dtype=x.dtype)
        for i in range(x.shape[0]):
            out[i] = W @ x[i] + b
        return out

    def backward(self):
        x = self.in_edges[0]
        W, b = self.W, self.b

        # for i in range(x.shape[0]):
        #     x.grad[i] += self.W.value * self.grad[i]
        #     W.grad += x.value[i] * self.grad[i]
        #     b.grad += self.grad[i]

        for i in range(x.shape[0]):
            for j in range(self.num_neurons):
                chain_grad_i = self.grad[i, j]
                for k in range(self.num_inputs):
                    x.grad[i, k] += W.value[j, k] * chain_grad_i
                    W.grad[j, k] += x.value[i, k] * chain_grad_i
                self.b.grad[j] += chain_grad_i

Linear = FC = FullyConnected
