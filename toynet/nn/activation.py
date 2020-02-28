import numpy as np

from .node import Node


class ReLU(Node):

    def forward(self):
        x = self.in_edges[0]
        return np.where(x.value < 0, 0, x.value)

    def backward(self):
        x = self.in_edges[0]
        x.grad += np.where(x.value < 0, 0, self.grad)


class Sigmoid(Node):

    def forward(self):
        x = self.in_edges[0]
        return 1. / (1. + np.exp(-x.value))

    def backward(self):
        x = self.in_edges[0]
        v = self.value
        x.grad += v * (1 - v) * self.grad


class Tanh(Node):

    def forward(self):
        x = self.in_edges[0]
        return np.tanh(x.value)

    def backward(self):
        x = self.in_edges[0]
        v = self.value
        x.grad += (1 - v * v) * self.grad


class Softmax(Node):

    def forward(self):
        x = self.in_edges[0].value
        x_max = np.max(x, axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)
        s = np.sum(x_exp, axis=-1, keepdims=True)
        return x_exp / s

    def backward(self):
        x = self.in_edges[0]
        p = self.value
        for i in range(p.shape[-1]):
            for j in range(p.shape[-1]):
                if i == j:
                    x.grad[:, i] += p[:, i] * (1 - p[:, i]) * self.grad[:, i]
                else:
                    x.grad[:, i] += - p[:, i] * p[:, j] * self.grad[:, i]

