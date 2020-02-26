import numpy as np

from .node import Node


class ReduceSum(Node):

    def __init__(self, axis=None, keepdims=False, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        x = self.in_edges[0]
        return np.sum(x.value, axis=self.axis, keepdims=self.keepdims)

    def backward(self):
        x = self.in_edges[0]
        x.grad += self.grad


class ReduceMean(Node):

    def __init__(self, axis=None, keepdims=False, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        x = self.in_edges[0]
        return np.mean(x.value, axis=self.axis, keepdims=self.keepdims)

    def backward(self):
        x = self.in_edges[0]
        x.grad += self.grad / (x.value.size / self.value.size)
