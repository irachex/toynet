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


class Reshape(Node):

    def __init__(self, to_shape, name=None):
        super().__init__(name=name)
        self.to_shape = to_shape
        self.from_shape = None

    def __call__(self, x):
        super().__call__(x)
        self._set_shape_by(x)
        return self

    def _set_shape_by(self, x):
        if -1 not in self.to_shape:
            self.shape = self.to_shape
            return

        p = 1
        for k in x.shape:
            p *= k
        for k in self.to_shape:
            if k != -1:
                p /= k
        self.shape = tuple(int(p) if k == -1 else k for k in self.to_shape)

    def forward(self):
        x = self.in_edges[0]
        self.from_shape = x.value.shape
        ret = x.value.reshape(self.to_shape)
        self.shape = ret.shape
        return ret

    def backward(self):
        x = self.in_edges[0]
        x.grad += self.grad.reshape(self.from_shape)
