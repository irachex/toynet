import numpy as np

from .node import Node


class Const(Node):

    def __init__(self, value, name=None):
        super().__init__(name=name)
        self.value = value

    def forward(self):
        return self.value


class Input(Node):

    def __init__(self, shape=(1,), dtype=np.float32, name=None):
        super().__init__(name=name)
        self.shape = shape
        self.dtype = dtype
        self.value = np.zeros(shape=shape, dtype=dtype)

    def forward(self):
        return self.value

    def assign(self, data):
        # TODO: check shape
        self.value = data


class Param(Node):

    def __init__(self, shape=(1,), dtype=np.float32, name=None):
        super().__init__(name=name)
        self.shape = shape
        self.dtype = dtype
        self.value = np.zeros(shape=shape, dtype=dtype)

    def forward(self):
        return self.value

    def assign(self, data):
        # TODO: check shape
        self.value = data

    def freeze(self):
        pass
        # TODO:
