from .node import Node
from .param import Param


class Linear(Node):

    r"""
    y = Wx + b
    """

    def __init__(self, units, name=name):
        super().__init__(name=name)
        self.units = units
        self.W = self.b = None

    def __call__(self, x):
        super().__call__(x)

        self.W = Param(name='{}:W'.format(self.name))
        self.b = Param(name='{}:b'.format(self.name))
        return self

    def forward(self):
        x = self.in_edges[0].value
        W, b = self.W.value, self.b.value
        return W @ x + b

    def backward(self):
        pass


FC = FullyConnected = Linear
