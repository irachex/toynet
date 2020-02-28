import numpy as np


global_cnt = 0

def gen_next():
    global global_cnt
    global_cnt += 1
    return global_cnt


class OperatorMixin:

    def __add__(self, other):
        from .arithmetic import Add
        return Add()(self, other)

    def __sub__(self, other):
        from .arithmetic import Sub
        return Sub()(self, other)

    def __mul__(self, other):
        from .arithmetic import Mul
        return Mul()(self, other)

    def __truediv__(self, other):
        from .arithmetic import Div
        return Div()(self, other)

    def __mod__(self, other):
        from .arithmetic import Mul
        return Mod()(self, other)

    def __pow__(self, other):
        from .arithmetic import Pow
        return Pow()(self, other)

    def __matmul__(self, other):
        from .arithmetic import MatMul
        return MatMul()(self, other)

    def sum(self, axis=None, keepdims=False):
        from .reduction import ReduceSum
        return ReduceSum(axis=axis, keepdims=keepdims)(self)

    def mean(self, axis=None, keepdims=False):
        from .reduction import ReduceMean
        return ReduceMean(axis=axis, keepdims=keepdims)(self)

    def reshape(self, to_shape):
        from .reduction import Reshape
        return Reshape(to_shape)(self)


class Node(OperatorMixin):

    def __init__(self, name=None, requires_grad=True):
        self.name = name or '{}{}'.format(self.__class__.__name__, gen_next())

        self.in_edges = []
        self.out_edges = []

        self.value = None
        self._grad = 0.
        self.shape = None
        self.requires_grad = requires_grad

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if self.requires_grad:
            self._grad = value

    def forward(self):
        raise NotImplementedError

    def backward(self):
        pass

    def eval(self, **inputs):
        from .network import forward_propagation
        return forward_propagation(self, inputs=inputs)

    def __call__(self, *nodes):
        from .param import Const
        nodes = [x if isinstance(x, Node) else Const(x)
                 for x in nodes]
        self._add_edges_from(nodes)

        self.shape = nodes[0].shape
        return self

    def _add_edges_from(self, nodes):
        self.in_edges.extend(nodes)

        for x in nodes:
            x.out_edges.append(self)
        return self

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return '{}'.format(self.name)
