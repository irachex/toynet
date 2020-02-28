import operator

from .node import Node


class ArithmeticNode(Node):

    def forward(self):
        args = list(x.value for x in self.in_edges)
        op = getattr(operator, self.__class__.__name__.lower())
        return op(*args)


class Add(ArithmeticNode):

    def backward(self):
        a, b = self.in_edges
        a.grad += self.grad
        b.grad += self.grad

class Sub(ArithmeticNode):

    def backward(self):
        a, b = self.in_edges
        a.grad += self.grad
        b.grad -= self.grad

class Mul(ArithmeticNode):

    def backward(self):
        a, b = self.in_edges
        a.grad += b.value * self.grad
        b.grad += a.value * self.grad

class TrueDiv(ArithmeticNode):

    def backward(self):
        return

Div = TrueDiv


class Mod(ArithmeticNode):

    def backward(self):
        return


class Pow(ArithmeticNode):

    def backward(self):
        a, b = self.in_edges
        a.grad += b.value * (a.value ** (b.value - 1)) * self.grad
        b.grad += 0  # TODO


class Exp(ArithmeticNode):

    def forward(self):
        x = self.in_edges[0]
        return np.exp(x)

    def backward(self):
        pass


class Log(ArithmeticNode):

    def forward(self):
        x = self.in_edges[0]
        return np.log(x)

    def backward(self):
        pass


class MatMul(ArithmeticNode):

    def backward(self):
        pass
