from .node import Node


class ReLU(Node):

    def forward(self):
        x = self.in_edges[0]
        return np.where(x.value < 0, 0, x.value)

    def backward(self):
        pass


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
        x = self.in_edges[0]
        return

    def backward(self):
        pass
