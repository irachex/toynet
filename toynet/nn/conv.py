from .node import Node
from .param import Param


class _ConvNd(Node):

    def __init__(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, name=None):
        super().__init__(name=name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = Param(shape=(self.out_channels, ) + self.kernel_size,
                       name='{}:W'.format(self.name))
        if bias:
            self.b = Param(shape=(self.out_channels, 1),
                           name='{}:b'.format(self.name))
        else:
            self.b = 0.

    def forward(self):
        pass

    def backward(self):
        pass


class Conv1d(_ConvNd):
    pass


class Conv2d(_ConvNd):

    def __init__(self, stride, padding):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
