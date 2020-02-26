from .node import Node
from .param import Param


class _ConvNd(Node):

    n_dim = NotImplemented

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, name=None):
        super().__init__(name=name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernal_size, ) * self.n_dim

        self.stride = stride
        self.padding = padding
        self.bias = bias

    def __call__(self, x):
        super().__call__(x)

        # in_channels = x.shape[-1]  # n, h, w, c
        # self.in_channels = in_channels

        self.kernal = Param(shape=(self.in_channels,) + self.kernel_size,
                            name='{}:W'.format(self.name))
        if self.bias:
            self.bias = Param(shape=(self.in_channels,),
                              name='{}:b'.format(self.name))

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Conv1d(_ConvNd):

    n_dim = 1


class Conv2d(_ConvNd):

    n_dim = 2

    def __init__(self, stride, padding):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


def conv_bf(x, kernel, padding, stride):
    out = np.zeros((x.shape))


def conv_matmul():
    pass
