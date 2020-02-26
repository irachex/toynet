import numpy as np

from .node import Node
from .param import Param


class _ConvND(Node):

    n_dim = NotImplemented

    def __init__(self, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, name=None):
        super().__init__(name=name)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, ) * self.n_dim
        else:
            self.kernel_size = tuple(self.kernel_size)

        self.stride = stride
        self.padding = padding
        self.bias = bias

    def __call__(self, x):
        in_channels = x.shape[-1]  # n, h, w, c
        self.in_channels = in_channels

        self.kernel = Param(shape=(self.out_channels,) + self.kernel_size,
                            name='{}:W'.format(self.name))
        if self.bias:
            self.bias = Param(shape=(self.out_channels,),
                              name='{}:b'.format(self.name))

        super().__call__(x, self.kernel, self.bias)
        self._set_shape_by(x)
        return self

    def _set_shape_by(self, x):
        raise NotImplementedError


class Conv2D(_ConvND):

    n_dim = 2

    def _set_shape_by(self, x):
        assert len(x.shape) == 4, 'shape should be NHWC'
        h, w, c = x.shape[-3:]
        kh, kw = self.kernel_size
        p = self.padding
        s = self.stride

        oh = int((h + 2 * p - kh) / s + 1)
        ow = int((w + 2 * p - kw) / s + 1)
        oc = self.out_channels
        self.shape = (x.shape[0], oh, ow, oc)

    def forward(self):
        x = self.in_edges[0].value
        kernel = self.kernel.value
        bias = self.bias.value if self.bias else 0
        out = np.zeros(self.shape, dtype=x.dtype)

        return conv2d_bf(out, x, kernel, bias, self.stride, self.padding)

    def backward(self):
        x = self.in_edges[0]
        return conv2d_backward_bf(
            self.grad, self.value, x, self.kernel, self.bias, self.stride, self.padding)


def conv2d_bf(out, x, kernel, bias, stride, padding):
    n, h, w, c = x.shape
    _, oh, ow, oc = out.shape
    kh, kw, kc = kernel.shape

    for i in range(n):

        for ic in range(oc):
            sh = -padding
            for ih in range(oh):
                sw = -padding
                for iw in range(ow):

                    for jh in range(kh):
                        for jw in range(kw):
                            for jc in range(kc):
                                if (sh + jh < 0 or sh + jh >= h or
                                    sw + jw < 0 or sw + jw >= w):
                                    continue

                                out[i, ih, iw, ic] += x[i, sh + jh, sw + jw, ic] * kernel[jh, jw, jc]

                    sw += stride
                sh += stride

    out[:, :, :] += bias
    return out


def conv2d_backward_bf(chain_grad, out, x, kernel, bias, stride, padding):
    n, h, w, c = x.shape
    _, oh, ow, oc = out.shape
    kh, kw, kc = kernel.shape

    for i in range(n):

        for ic in range(oc):
            sh = -padding
            for ih in range(oh):
                sw = -padding
                for iw in range(ow):
                    chain_grad_i = chain_grad[i, ih, iw, ic]

                    for jh in range(kh):
                        for jw in range(kw):
                            for jc in range(kc):
                                if (sh + jh < 0 or sh + jh >= h or
                                    sw + jw < 0 or sw + jw >= w):
                                    continue

                                # out[i, ih, iw, ic] += x[i, sh + jh, sw + jw, ic] * kernel[jh, jw, jc]
                                kernel.grad[jh, jw, jc] += x.value[i, sh + jh, sw + jw, ic] * chain_grad_i
                                x.grad[jh, jw, jc] += kernel.value[i, jh, jw, jc] * chain_grad_i

                    bias.grad[ic] += chain_grad_i

                    sw += stride
                sh += stride


def conv2d_matmul():
    pass
