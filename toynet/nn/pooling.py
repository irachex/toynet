import numpy as np

from .node import Node


class Pooling(Node):

    n_dim = NotImplemented

    def __init__(self, window_size, stride=1, padding=0, name=None):
        super().__init__(name=name)

        self.window_size = window_size
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, ) * self.n_dim
        else:
            self.window_size = tuple(self.window_size)

        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        super().__call__(x)
        self._set_shape_by(x)
        return self

    def _set_shape_by(x):
        raise NotImplementedError


class Pooling2D(Pooling):

    n_dim = 2
    mode = NotImplemented

    def _set_shape_by(self, x):
        assert len(x.shape) == 4, 'shape should be NHWC'
        h, w, c = x.shape[-3:]
        kh, kw = self.window_size
        p = self.padding
        s = self.stride

        oh = int((h + 2 * p - kh) / s + 1)
        ow = int((w + 2 * p - kw) / s + 1)
        oc = c
        self.shape = (x.shape[0], oh, ow, oc)

    def forward(self):
        x = self.in_edges[0].value
        self.value = np.zeros(self.shape, dtype=x.dtype)
        # store switches for x,y coordinates for where the max comes from, for each output neuron
        self.switch = np.zeros(self.shape, dtype=np.object)

        return pooling_2d_bf(self.value, self.switch, x, self.window_size, self.stride, self.padding, mode=self.mode)

    def backward(self):
        x = self.in_edges[0]
        return pooling_2d_backward_bf(
            self.grad, self.value, self.switch, x, self.kernel, self.stride, self.padding, mode=self.mode)



class MaxPooling2D(Pooling2D):

    mode = 'max'


class AvgPooling2D(Pooling2D):

    mode = 'avg'


def pooling_2d_bf(out, switch, x, window_size, stride, padding, mode):
    n, h, w, c = x.shape
    _, oh, ow, oc = out.shape
    wh, ww = window_size
    assert mode in ('max', 'avg')
    if mode == 'max':
        fn = np.max
        argfn = np.argmax
    elif mode == 'avg':
        fn = np.mean
        argfn = None

    for i in range(n):

        for ic in range(oc):
            sh = -padding
            for ih in range(oh):
                sw = -padding
                for iw in range(ow):
                    ph0, ph1 = max(sh, 0), min(sh + wh, h)
                    pw0, pw1 = max(sw, 0), min(sw + ww, w)
                    out[i, ih, iw, ic] = fn(x[:, ph0:ph1, pw0:pw1, ic])
                    if argfn:
                        idx = argfn(x[:, ph0:ph1, pw0:pw1, ic])
                        switch[i, ih, iw, ic] = (idx / wh, idx % wh)

                    sw += stride
                sh += stride
    return out


def pooling_2d_backward_bf(chain_grad, out, switch, x, window_size, stride, padding, mode):
    n, h, w, c = x.shape
    _, oh, ow, oc = out.shape
    wh, ww = window_size

    for i in range(n):

        for ic in range(oc):
            sh = -padding
            for ih in range(oh):
                sw = -padding
                for iw in range(ow):
                    chain_grad_i = chain_grad[i, ih, iw, ic]

                    if mode == 'max':
                        x, y = self.switch[i, ih, iw, ic]
                        x.grad[i, x, y, ic] += chain_grad_i
                    elif mode == 'avg':
                        ph0, ph1 = max(sh, 0), min(sh + wh, h)
                        pw0, pw1 = max(sw, 0), min(sw + ww, w)
                        x.grad[:, ph0:ph1, pw0:pw1, ic] += chain_grad_i / (wh * ww)

                    sw += stride
                sh += stride
