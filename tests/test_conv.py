import numpy as np
import toynet as tn


x = np.array([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1]
], dtype=np.float32).reshape(1, 5, 5, 1)

kernel = np.array([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2],
], dtype=np.float32).reshape(3, 3, 1, 1)

# test case come from https://arxiv.org/pdf/1603.07285.pdf

conv = tn.nn.Conv2D(1, kernel.shape[0], stride=2, padding=1)(x)
conv.kernel.value = kernel
out = conv.forward()
print(out)
assert np.array_equal(
    out,
    np.array([
        [ 6., 17., 3.],
        [ 8., 17., 13],
        [ 6., 4., 4.]], dtype=np.float32).reshape(1, 3, 3, 1)
    )


conv = tn.nn.Conv2D(1, kernel.shape[0], stride=1, padding=0)(x)
conv.kernel.value = kernel
out = conv.forward()
print(out)
assert np.array_equal(
    out,
    np.array([
        [12., 12., 17.],
        [10., 17., 19.],
        [9., 6., 14.]], dtype=np.float32).reshape(1, 3, 3, 1)
    )
