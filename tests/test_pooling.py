import numpy as np
import toynet as tn


x = np.array([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1]
], dtype=np.float32).reshape(1, 5, 5, 1)

# test case come from https://arxiv.org/pdf/1603.07285.pdf

pool = tn.nn.MaxPooling2D(3, stride=1, padding=0)(x)
out = pool.forward()
print(out)
