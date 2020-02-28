import numpy as np

from .param import Input, Const, Param


class Network:

    def __init__(self, outputs=None, loss=None):
        self.outputs = outputs or []
        self.loss = loss

    def params(self):
        return get_depend_params(self.loss)

    def init_params(self, r=1e-3):
        for p in self.params():
            p.value = np.random.randn(*p.shape) * r

    def fprop(self, *targets, inputs=None):
        targets = targets or (self.outputs + [self.loss])
        ret = forward_propagation(targets, inputs=inputs)
        return ret if len(targets) > 1 else ret[0]

    def bprop(self):
        return backward_propagation(self.loss)


def get_depend_nodes(targets):
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    nodes = set(targets)
    q = targets[:]
    head, tail = 0, len(q)
    while head < tail:
        h = q.pop()
        for x in h.in_edges:
            if x not in nodes:
                nodes.add(x)
                q.append(x)
                tail += 1
        head += 1
    return nodes


def get_depend_params(targets):
    nodes = get_depend_nodes(targets)
    return [x for x in nodes if isinstance(x, Param)]


def forward_propagation(targets, inputs=None):
    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    nodes = get_depend_nodes(targets)

    q = [x for x in nodes if len(x.in_edges) == 0]
    in_degrees = {x: len(x.in_edges) for x in nodes}
    head, tail = 0, len(q)
    while head < tail:
        h = q[head]
        for x in h.out_edges:
            in_degrees[x] -= 1
            if in_degrees[x] == 0:
                q.append(x)
                tail += 1
        head += 1

    inputs = inputs or {}
    for x in q:
        if isinstance(x, Input):
            assert x in inputs or x.name in inputs, 'missing input {}'.format(x.name)
            x.value = inputs.get(x) or inputs.get(x.name)
        elif isinstance(x, (Const, Param)):
            pass
        else:
            x.value = x.forward()
        x.grad = np.zeros_like(x.value, dtype=np.float32)

    return [x.value for x in targets]


def backward_propagation(target):
    nodes = get_depend_nodes(target)

    q = [target]
    target.grad = 1
    degrees = {x: len(x.out_edges) for x in nodes}
    head, tail = 0, len(q)
    while head < tail:
        h = q[head]
        h.backward()
        for x in h.in_edges:
            degrees[x] -= 1
            if degrees[x] == 0:
                q.append(x)
                tail += 1
        head += 1
    return
