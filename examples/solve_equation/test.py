import numpy as np
import toynet as tn



def main():
    x = tn.nn.Param(shape=(1, 4), dtype=np.float32)

    b = np.array([4, 6, 8, 10], dtype=np.float32).reshape(1, -1)
    c = np.array([4, 9, 16, 25], dtype=np.float32).reshape(1, -1)

    bi = tn.nn.Input(name='b', shape=b.shape, dtype=np.float32)
    ci = tn.nn.Input(name='c', shape=c.shape, dtype=np.float32)

    # x.value = np.array([0.], )
    loss = (x ** 2 - x * bi + ci).sum()
    # loss = (x ** 2 - x * b + c).sum()
    loss_val = loss.eval(b=b, c=c)
    print(loss_val)
    # assert float(loss_val) == 25.

    net = tn.nn.Network(loss)

    lr = 1e-3
    optimizer = tn.optim.SGD(net.params(), lr=lr)
    # optimizer = tn.optim.Momentum(loss, lr=lr)
    # optimizer = tn.optim.Adam(loss, lr=lr)

    for i in range(10000):
        optimizer.zero_grad()
        loss_val = net.fprop(inputs={'b': b, 'c': c})
        net.bprop()
        optimizer.step()
        if i % 500 == 0 or True:
            print('iter:', i, 'loss:', loss_val, 'x:', x.value, 'x.grad:', x.grad)
        # if i > 10:
        #     break
        if loss_val < 1e-5 or np.isnan(loss_val):
            break
    print('iter:', i, 'loss:', loss_val, 'x:', x.value)



if __name__ == '__main__':
    main()


"""
-----------------------------
Input, Tensor, Const, Param
-----------------------------
Add, MatMul, ReLU, ...
-----------------------------
Linear, : weights



"""