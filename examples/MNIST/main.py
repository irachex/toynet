import numpy as np

import toynet as tn


def make_network(batch_size=1):
    num_classes = 10

    img = tn.nn.Input(name='img', shape=(batch_size, 28, 28, 1), dtype='uint8')
    label = tn.nn.Input(name='label', shape=(batch_size, num_classes), dtype='uint8')

    x = (img - 128.) / 128.

    # for layer in [20, 20]:
    #     w = tn.nn.Param()
    #     b = tn.nn.Param()
    #     z = w @ data + b
    #     a = tn.nn.ReLU()()
    #     x = a

    x = tn.nn.Conv2D(out_channels=8, kernel_size=5, padding=2)(x)
    x = tn.nn.ReLU()(x)
    x = tn.nn.MaxPooling2D(window_size=2, stride=2)(x)

    x = tn.nn.Conv2D(out_channels=16, kernel_size=5, padding=2)(x)
    x = tn.nn.ReLU()(x)
    x = tn.nn.MaxPooling2D(window_size=2, stride=2)(x)

    x = tn.nn.Reshape((x.shape[0], -1))(x)
    x = tn.nn.FullyConnected(num_classes)(x)
    x = tn.nn.Reshape((x.shape[0], -1))(x)

    pred = tn.nn.Softmax()(x)
    loss = ((pred - label) ** 2).mean()

    net = tn.nn.Network(outputs=[pred], loss=loss)
    return net


def train(batch_size=32, epoch_size=100):
    dataset = tn.dataset.BatchDataset(
        tn.dataset.MNIST('/Users/zhy/Downloads/'),
        batch_size=batch_size)

    net = make_network(batch_size=batch_size)
    net.init_params()

    lr = 1e-3
    optimizer = tn.optim.SGD(net.params(), lr=lr)

    for epoch in range(epoch_size):
        for bi, batch in enumerate(dataset):
            pred, loss = net.fprop(inputs=batch)
            acc = np.sum(np.argmax(pred, axis=-1) == np.argmax(batch['label'], axis=-1)) / batch_size

            print('epoch={} batch={} loss={} acc={}'.format(
                epoch, bi, loss, acc))

            net.bprop()
            optimizer.step()

    # tn.io.dump(net, path)


def evaluate():
    net = tn.io.load(path)
    data, label = None # load test data
    pred = net.forward(data)
    acc = tn.eval.accuracy(pred, label)
    print('acc:', acc)


def main():
    train()
    # evaluate()


if __name__ == '__main__':
    main()
