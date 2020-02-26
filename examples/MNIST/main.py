import toynet as tn


def make_network():
    data = tn.Input()
    label = tn.Input()

    x = data
    for layer in [20, 20]:
        w = tn.Param()
        b = tn.Param()
        z = w @ data + b
        a = tn.ReLU(z)
        x = a

    x = tn.softmax(x)

    loss = tn.cross_entropy(x, label)

    net = Network(outputs=[x], loss=loss)
    return net


def train():
    net = make_network()
    tn.io.dump(net, path)


def evaluate():
    net = tn.io.load(path)
    data, label = None # load test data
    pred = net.forward(data)
    acc = tn.eval.accuracy(pred, label)
    print('acc:', acc)


def main():
    train()
    evaluate()


if __init__ == '__main__':
    main()
