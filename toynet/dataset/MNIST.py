import os
import struct

import numpy as np

from .dataset import Dataset


class MNIST(Dataset):

    train_img_fname = 'train-images.idx3-ubyte'
    train_lbl_fname = 'train-labels.idx1-ubyte'

    test_img_fname = 't10k-images.idx3-ubyte'
    test_lbl_fname = 't10k-labels.idx1-ubyte'

    def __init__(self, path='.', mode='train'):
        self.path = path
        self.mode = mode
        self.size = 0

        if mode == 'train':
            img_fname = self.train_img_fname
            lbl_fname = self.train_lbl_fname
        elif mode == 'test':
            img_fname = self.test_img_fname
            lbl_fname = self.test_lbl_fname

        self.img_fpath = os.path.join(path, img_fname)
        self.lbl_fpath = os.path.join(path, lbl_fname)

    def __iter__(self):
        path_img, path_lbl = self.img_fpath, self.lbl_fpath

        file_lbl = open(path_lbl, 'rb')
        magic, lb_size = struct.unpack(">II", file_lbl.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        file_img = open(path_img, 'rb')
        magic, im_size, rows, cols = struct.unpack(">IIII", file_img.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        if lb_size != im_size:
            raise ValueError('image size is not equal to label size')

        for i in range(im_size):
            label = np.frombuffer(file_lbl.read(1), dtype=np.uint8)
            label = np.eye(10)[label].reshape((10,))
            img = np.frombuffer(file_img.read(rows * cols), dtype=np.uint8).reshape(rows, cols, 1)
            yield {'img': img, 'label': label}

        file_img.close()
        file_lbl.close()


if __name__ == '__main__':
    import cv2
    dataset = MNIST('/Users/zhy/Downloads')
    for item in dataset:
        img, label = item['img'], item['label']
        cv2.imshow(str(label), img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
