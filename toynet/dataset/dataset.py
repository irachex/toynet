import collections

import numpy as np


class Dataset:

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError


class BatchDataset(Dataset):

    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = collections.defaultdict(list)
        cnt = 0
        for data in self.dataset:
            for k, v in data.items():
                batch[k].append(v)
            cnt += 1
            if cnt == self.batch_size:
                for k in batch:
                    batch[k] = np.array(batch[k])
                yield batch
                batch = collections.defaultdict(list)
                cnt = 0
        if cnt:
            for k in batch:
                batch[k] = np.array(batch[k])
            yield batch


class EpochDataset(Dataset):

    def __init__(self, dataset, epoch_size=None):
        self.dataset = dataset
        self.epoch_size = epoch_size

    def __iter__(self):
        self.epoch = 0
        while self.epoch < self.epoch_size or self.epoch_size is None:
            for data in self.dataset:
                yield data
            self.epoch += 1

