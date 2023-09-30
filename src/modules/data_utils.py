from math import ceil

import torch


def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch sampling
    and adaptive augmentation on GPU."""

    def __init__(
        self,
        x,
        y,
        batch_size=500,
        shuffle=False,
    ):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.device = x.device
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data, device=self.device)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        x, y = self.x[start:end], self.y[start:end]
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    def attach(self):
        self._detach = False
        return self

    def detach(self):
        self._detach = True
        return self

    @property
    def dataset(self):
        return DatasetDummy(self.n_data)


class DatasetDummy:
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return int(self.N)
