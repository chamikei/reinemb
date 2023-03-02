import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class CategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls, n_per):
        super(CategoriesSampler, self).__init__(None)
        self.n_task = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = torch.from_numpy(np.argwhere(label == i).reshape(-1))
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_task

    def __iter__(self):
        for _ in range(self.n_task):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(l.size(0))[:self.n_per]
                batch.append(l[pos])
            batch_idxs = torch.stack(batch).t().reshape(-1)
            yield batch_idxs


class RandomSampler(Sampler):
    def __init__(self, label, n_batch, n_per):
        super(RandomSampler, self).__init__(None)
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


class ClassSampler(Sampler):
    def __init__(self, label, n_per=None):
        super(ClassSampler, self).__init__(None)
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = torch.from_numpy(np.argwhere(label == i).reshape(-1))
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]


class InSetSampler(Sampler):
    def __init__(self, n_batch, n_sbatch, pool):
        super(InSetSampler, self).__init__(None)
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch
