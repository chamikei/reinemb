import os
import shutil
import time
import torch
import numpy as np


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]), device=torch.device('cuda'))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('Using gpu %s : %s' % (x, torch.cuda.get_device_name(0)))


def ensure_path(dir_path: str):
    if os.path.exists(dir_path):
        if str(input('{} exists, remove? ([y]/n)'.format(dir_path))) != 'n':
            shutil.rmtree(dir_path, True)
            os.makedirs(dir_path)
        else:
            raise ValueError('User Terminated.')
    else:
        os.makedirs(dir_path)
    print('Experiment dir: {}'.format(dir_path))


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=-1)
    return (pred == label).float().mean().item()


def euclidean_metric(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -1. * torch.sum((a - b) ** 2, dim=2)
    return logits


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = int((time.time() - self.o) / p)
        if x >= 3600:
            return '{:.1f} h'.format(x / 3600)
        elif x >= 60:
            return '{} m'.format(round(x / 60))
        else:
            return '{} s'.format(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = np.array(data, dtype=np.float32)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n, d) Tensor.
      B:  a (m, d) Tensor.
    Returns: a (n, m) Tensor.
    """
    assert A.dim() == 2
    assert B.dim() == 2
    assert A.size(1) == B.size(1)
    return torch.matmul(A, B.t())


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output
    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print('In', self.__class__.__name__)
            raise RuntimeError(f'Found NAN in output {i} at indices: ', nan_mask.nonzero(), 'where:', out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
