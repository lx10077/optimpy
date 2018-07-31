"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import numpy as np
import torch


__all__ = ['mixup_data', 'mixup_criterion', 'make_train_path', 'mkdir']


def mixup_data(x, y, alpha=1.0, use_cuda=False):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda."""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def make_train_path(train_prefix=None):
    # make train dir
    cwd = os.path.dirname(__file__)
    path = os.path.dirname(cwd)
    assert path[-6:] == 'config'

    basename = os.path.basename(cwd)
    if train_prefix is not None:
        base_train_path = os.path.join(train_prefix)
        if not os.path.exists(base_train_path):
            os.makedirs(base_train_path)
        make_soft_link(base_train_path, os.path.join(path[:-6], 'train_log'))

    pre_train_path = os.path.join(path[:-6], 'train_log', basename)
    train_path = os.path.join(cwd, 'train_log')

    if not os.path.exists(pre_train_path):
        os.makedirs(pre_train_path)
    make_soft_link(pre_train_path, train_path)
    return train_path


def make_soft_link(base_path, path):
    if not os.path.exists(path):
        os.system('ln -s {} {}'.format(base_path, path))
    elif os.path.realpath(path) != os.path.realpath(base_path):
        os.system('rm {}'.format(path))
        os.system('ln -s {} {}'.format(base_path, path))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
