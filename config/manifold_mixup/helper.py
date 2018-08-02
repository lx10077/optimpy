import os
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['mixup_data', 'mixup_criterion', 'make_train_path', 'mkdir']


# ====================================================================================== #
# Manifold mixup helper
# ====================================================================================== #
def mixup_data(alpha=1.0):
    """Return lambda."""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ====================================================================================== #
# Common helper
# ====================================================================================== #
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


def plot(exp_dir, train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    plt.plot(np.asarray(train_loss_list), label='train_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_loss.png'))
    plt.clf()

    plt.plot(np.asarray(train_acc_list), label='train_acc')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_acc.png'))
    plt.clf()

    plt.plot(np.asarray(test_loss_list), label='test_loss')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_loss.png'))
    plt.clf()

    plt.plot(np.asarray(test_acc_list), label='test_acc')
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_acc.png'))
    plt.clf()
