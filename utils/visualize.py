import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

__all__ = ['make_image', 'show_batch']


# functions to show an image
def make_image(img, mean=(0, 0, 0), std=(1, 1, 1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]  # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def gauss(x, a, mu, sigma):
    return torch.exp(-torch.pow(torch.add(x, -mu), 2).div(2 * sigma * sigma)).mul(a)


def colorize(x):
    """Converts a one-channel grayscale image to a color heatmap image. """
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
        return
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[1] = gauss(x, 1, .5, .3)
        cl[2] = gauss(x, 1, .2, .3)
        cl[cl.gt(1)] = 1
        return cl
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:, 0, :, :] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
        cl[:, 1, :, :] = gauss(x, 1, .5, .3)
        cl[:, 2, :, :] = gauss(x, 1, .2, .3)
        return cl


def show_batch(images, mean=(2, 2, 2), std=(0.5, 0.5, 0.5)):
    images = make_image(torchvision.utils.make_grid(images), mean, std)
    plt.imshow(images)
    plt.show()

