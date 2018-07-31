import os
import torch
import torchvision
import torchvision.transforms as transforms
use_cuda = torch.cuda.is_available()


__all__ = ['prepare_cifar10', 'get_flat_grad_from', 'get_flat_para_from']


def get_project_dirpath():
    path = os.path.dirname(__file__)
    while path[-7:] != 'optimpy':
        path = os.path.dirname(path)
    return os.path.dirname(path)


def prepare_cifar10(batch_size, rtclass=False):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(get_project_dirpath(), 'data'),
                                            train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(get_project_dirpath(), 'data'),
                                           train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return (trainloader, testloader) if not rtclass else (trainloader, testloader, classes)


def get_flat_grad_from(model_params, grad_grad=False):
    grads = []
    for param in model_params:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(torch.zeros(param.data.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def get_flat_para_from(model_params):
    params = []
    for param in model_params:
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params
