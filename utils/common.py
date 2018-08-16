import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
use_cuda = torch.cuda.is_available()


__all__ = ['prepare_dataset', 'get_flat_grad_from', 'get_flat_para_from']


def get_project_dirpath():
    path = os.path.dirname(__file__)
    while path[-7:] != 'optimpy':
        path = os.path.dirname(path)
    return os.path.dirname(path)


def prepare_dataset(batch_size, data_name='cifar10', num_workers=4):
    print('==> Preparing dataset..')
    print('==> Dataset is {}, without validation sets..'.format(data_name))

    data_root = os.path.join(get_project_dirpath(), 'data')
    data_name = str(data_name).strip().lower()

    if data_name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        class_num = 10

    elif data_name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        class_num = 100

    elif data_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform_test)
        class_num = 10

    elif data_name == 'svhn':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # may need double check
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        testset = datasets.SVHN(root=data_root, split='test', download=True, transform=transform_test)
        class_num = 10

    elif data_name == 'stl10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # may need double check
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.STL10(root=data_root, split='train', download=True, transform=transform_train)
        testset = datasets.STL10(root=data_root, split='test', download=True, transform=transform_test)
        class_num = 10

    elif data_name == 'imagenet':
        raise NotImplementedError
    else:
        raise ValueError('   No {} dataset..'.format(data_name))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader, class_num


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
