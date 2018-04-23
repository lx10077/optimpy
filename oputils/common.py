import torch
from torch.autograd import Variable


def get_flat_grad_from(model, grad_grad=False):
    grads = []
    for param in model.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(Variable(torch.zeros(param.data.view(-1).shape)))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def get_flat_para_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return Variable(flat_params)
