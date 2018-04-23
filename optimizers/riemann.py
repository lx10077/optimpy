from torch.optim import SGD


class RiemannSGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(RiemannSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):


        loss = super(RiemannSGD, self).step(closure)
        pass
