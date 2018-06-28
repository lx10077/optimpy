from utils import *
from torch.optim.optimizer import Optimizer, required
import math


class RiemannSGD(Optimizer):
    def __init__(self, params, lr_r=required, lr_w=required, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr_r=lr_r, lr_w=lr_w, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(RiemannSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RiemannSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            old_r = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                old_r += p.data.norm() ** 2
            old_r = math.sqrt(old_r)
            norm_w = 0
            d_r = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                w = p.data.div(old_r)
                d_r += (w * d_p).sum()
                w.add_(-group['lr_w'], old_r * d_p)
                norm_w += w.norm() ** 2
                p.data = w

            r = old_r - group['lr_r'] * d_r
            norm_w = math.sqrt(norm_w)

            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(r/norm_w)

        return loss
