"""Defines prior modules."""

import numpy as np
import torch
import torch.nn as nn

# TODO: Write docstring


class PriorModule(nn.Module):
    def __init__(self):
        super(PriorModule, self).__init__()

    def forward(self, net):
        return -self.logp(net)

    def initialise(self, net):
        for name, param in net.named_parameters():
            if param.requires_grad:
                value = self.sample(name, param)
                param.data.copy_(value)

    def logp(self, net):
        raise NotImplementedError

    def sample(self, name, param):
        raise NotImplementedError


class PriorGaussian(PriorModule):
    def __init__(self, sw2, sb2=None):
        '''
        Args:
            sw2: weights variance
            sb2: bias variance   
        '''
        
        super(PriorGaussian, self).__init__()
        self.sw2 = sw2
        self.sb2 = sw2 if sb2 is None else sb2

    def logp(self, net):
        w2 = 0
        for name, w in net.named_parameters():
            sw2 = self.sb2 if 'bias' in name else self.sw2
            w2 += torch.sum(w * w) / sw2
        return - 0.5 * w2

    def sample(self, name, param):
        sw2 = self.sb2 if 'bias' in name else self.sw2
        return torch.randn(param.shape) * np.sqrt(sw2)


class PriorLaplace(PriorModule):
    def __init__(self, scale_w, scale_b=None):
        super(PriorLaplace, self).__init__()
        self.scale_w = scale_w
        self.scale_b = scale_w if scale_b is None else scale_b

    def logp(self, net):
        w_sum = 0
        for name, w in net.named_parameters():
            scale = self.scale_b if 'bias' in name else self.scale_w
            w_sum += torch.sum(w.abs()) / scale
        return - w_sum

    def sample(self, name, param):
        scale = self.scale_b if 'bias' in name else self.scale_w
        distr = torch.distributions.Laplace(0, scale)
        return distr.sample(param.shape)