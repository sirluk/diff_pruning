import torch
from torch.optim.lr_scheduler import LambdaLR


def concrete_stretched(alpha, l=-1.5, r=1.5):
    # l = gamma, r = zeta
    u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
    u_term = u.log() - (1-u).log()
    s = (torch.sigmoid(u_term + alpha))
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z


def concrete_stretched_deterministic(alpha, l=-1.5, r=1.5):
    s_stretched = torch.sigmoid(alpha)*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z


def dict_to_device(d, device):
    return {k:v.to(device) for k,v in d.items()}
