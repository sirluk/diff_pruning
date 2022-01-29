from functools import partial
import torch
from  torch import nn
from torch.nn.parameter import Parameter
import torch.nn.utils.parametrize as parametrize

from typing import Callable

from utils import concrete_stretched, concrete_stretched_deterministic


class DiffWeight(nn.Module):
    
    def __init__(self, weight, alpha_init, concrete_lower, concrete_upper, structured):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured
        
        weight.requires_grad = False
        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))
        
        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))
            
        self.active = True
                
    def forward(self, X):
        if not self.active: return X
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.z * (self.finetune - X)
    
    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z
    
    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, x) -> torch.Tensor:
        if self.training:
            return concrete_stretched(x, l=self.concrete_lower, r=self.concrete_upper)
        else:
            return concrete_stretched_deterministic(x, l=self.concrete_lower, r=self.concrete_upper)
            
    def set_mode(self, train: bool) -> None:
        if train:
            self.train()
        else:
            self.eval()
        for p in self.parameters():
            p.requires_grad = train
            
            
class DiffWeightFixmask(nn.Module):
    
    def __init__(self, pre_trained: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("pre_trained", Parameter(pre_trained, requires_grad=False))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
    
    def forward(self, X):
        return self.pre_trained + self.mask * X     