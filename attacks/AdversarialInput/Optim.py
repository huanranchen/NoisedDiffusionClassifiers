'''
PGD: Projected Gradient Descent
'''

import torch
from torch import nn, Tensor
from typing import Callable, List
from attacks.utils import *
from .AdversarialInputBase import AdversarialInputAttacker
from torch.optim import Optimizer, Adam


class OptimizerAttacker(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 optimizer: Callable = lambda x: Adam([x], lr=0.1),
                 total_step: int = 10,
                 random_start: bool = False,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.criterion = criterion
        self.optimizer = optimizer
        self.total_step = total_step
        super(OptimizerAttacker, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        x.requires_grad_()
        optimizer = self.optimizer(x)
        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.clamp(x, original_x)
        x.requires_grad_(False)
        return x

    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.norm == "Linf":
            torch.clamp_(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == "L2":
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x.mul_(0).add_(ori_x + difference)  # x = ori_x + difference
        torch.clamp_(x, min=0, max=1)
        return x
