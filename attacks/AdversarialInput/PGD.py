"""
PGD: Projected Gradient Descent
"""

import torch
from torch import nn, Tensor
from typing import Callable, List
import numpy as np
from attacks.utils import *
from .AdversarialInputBase import AdversarialInputAttacker
from .L1aux import normalize_by_pnorm

l1_sparsity = 0.95


class PGD(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = True,
        step_size: float = 16 / 255 / 10,
        eot_step: int = 1,
        eot_batch_size: int = 1024,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.eot_step = eot_step
        self.eot_batch_size = eot_batch_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        super(PGD, self).__init__(model, *args, **kwargs)
        self.verbose = verbose

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x: Tensor, y: Tensor):
        assert len(x.shape) == 4, "input should have size B, C, H, D"
        B, C, H, D = x.shape
        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            eot_xs = x.repeat(self.eot_step, 1, 1, 1).split(self.eot_batch_size * B, dim=0)
            for eot_x in eot_xs:
                logit = 0
                for model in self.models:
                    logit += model(eot_x.to(model.device)).to(x.device)
                loss = self.criterion(logit, y.repeat(eot_x.shape[0] // y.shape[0], *[1] * (len(y.shape) - 1)))
                loss.backward()
            grad = x.grad / self.eot_step
            if self.verbose:
                print(f"loss={round(loss.item(), 5)}, grad_magnitude={round(grad.abs().mean().item(), 5)}")
            x.requires_grad = False
            if self.norm == "L2" or self.norm == "Linf":
                grad = grad.sign()
            elif self.norm == "L1":
                abs_grad = torch.abs(grad)

                batch_size = grad.size(0)
                view = abs_grad.view(batch_size, -1)
                view_size = view.size(1)

                # vals, idx = view.topk(1)
                vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))

                out = torch.zeros_like(view).scatter_(1, idx, vals)
                out = out.view_as(grad)
                grad = grad.sign() * (out > 0).float()
                grad = normalize_by_pnorm(grad, p=1)
            # update
            if self.targerted_attack:
                x = x - self.step_size * grad
            else:
                x = x + self.step_size * grad
            x = self.clamp(x, original_x)
        print(f"loss={round(loss.item(), 5)}, grad_magnitude={round(grad.abs().mean().item(), 5)}")
        return x
