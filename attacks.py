import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn


# --- White-box attacks ---


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.step_size = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        
        x = X_nat.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            
        for i in range(self.k):
            
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, X_nat - self.epsilon), X_nat + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return x