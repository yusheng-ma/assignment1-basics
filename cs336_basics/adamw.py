import torch
import math
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 < betas[0] < 1):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0 < betas[1] < 1):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["second_moment"] = torch.zeros_like(p.data)
                t = state["t"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                
                grad = p.grad.data # compute the gradient of the loss at the current time step
                first_moment = beta1 * first_moment + (1 - beta1) * grad # update the first moment estimate
                second_moment = beta2 * second_moment + (1 - beta2) * (grad ** 2) # update the second moment estimate
                lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t) # compute the adjusted lr for iteration t
                p.data -= lr_t * first_moment / ((second_moment ** 0.5) + eps) # update the (training) parameters
                p.data -= lr * weight_decay * p.data # apply weight decay

                state["t"] = t + 1
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment

        return loss