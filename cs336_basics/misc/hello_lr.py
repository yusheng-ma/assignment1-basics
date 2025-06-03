import torch
import math
from collections.abc import Callable
from typing import Optional

# 自訂帶 sqrt 衰減的 SGD optimizer
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

# 執行實驗並印出 loss
def run_experiment(lr):
    print(f"\nLearning rate = {lr}")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    for i in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Iter {i+1:2d}: Loss = {loss.item():.6f}")
        loss.backward()
        opt.step()

# 執行三個 learning rate 的實驗
for lr in [1e1, 1e2, 1e3]:
    run_experiment(lr)
