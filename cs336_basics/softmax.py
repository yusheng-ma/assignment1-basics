import torch
from torch import Tensor
from jaxtyping import Float

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x