import torch
from typing import Iterable
from jaxtyping import Float
from torch import Tensor

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    # g: could be any shape
    grads: list[Float[Tensor, "..."]] = [p.grad for p in parameters if p.grad is not None]

    # individual L2 norms: list of scalar tensors
    individual_norms: list[Float[Tensor, ""]] = []
    for g in grads:
        norm: Float[Tensor, ""] = torch.norm(g, 2)  # scalar
        individual_norms.append(norm)

    # stacked into 1D tensor of shape (N,)
    stacked_norms: Float[Tensor, "n"] = torch.stack(individual_norms)

    # final scalar total norm
    total_norm: Float[Tensor, ""] = torch.norm(stacked_norms, 2)

    if total_norm > max_l2_norm:
        for g in grads:
            g *= max_l2_norm / (total_norm + eps)  # in-place scale
