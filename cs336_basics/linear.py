import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float
from einops import einsum

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight
        weight: Float[Tensor, "out_features in_features"] = torch.empty(
            out_features, in_features, device=device, dtype=dtype
        )
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
