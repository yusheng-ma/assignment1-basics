import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.eps = eps

        # Initialize weight
        weight: Float[Tensor, "... d_model"] = torch.ones(
            d_model, device=device, dtype=dtype
        )

        self.weight = nn.Parameter(weight)

    def forward(
            self,
            x: Float[Tensor, "batch_size sequence_length d_model"]
    ) -> Float[Tensor, "batch_size sequence_length d_model"]:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # rms
        x_squared = x ** 2
        mean_squared: Float[Tensor, "b s 1"] = x_squared.mean(dim=-1, keepdim=True)
        rms: Float[Tensor, "b s 1"] = torch.sqrt(mean_squared + self.eps)

        # rmsnorm
        x_norm: Float[Tensor, "b s d"] = x / rms
        result: Float[Tensor, "b s d"] = einsum(
            x_norm, self.weight, "b s d, d -> b s d"
        )

        return result.to(in_dtype)