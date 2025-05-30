import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float
from einops import einsum
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
        ):
        super().__init__()

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        w1x: Float[Tensor, "... d_ff"] = self.w1(x)
        silu_w1x = w1x * torch.sigmoid(w1x)
        
        w3x: Float[Tensor, "... d_ff"] = self.w3(x)

        silu_w1x_w3x = einsum(silu_w1x, w3x, "... d_ff, ... d_ff -> ... d_ff")

        w2_silu_w1x_w3x: Float[Tensor, " ... d_model"] = self.w2(silu_w1x_w3x)

        return w2_silu_w1x_w3x