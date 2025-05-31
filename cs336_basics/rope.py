import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: Optional[torch.device] = None
    ):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k

        sin_buffer = torch.zeros(max_seq_len, d_k, device=device)
        cos_buffer = torch.zeros(max_seq_len, d_k, device=device)

        for i in range(max_seq_len):
            for k in range(1, (d_k // 2) + 1):
                freq_index = k - 1
                exponent = 2 * (k - 1) / d_k
                inv_freq = torch.tensor(1.0 / (theta ** exponent), device=device)
                angle = i * inv_freq

                sin_val = torch.sin(angle)
                cos_val = torch.cos(angle)

                sin_buffer[i, 2 * freq_index] = sin_val
                sin_buffer[i, 2 * freq_index + 1] = sin_val
                cos_buffer[i, 2 * freq_index] = cos_val
                cos_buffer[i, 2 * freq_index + 1] = cos_val

        self.register_buffer("sin_buffer", sin_buffer, persistent=False)
        self.register_buffer("cos_buffer", cos_buffer, persistent=False)

    def forward(
            self,
            x: Float[Tensor, " ... sequence_length d_k"],
            token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        sin: Float[Tensor, "... seq_len d_k"] = self.sin_buffer[token_positions]
        cos: Float[Tensor, "... seq_len d_k"] = self.cos_buffer[token_positions]
        
        x_out = torch.empty_like(x)

        for k in range(1, (self.d_k // 2) + 1):
            freq_index = k - 1

            x1 = x[..., 2 * freq_index]
            x2 = x[..., 2 * freq_index + 1]

            sinsin = sin[..., 2 * freq_index]
            coscos = cos[..., 2 * freq_index]

            x_out[..., 2 * freq_index] = x1 * coscos - x2 * sinsin
            x_out[..., 2 * freq_index + 1] = x1 * sinsin + x2 * coscos

        return x_out

