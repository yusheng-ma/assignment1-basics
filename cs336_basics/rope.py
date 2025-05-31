import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: Optional[torch.device] = None
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be an odd number"

        # inverse frequency: constant^(2k/d)
        inv_freq: Float[Tensor, "half_d_k"] = \
            1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # i
        positions: Float[Tensor, "seq_len"] = \
            torch.arange(max_seq_len, device=device).float()
        
        # freqs(angle, or theta): i * constant^(2k/d)
        freqs: Float[Tensor, "seq_len half_d_k"] = \
            einsum(positions, inv_freq, "s, h_d -> s h_d")

        half_sin_buffer: Float[Tensor, "seq_len half_d_k"] = torch.sin(freqs)
        half_cos_buffer: Float[Tensor, "seq_len half_d_k"] = torch.cos(freqs)

        sin_buffer: Float[Tensor, "seq_len d_k"] = self._interleave_half_buffer(half_sin_buffer)
        cos_buffer: Float[Tensor, "seq_len d_k"] = self._interleave_half_buffer(half_cos_buffer)

        self.register_buffer("sin_buffer", sin_buffer, persistent=False)
        self.register_buffer("cos_buffer", cos_buffer, persistent=False)

    def _interleave_half_buffer(self, x: Float[Tensor, "seq_len half_d_k"]) -> Float[Tensor, "seq_len d_k"]:
        # stack_time is we stack x for 2 times
        # (d_k stack_time): first look d_k=0's all stack_time, then look d_k=1's all stack_time
        return rearrange([x, x], "stack_time seq_len d_k -> seq_len (d_k stack_time)") 

    def forward(
            self,
            x: Float[Tensor, " ... sequence_length d_k"],
            token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        x1: Float[Tensor, " ... sequence_length half_d_k"] = x[..., ::2]  # even
        x2: Float[Tensor, " ... sequence_length half_d_k"] = x[..., 1::2]  # odd

        sin: Float[Tensor, "seq_len half_d_k"] = self.sin_buffer[token_positions][..., ::2]
        cos: Float[Tensor, "seq_len half_d_k"] = self.cos_buffer[token_positions][..., ::2]
        
        x_even: Float[Tensor, " ... sequence_length half_d_k"] = \
            einsum(x1, cos, "... s d, s d -> ... s d") - einsum(x2, sin, "... s d, s d -> ... s d")
        x_odd: Float[Tensor, " ... sequence_length half_d_k"] = \
            einsum(x1, sin, "... s d, s d -> ... s d") + einsum(x2, cos, "... s d, s d -> ... s d")

        return rearrange([x_even, x_odd], "stack_time ... seq_len d_k -> ... seq_len (d_k stack_time)")