import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Bool, Int
from einops import rearrange
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: Optional[RotaryPositionalEmbedding] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"
        
        self.num_heads = num_heads
        self.rope = rope
        if self.rope is not None:
            assert isinstance(self.rope, RotaryPositionalEmbedding)

        d_k = d_model // num_heads
        d_v = d_model // num_heads

        # d_in = d_model
        self.qkv_linear = Linear(d_model, 3 * num_heads * d_k, device=device, dtype=dtype)
        self.o_linear = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: Float[Tensor, "... sequence_length d_in"],
            token_positions: Int[Tensor, "... sequence_length"] | None = None
    ) -> Float[Tensor, "... sequence_length d_out"]:
        
        sequence_length = x.shape[-2]

        qkv_x: Float[Tensor, "... sequence_length 3_num_head_d_k"] = self.qkv_linear(x)

        # rearrange
        qkv_x = rearrange(qkv_x, "... sequence_length (three num_head d_k) -> ... three num_head sequence_length d_k", three=3, num_head=self.num_heads)
        q_x, k_x, v_x = qkv_x.unbind(dim=-4) # : Float[Tensor, "... num_head sequence_length d_k"]

        # rope
        if self.rope is not None:
            assert token_positions is not None, "token_positions must be provided when RoPE is used"
            q_x = self.rope(q_x, token_positions)
            k_x = self.rope(k_x, token_positions)

        # create mask
        mask: Bool[Tensor, "sequence_length sequence_length"] = ~torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device), diagonal=1
        ).bool()

        # d_v not d_in: d_v = d_in/num_head
        out: Float[Tensor, "... num_head sequence_length d_v"] = scaled_dot_product_attention(q_x, k_x, v_x, mask)
        
        # notice the order
        # out2 = rearrange(out, "... num_head sequence_length d_in -> ... sequence_length (d_in num_head)", num_head=self.num_heads)
        out2 = rearrange(out, "... num_head sequence_length d_v -> ... sequence_length (num_head d_v)", num_head=self.num_heads)

        # although o_proj is square, do not mess up d_model with num_heads*d_v
        # result = einsum(self.o_proj_weight, out2, "d_model d_v, ... sequence_length d_model -> ... sequence_length d_v")
        # result = einsum(self.o_proj_weight, out2, "d_model num_heads_d_v, ... sequence_length num_heads_d_v -> ... sequence_length d_model")
        result: Float[Tensor, "... sequence_length d_out"] = self.o_linear(out2)
        return result