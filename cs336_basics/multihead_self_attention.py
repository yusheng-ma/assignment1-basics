import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Bool
from einops import einsum, rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"
        
        self.num_heads = num_heads

        d_k = d_model // num_heads
        d_v = d_model // num_heads

        # d_in = d_model
        self.q_proj_weight: Float[Tensor, "num_heads*d_k d_in"] = nn.Parameter(torch.empty(d_k * num_heads, d_model, device=device, dtype=dtype))
        self.k_proj_weight: Float[Tensor, "num_heads*d_k d_in"] = nn.Parameter(torch.empty(d_k * num_heads, d_model, device=device, dtype=dtype))
        self.v_proj_weight: Float[Tensor, "num_heads*d_v d_in"] = nn.Parameter(torch.empty(d_v * num_heads, d_model, device=device, dtype=dtype))
        self.o_proj_weight: Float[Tensor, "d_model num_heads*d_v"] = nn.Parameter(torch.empty(d_model, d_v * num_heads, device=device, dtype=dtype))

    def forward(
            self,
            x: Float[Tensor, "... sequence_length d_in"]
    ) -> Float[Tensor, "... sequence_length d_out"]:
        
        sequence_length = x.shape[-2]
        # rearrange
        q_rearranged = rearrange(self.q_proj_weight, "(num_head d_k) d_in -> num_head d_k d_in", num_head=self.num_heads)
        k_rearranged = rearrange(self.k_proj_weight, "(num_head d_k) d_in -> num_head d_k d_in", num_head=self.num_heads)
        v_rearranged = rearrange(self.v_proj_weight, "(num_head d_v) d_in -> num_head d_v d_in", num_head=self.num_heads)

        q_x = einsum(q_rearranged, x, "num_head d_k d_in, ... sequence_length d_in -> ... num_head sequence_length d_k")
        k_x = einsum(k_rearranged, x, "num_head d_k d_in, ... sequence_length d_in -> ... num_head sequence_length d_k")
        v_x = einsum(v_rearranged, x, "num_head d_v d_in, ... sequence_length d_in -> ... num_head sequence_length d_v")

        # create mask
        mask: Bool[Tensor, "sequence_length sequence_length"] = ~torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()

        # d_v not d_in: d_v = d_in/num_head
        out: Float[Tensor, "... num_head sequence_length d_v"] = scaled_dot_product_attention(q_x, k_x, v_x, mask)
        
        # notice the order
        # out2 = rearrange(out, "... num_head sequence_length d_in -> ... sequence_length (d_in num_head)", num_head=self.num_heads)
        out2 = rearrange(out, "... num_head sequence_length d_v -> ... sequence_length (num_head d_v)", num_head=self.num_heads)

        # although o_proj is square, do not mess up d_model with num_heads*d_v
        # result = einsum(self.o_proj_weight, out2, "d_model d_v, ... sequence_length d_model -> ... sequence_length d_v")
        result = einsum(self.o_proj_weight, out2, "d_model num_heads_d_v, ... sequence_length num_heads_d_v -> ... sequence_length d_model")
        return result