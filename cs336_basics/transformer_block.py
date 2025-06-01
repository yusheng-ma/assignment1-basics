import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)

        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)

        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
            self,
            x: Float[Tensor, "batch sequence_length d_model"],
            token_positions: Int[Tensor, "batch sequence_length"] | None = None
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        
        ln1ed = self.ln1(x)
        attned = self.attn(ln1ed, token_positions)
        # add1ed = attned + x
        add1ed = x + attned

        ln2ed = self.ln2(add1ed)
        ffned = self.ffn(ln2ed)
        # add2ed = ffned + add1ed
        add2ed = add1ed + ffned

        return add2ed
