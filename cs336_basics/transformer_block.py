import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.swiglu import SwiGLU
from cs336_basics.silu import SiLU

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            use_rmsnorm=True,
            post_norm=False,
            use_rope=True,
            activation="swiglu"
    ):
        super().__init__()
        # --- ablation -------------------
        self.use_rope = use_rope
        self.post_norm = post_norm
        self.use_rmsnorm = use_rmsnorm
        # --------------------------------

        self.ln1 = RMSNorm(d_model)

        d_k = d_model // num_heads
        if self.use_rope:
            rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
            self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)
        else:
            self.attn = CausalMultiHeadSelfAttention(d_model, num_heads)

        self.ln2 = RMSNorm(d_model)

        if activation == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        elif activation == "silu":
            self.ffn = SiLU(d_model, d_ff)

    def forward(
            self,
            x: Float[Tensor, "batch sequence_length d_model"],
            token_positions: Int[Tensor, "batch sequence_length"] | None = None
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        if not self.use_rmsnorm:
            z = x + self.attn(x, token_positions)
            y = z + self.ffn(z)
            return y

        elif self.post_norm:
            z = self.ln1(x + self.attn(x, token_positions))
            y = self.ln2(z + self.ffn(z))

            return y
        else:
            ln1ed = self.ln1(x)
            attned = self.attn(ln1ed, token_positions)
            # add1ed = attned + x
            add1ed = x + attned

            ln2ed = self.ln2(add1ed)
            ffned = self.ffn(ln2ed)
            # add2ed = ffned + add1ed
            add2ed = add1ed + ffned

            return add2ed

       
