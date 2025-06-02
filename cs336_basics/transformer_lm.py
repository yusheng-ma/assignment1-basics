import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.softmax import softmax
from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_context_length, rope_theta)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(
            self,
            x: Int[Tensor, " batch_size sequence_length"],
            token_positions: Int[Tensor, "batch sequence_length"] | None = None
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        
        token_embeded = self.token_embeddings(x)

        transformed = token_embeded
        for layer in self.layers:
            transformed = layer(transformed, token_positions)

        ln_finaled = self.ln_final(transformed)

        lm_headed = self.lm_head(ln_finaled)

        return lm_headed