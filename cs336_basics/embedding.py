import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from typing import Optional
from jaxtyping import Float
from einops import einsum

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
        ):
        super().__init__()

        # Initialize weight
        weight: Float[Tensor, "vocab_size d_model"] = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )

        nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Float[LongTensor, "..."]) -> Float[Tensor, "... d_model"]:
        # In PyTorch, LongTensor is used for index operation
        # token_ids shape: batch_size sequence(of token_id)
        # so return shape: batch_size sequence(of token_id) d_model
        return self.weight[token_ids]