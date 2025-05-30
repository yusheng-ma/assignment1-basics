import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int

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

    # GPT says Int[LongTensor, "..."] is invalid because LongTensor is factory
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        # In PyTorch, LongTensor is used for index operation
        # token_ids shape: batch_size sequence(of token_id)
        # so return shape: batch_size sequence(of token_id) d_model
        return self.weight[token_ids]