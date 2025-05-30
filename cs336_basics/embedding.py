import torch
import torch.nn as nn
from torch import Tensor
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

    def forward(self, token_ids: Float[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        one_hot = torch.nn.functional.one_hot(token_ids, num_classes=self.weight.shape[0])
        
        one_hot = one_hot.to(dtype=self.weight.dtype)

        embedding = einsum(
            one_hot, self.weight,
            "... vocab_size, vocab_size d_model -> ... d_model"
        ) # vocab_size = seq_len
        
        return embedding