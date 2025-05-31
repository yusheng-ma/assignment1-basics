import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    d_k = Q.shape[-1]

    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k ** 0.5

    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))

    post_softmax = softmax(pre_softmax, -1)

    # keys = values
    return einsum(post_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v")
