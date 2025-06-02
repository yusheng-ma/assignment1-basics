import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    inputs_max: Float[Tensor, "batch_size 1"] = inputs.max(dim=-1, keepdim=True).values
    inputs_stable: Float[Tensor, "batch_size vocab_size"] = inputs - inputs_max

    exp: Float[Tensor, "batch_size vocab_size"] = torch.exp(inputs_stable) # element wise
    sumexp: Float[Tensor, "batch_size"] = torch.sum(exp, dim=-1)
    logsumexp: Float[Tensor, "batch_size"] = torch.log(sumexp)

    target_logits: Float[Tensor, "batch_size"] = inputs_stable[torch.arange(inputs.shape[0]), targets]

    loss: Float[Tensor, "batch_size"] = -target_logits + logsumexp

    return loss.mean()