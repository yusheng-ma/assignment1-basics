import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    
    x = np.stack([dataset[s:s+context_length] for s in starts])
    y = np.stack([dataset[s+1:s+context_length+1] for s in starts])
    
    x_tensor = torch.tensor(x, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    return x_tensor, y_tensor
