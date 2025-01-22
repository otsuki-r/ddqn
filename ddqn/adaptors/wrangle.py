from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def torch_from_array(
    vals: list[T] | np.ndarray, *, dtype=torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.tensor(vals, device=device, dtype=dtype)
