from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def torch_from_array(
    vals: list[T] | np.ndarray, *, device: torch.device, dtype=torch.dtype
) -> torch.Tensor:
    return torch.tensor(vals, device=device, dtype=dtype)
