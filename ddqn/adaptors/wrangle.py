from typing import TypeVar

import numpy as np
import torch

from ..configure import DEVICE

T = TypeVar("T")


def torch_from_array(
    vals: list[T] | np.ndarray, *, dtype=torch.dtype
) -> torch.Tensor:
    return torch.tensor(vals, device=DEVICE, dtype=dtype)
