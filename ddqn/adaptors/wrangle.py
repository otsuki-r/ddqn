from typing import Callable, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def torch_from_array(
    vals: list[T] | np.ndarray, *, device: torch.device, dtype=torch.dtype
) -> torch.Tensor:
    return torch.tensor(vals, device=device, dtype=dtype)


def make_torch_from_int(dim_state_space: int) -> Callable[[...], torch.Tensor]:
    def torch_from_int(
        val: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        z = torch.zeros(dim_state_space, device=device, dtype=dtype)
        z[val] = 1
        return z

    return torch_from_int
