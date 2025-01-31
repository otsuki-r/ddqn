from typing import Callable

import torch
import torch.nn as nn

ActionSelector = Callable[[float, torch.Tensor, nn.Module], torch.Tensor]

__all__ = ["ActionSelector"]
