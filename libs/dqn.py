from typing import TypedDict
from typing_extensions import Self, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as fn


class DQNParams(TypedDict):
    n_observations: int
    n_actions: int


class DQN(nn.Module):
    def __init__(self, **kwargs: Unpack[DQNParams]) -> None:
        super().__init__()
        self.l1 = nn.Linear(kwargs["n_observations"], 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, kwargs["n_actions"])

    def forward(self, x: torch.Tensor) -> None:
        x = fn.relu(self.l1(x))
        x = fn.relu(self.l2(x))
        return self.l3(x)


class DoubleDQN:
    def __init__(self, **kwargs: Unpack[DQNParams]) -> None:
        self.pnet = DQN(**kwargs)  # policy network
        self.tnet = DQN(**kwargs)  # target network

        # Ensure initialised to same state
        self.tnet.load_state_dict(self.pnet.state_dict())

    def to(self, device: torch.device) -> Self:
        self.pnet = self.pnet.to(device)
        self.tnet = self.tnet.to(device)
        return self
