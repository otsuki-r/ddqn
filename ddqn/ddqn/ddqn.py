import pathlib
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
        self.l1 = nn.Linear(kwargs["n_observations"], 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, kwargs["n_actions"])

    def forward(self, x: torch.Tensor) -> None:
        x = fn.relu(self.l1(x))


class ConvolutionDQN(nn.Module):
    def __init__(self, **kwargs: Unpack[DQNParams]) -> None:
        super().__init__()
        # # (210, 160)
        # self.l1 = nn.Conv2d(1, 1, 8, stride=4)
        # # (51, 39)
        # self.l2 = nn.Conv2d(1, 1, 16, stride=2)
        # # (18, 12)
        # self.l3 = nn.Linear(18 * 12, kwargs["n_actions"])

        # (210, 160)
        self.l1 = nn.Conv2d(1, 1, (8, 64), stride=4)
        # (51, 25)
        self.l2 = nn.Conv2d(1, 1, (8, 16), stride=1)
        # (44, 10)
        self.l3 = nn.Linear(44 * 10, kwargs["n_actions"])

    def forward(self, x: torch.Tensor) -> None:
        single = len(x.shape) == 2
        if single:
            # Add in batch_size and num_channels
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            # Add in num_channels
            x = x.unsqueeze(1)

        x = fn.relu(self.l1(x))
        x = fn.relu(self.l2(x))
        # Squeeze out num_channels and flatten H * W
        if single:
            return self.l3(x.squeeze(1).flatten(1)).squeeze(0)
        else:
            return self.l3(x.squeeze(1).flatten(1))


class DoubleDQN:
    def __init__(
        self,
        **kwargs: Unpack[DQNParams],
    ) -> None:
        """
        Initialise the DDQN.

        Parameters
        ----------
        kwargs : Unpack[DQNParams]
            Parameters passed to the two underlying DQN instances.
        """

        self.pnet = ConvolutionDQN(
            **kwargs
        )  # policy network for selecting action
        self.tnet = ConvolutionDQN(
            **kwargs
        )  # target network for evaluating action

        # Ensure initialised to same state
        self.tnet.load_state_dict(self.pnet.state_dict())

    def to(self, device: torch.device) -> Self:
        self.pnet = self.pnet.to(device)
        self.tnet = self.tnet.to(device)
        return self

    def save(self, outdir: pathlib.Path) -> None:
        torch.save(self.pnet.state_dict(), outdir / "policy_net.pt")
        torch.save(self.tnet.state_dict(), outdir / "target_net.pt")

    def load_state_dict(self, outdir: pathlib.Path) -> None:
        self.pnet.load_state_dict(
            torch.load(str(outdir / "policy_net.pt"), weights_only=True)
        )
        self.tnet.load_state_dict(
            torch.load(str(outdir / "target_net.pt"), weights_only=True)
        )

    def eval(self) -> None:
        self.pnet.eval()
        self.tnet.eval()
