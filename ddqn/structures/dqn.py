import pathlib
from typing import TypedDict
from typing_extensions import Self, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as fn

from .replay_memory import DoubleReplayMemory, StateChange


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
    def __init__(
        self,
        memory_size: int,
        warmup_episodes_upper: int,
        main_episodes_lower: int,
        **kwargs: Unpack[DQNParams],
    ) -> None:
        """
        Initialise the DDQN.

        Parameters
        ----------
        memory_size : int
            Capacity of replay memory.
        warmup_episodes_upper : int
            Warmup memory stopping parameter.
        main_episodes_lower : int
            Main memory starting parameter.
        """

        self.replay_memory: DoubleReplayMemory[StateChange] = (
            DoubleReplayMemory(
                memory_size,
                warmup_episodes_upper=warmup_episodes_upper,
                main_episodes_lower=main_episodes_lower,
            )
        )

        self.pnet = DQN(**kwargs)  # policy network for selecting action
        self.tnet = DQN(**kwargs)  # target network for evaluating action

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
