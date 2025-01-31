from dataclasses import dataclass

import torch


@dataclass
class StateChange:
    state: torch.Tensor  # s_t
    action: torch.Tensor  # a_t
    reward: torch.Tensor  # r_{t+1}
    next_state: torch.Tensor | None  # s_{t+1}


@dataclass
class StateChanges:
    states: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    next_states: list[torch.Tensor | None]

    def __init__(self, state_changes: list[StateChange]) -> None:
        self.states = [s.state for s in state_changes]
        self.actions = [s.action for s in state_changes]
        self.rewards = [s.reward for s in state_changes]
        self.next_states = [s.next_state for s in state_changes]
