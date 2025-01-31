import dataclasses
import logging
import pathlib
import signal
import sys

import enlighten  # type: ignore
import torch

from ..early_stop import EarlyStop
from ..structures import ReplayBuffer, StateChange
from ..watcher import Watcher, WatcherTarget, WatcherType


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StepSummary:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor  # step reward
    next_state: torch.Tensor | None
    epsilon: float
    loss: float
    step_number: int
    episode_number: int
    global_step_number: int
    episode_duration: int | None
    episode_reward: float | None


class StepHandler:
    def __init__(
        self,
        path: pathlib.Path,
        num_episodes: int,
        *,
        buffering_episodes: int = 100,
        watchers: list[Watcher] | None = None,
        early_return_fn: EarlyStop | None = None,
    ) -> None:
        self.buffering_episodes = buffering_episodes
        self.path = path

        self.losses_buffer = []
        self.rewards_buffer = []
        self.epsilons_buffer = []
        self.epsiode_durations_buffer = []

        self.buffer_map = {
            "losses.txt": self.losses_buffer,
            "rewards.txt": self.rewards_buffer,
            "epsilons.txt": self.epsilons_buffer,
            "episode_durations.txt": self.epsiode_durations_buffer,
        }
        for fname in self.buffer_map.keys():
            # Trunacte files
            (self.path / fname).open("w")

        self.episode_watchers = [
            w for w in watchers or [] if w.watcher_type is WatcherType.EPISODES
        ]
        self.step_watchers = [
            w
            for w in watchers or []
            if w.watcher_type is WatcherType.GLOBAL_STEP
        ]

        self.progress_manager = enlighten.get_manager()
        self.pbar = self.progress_manager.counter(
            total=num_episodes,
            desc="Episode num.",
            unit="episodes",
            color="green",
        )

        signal.signal(signal.SIGINT, self.interrupt_handler)

    def step_end(
        self, step_summary: StepSummary, replay_buffer: ReplayBuffer
    ) -> None:
        exp = StateChange(
            state=step_summary.state,
            action=step_summary.action,
            reward=step_summary.reward,
            next_state=step_summary.next_state,
        )
        self.losses_buffer.append(
            (
                step_summary.episode_number,
                step_summary.loss,
            )
        )
        self.rewards_buffer.append(
            (
                step_summary.episode_number,
                step_summary.reward.item(),
            )
        )
        self.epsilons_buffer.append(
            (
                step_summary.episode_number,
                step_summary.epsilon,
            )
        )

        replay_buffer.append(exp)

        for w in self.step_watchers:
            match w.target:
                case WatcherTarget.LOSS:
                    w.append(
                        step_summary.loss,
                        counter=step_summary.global_step_number,
                    )
                case WatcherTarget.REWARD:
                    w.append(
                        step_summary.reward,
                        counter=step_summary.global_step_number,
                    )
                case WatcherTarget.EPSILON:
                    w.append(
                        step_summary.epsilon,
                        counter=step_summary.global_step_number,
                    )
                case _:
                    raise ValueError(
                        "Watcher target %s is unsupported for steps",
                        w.target,
                    )

    def episode_end(self, step_summary: StepSummary) -> None:
        logger.debug(
            "Episode: %s, duration: %s, reward: %s",
            step_summary.episode_number,
            step_summary.step_number,
            step_summary.reward.item(),
        )

        self.pbar.update()

        self.epsiode_durations_buffer.append(
            (
                step_summary.episode_number,
                step_summary.episode_duration,
            )
        )

        for w in self.episode_watchers:
            match w.target:
                case WatcherTarget.EPISODE_DURATIONS:
                    w.append(
                        step_summary.episode_duration,
                        counter=step_summary.episode_number,
                    )
                case WatcherTarget.REWARD:
                    w.append(
                        step_summary.episode_reward,
                        counter=step_summary.episode_number,
                    )
                case _:
                    raise ValueError(
                        "Watcher target %s is unsupported for episodes",
                        w.target,
                    )

        if step_summary.episode_number % self.buffering_episodes == 0:
            self.flush()

        return

    @classmethod
    def _format_tuple(cls, tup: tuple[int, int | float]) -> str:
        return str(tup).strip("()").replace(" ", "")

    def flush(self) -> None:
        with (self.path / "losses.txt").open("a") as f:
            f.write(
                "\n"
                + "\n".join([self._format_tuple(t) for t in self.losses_buffer])
            )
        with (self.path / "rewards.txt").open("a") as f:
            f.write(
                "\n"
                + "\n".join(
                    [self._format_tuple(t) for t in self.rewards_buffer]
                )
            )
        with (self.path / "epsilons.txt").open("a") as f:
            f.write(
                "\n"
                + "\n".join(
                    [self._format_tuple(t) for t in self.epsilons_buffer]
                )
            )
        with (self.path / "episode_durations.txt").open("a") as f:
            f.write(
                "\n"
                + "\n".join(
                    [
                        self._format_tuple(t)
                        for t in self.epsiode_durations_buffer
                    ]
                )
            )

        self.losses_buffer = []
        self.rewards_buffer = []
        self.epsilons_buffer = []
        self.epsiode_durations_buffer = []

    def interrupt_handler(self, sig, frame) -> None:
        logger.warning("Interrupt called. Flushing buffers to file")
        self.flush()
        sys.exit(0)
