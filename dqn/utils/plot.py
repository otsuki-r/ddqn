import dataclasses
import pathlib
from typing import TypeVar

import matplotlib.pyplot as plt
import torch

T = TypeVar("T")


@dataclasses.dataclass
class PlotConfig:
    plot_id: int
    title: str
    xlabel: str
    ylabel: str


EPISODE_PLOT_CONFIG = PlotConfig(
    plot_id=1,
    title="Episode duration",
    xlabel="Episode number",
    ylabel="Duration",
)
LOSS_PLOT_CONFIG = PlotConfig(
    plot_id=2,
    title="Loss",
    xlabel="Episode number",
    ylabel="Loss",
)
REWARD_PLOT_CONFIG = PlotConfig(
    plot_id=3,
    title="Cumulative reward",
    xlabel="Episode number",
    ylabel="Reward",
)
EPSILON_PLOT_CONFIG = PlotConfig(
    plot_id=4,
    title="Epsilon decay",
    xlabel="Global step number",
    ylabel="Epsilon",
)


def _plot(
    data: list[T], *, path: pathlib.Path, plot_config: PlotConfig
) -> None:
    plt.figure(plot_config.plot_id)
    plt.title(plot_config.title)
    plt.xlabel(plot_config.xlabel)
    plt.ylabel(plot_config.ylabel)
    data_t = torch.tensor(data, dtype=torch.float)
    plt.plot(data_t.numpy())
    plt.savefig(str(path))
    return


def plot_episode_durations(data: list[int], path: pathlib.Path) -> None:
    _plot(data, path=path, plot_config=EPISODE_PLOT_CONFIG)


def plot_losses(data: list[float], path: pathlib.Path) -> None:
    _plot(data, path=path, plot_config=LOSS_PLOT_CONFIG)


def plot_rewards(data: list[float], path: pathlib.Path) -> None:
    _plot(data, path=path, plot_config=REWARD_PLOT_CONFIG)


def plot_epsilons(data: list[float], path: pathlib.Path) -> None:
    _plot(data, path=path, plot_config=EPSILON_PLOT_CONFIG)
