import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_episode_durations(path: pathlib.Path) -> None:
    _, data = np.genfromtxt(path, delimiter=",").T
    plt.figure(0)
    plt.title("Epsiode Durations")
    plt.xlabel("Episode number")
    plt.ylabel("Duration / steps")
    plt.plot(data)
    plt.savefig(str(path.with_suffix(".png")))


def plot_episode_rewards(path: pathlib.Path) -> None:
    _, data = np.genfromtxt(path, delimiter=",").T
    plt.figure(1)
    plt.title("Epsiode Rewards")
    plt.xlabel("Episode number")
    plt.ylabel("Reward")
    plt.plot(data)
    plt.savefig(str(path.with_suffix(".png")))


def plot_epsilons(path: pathlib.Path) -> None:
    _, data = np.genfromtxt(path, delimiter=",").T
    plt.figure(2)
    plt.title("Epsilon decay")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.plot(data)
    plt.savefig(str(path.with_suffix(".png")))


def plot_losses(path: pathlib.Path) -> None:
    data = np.genfromtxt(path, delimiter=",")
    episode_nums, vals = data.T
    episode_vals = np.split(
        data[:, 1], np.unique(data[:, 0], return_index=True)[1][1:]
    )
    episode_means = [
        arr[~np.isnan(arr)].mean() if arr[~np.isnan(arr)].size != 0 else None
        for arr in episode_vals
    ]
    plt.figure(3)
    plt.title("Mean loss")
    plt.xlabel("Epsiode number")
    plt.ylabel("Loss")
    plt.plot(episode_means)
    plt.xlim([0, episode_nums.max()])
    plt.savefig(str(path.with_suffix(".png")))
