import time
import logging
import numpy as np
import random

import torch
from ddqn.configure import EnvConfig, RunConfig

from ..structures import DoubleDQN
from .gif import save_frames_as_gif

logger = logging.getLogger(__name__)


def run(ddqn: DoubleDQN, env_config: EnvConfig, run_config: RunConfig) -> None:
    """
    Run the model in the environment `env` using random seeds at each
    iteration. Opens a new window at the start of each iteration and
    closes it upon the episode ending.

    Parameters
    ----------
    ddqn : DoubleDQN
        Double DQN instance
    env_config : EnvConfig
        Enironemtn configuration
    run_config : RunConfig
        Configuration of how to run the model
    """

    env = env_config.env

    logger.debug("Evaluating model at %s", str(run_config.weights_dir))
    ddqn.load_state_dict(run_config.weights_dir)
    ddqn.eval()

    iteration = 0
    while True:
        this_seed = random.randint(1, 1_000_000)

        _state, _ = env.reset(seed=this_seed)
        state = torch.tensor(_state, dtype=torch.float32)
        logger.debug(
            "Scene: %s, seed: %s, initial_state: %s",
            iteration,
            this_seed,
            _state,
        )

        completed = False
        this_duration = 0
        this_frames: list[np.ndarray] = []

        while not completed:
            if run_config.save_path:
                this_frames.append(env.render())  # type:ignore
            action = ddqn.pnet(state).max(0).indices.item()
            next_state, _, terminated, truncated, _ = env.step(action)
            state = torch.tensor(next_state)
            completed = terminated or truncated
            this_duration += 1

        logger.debug("This episode duration: %s", this_duration)

        if run_config.save_path:
            # Only run one iteration if saving
            save_frames_as_gif(this_frames, run_config.save_path)
            return

        iteration += 1
        time.sleep(1)
    return
