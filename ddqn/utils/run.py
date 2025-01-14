import time
import logging
import pathlib
import random
from typing import NoReturn

import gymnasium
import torch
from ..structures import DoubleDQN

logger = logging.getLogger(__name__)


def run(
    ddqn: DoubleDQN, *, env: gymnasium.Env, weights_dir: pathlib.Path
) -> NoReturn:
    """
    Run the model in the environment `env` using random seeds at each
    iteration. Opens a new window at the start of each iteration and
    closes it upon the episode ending.

    Parameters
    ----------
    ddqn : DoubleDQN
        Double DQN instance
    env : gymnasium.Env
        Environment to run the model in. Render mode should be set
        to human to visualise the output.
    weights_dir : pathlib.Pathlib
        Path to directory holding weights
    """

    logger.debug("Evaluating model at %s", weights_dir)
    ddqn.load_state_dict(weights_dir)
    ddqn.eval()

    iteration = 0
    while True:
        this_seed = random.randint(1, 1_000_000)
        logger.debug("Scene: %s, seed: %s", iteration, this_seed)

        _state, _ = env.reset(seed=this_seed)
        state = torch.tensor(_state, dtype=torch.float32)

        completed = False
        this_duration = 0

        while not completed:
            action = ddqn.pnet(state).max(0).indices.item()
            next_state, _, terminated, truncated, _ = env.step(action)
            state = torch.tensor(next_state)
            completed = terminated or truncated
            this_duration += 1

        logger.debug("This episode duration: %s", this_duration)

        iteration += 1
        time.sleep(1)
