import dataclasses
import time
import logging
import numpy as np
import random

import torch
from ddqn.configure import EnvConfig, RunConfig

from ..structures import DoubleDQN
from .gif import save_frames_as_gif
from ..adaptors import make_torch_from_int

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunResult:
    total_reward: float | int
    epsiode_duration: int


def run(
    ddqn: DoubleDQN, env_config: EnvConfig, run_config: RunConfig
) -> RunResult:
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
    last_reward = -1
    last_epsiode_duration = -1
    while iteration != run_config.num_iterations:
        this_seed = random.randint(1, 1_000_000)

        _state, _ = env.reset(seed=this_seed)
        state = env_config.state_space_adaptor(
            _state, dtype=torch.float32, device=torch.device("cpu")
        )
        # state = torch.tensor(_state, dtype=torch.float32)
        logger.debug(
            "Scene: %s, seed: %s, initial_state: %s",
            iteration,
            this_seed,
            _state,
        )

        completed = False
        this_duration = 0
        this_frames: list[np.ndarray] = []
        total_reward = 0.0

        while not completed:
            if run_config.save_path:
                this_frames.append(env.render())  # type:ignore
            with torch.no_grad():
                action = ddqn.pnet(state).max(0).indices.item()
            _next_state, reward, terminated, truncated, _ = env.step(action)
            state = env_config.state_space_adaptor(
                _next_state, dtype=torch.float32, device=torch.device("cpu")
            )
            completed = terminated or truncated
            this_duration += 1
            total_reward += reward

        logger.debug("This episode duration: %s", this_duration)

        if run_config.save_path:
            # Only run one iteration if saving
            save_frames_as_gif(this_frames, run_config.save_path)
            return

        if run_config.num_iterations is None:
            # Run indefinitely if `num_iterations` is None
            iteration = 0
            time.sleep(1)
        else:
            iteration += 1
            last_epsiode_duration = this_duration
            last_reward = total_reward

    return RunResult(
        total_reward=last_reward, epsiode_duration=last_epsiode_duration
    )
