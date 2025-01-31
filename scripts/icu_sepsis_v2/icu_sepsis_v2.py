"""Quickstart example for the ICU-Sepsis environment."""

import logging
import os
import sys

from ddqn.configure import ddqn_parser, process_cli_args
from ddqn.structures import DoubleDQN, ExperienceReplay
from ddqn.utils import train

from utils import (
    build_env_config,
    build_run_config,
    build_training_config,
    custom_eval,
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout)
logging.getLogger("ddqn").setLevel(logging.INFO)


if __name__ == "__main__":
    cli_args = process_cli_args(ddqn_parser)
    env_config = build_env_config(cli_args)

    replay_buffer = ExperienceReplay(capacity=cli_args.memory_size)
    ddqn = DoubleDQN(
        replay_memory=replay_buffer,
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval:
        run_config = build_run_config(cli_args)
        custom_eval(ddqn, env_config=env_config, run_config=run_config)
    else:
        training_config = build_training_config(ddqn, cli_args, env_config)
        train(
            ddqn,
            replay_buffer,
            env_config=env_config,
            training_config=training_config,
        )
