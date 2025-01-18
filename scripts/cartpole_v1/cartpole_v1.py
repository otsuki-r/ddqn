import logging
import os
import sys

from ddqn.configure import process_cli_args
from ddqn.structures import DoubleDQN, DoubleReplayMemory, StateChange
from ddqn.utils import run, train

from utils import make_env_config, build_run_config, build_training_config

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout)
logging.getLogger("ddqn").setLevel(logging.DEBUG)


if __name__ == "__main__":
    cli_args = process_cli_args()
    env_config = make_env_config(cli_args)

    replay_memory = DoubleReplayMemory[StateChange](
        capacity=cli_args.memory_size,
        warmup_episodes_upper=cli_args.warmup_episodes_upper,
        main_episodes_lower=cli_args.main_episodes_lower,
    )
    ddqn = DoubleDQN(
        replay_memory=replay_memory,
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval or cli_args.save:
        run_config = build_run_config(cli_args)
        run(ddqn, env_config=env_config, run_config=run_config)
    else:
        training_config = build_training_config(ddqn, cli_args, env_config)
        train(ddqn, env_config=env_config, training_config=training_config)
