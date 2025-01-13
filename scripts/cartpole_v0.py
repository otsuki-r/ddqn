import logging
import pathlib

import gymnasium as gym
from ddqn.configure import (
    build_training_config,
    get_env_config,
    process_cli_args,
)
from ddqn.structures import DoubleDQN, ReplayMemory, StateChange
from ddqn.utils import run, train

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    cli_args = process_cli_args()
    env_config = get_env_config(cli_args)

    outdir = pathlib.Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ddqn = DoubleDQN(
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval:
        env = gym.make("CartPole-v1", render_mode="human")
        run(ddqn, env=env, weights_dir=cli_args.outdir)
    else:
        logger.info("Training args: %s", vars(cli_args))

        replay_memory: ReplayMemory[StateChange] = ReplayMemory(10_000)
        training_config = build_training_config(
            cli_args, env_config, parameters=ddqn.pnet.parameters()
        )

        train(
            ddqn,
            env_config=env_config,
            training_config=training_config,
            replay_memory=replay_memory,
            outdir=outdir,
        )
