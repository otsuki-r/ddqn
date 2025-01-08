import pathlib

from libs.base_logger import logger
from libs.configure import (
    build_training_config,
    get_env_config,
    process_cli_args,
)
from libs.dqn import DoubleDQN
from libs.replay_memory import ReplayMemory, StateChange
from libs.train import train

if __name__ == "__main__":
    # Configuration
    cli_args = process_cli_args()
    env_config = get_env_config()
    logger.info("Running with args %s", vars(cli_args))

    outdir = pathlib.Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Neural net to train
    dqn = DoubleDQN(
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    # Set up training
    replay_memory: ReplayMemory[StateChange] = ReplayMemory(1_000)
    training_config = build_training_config(
        cli_args, env_config, parameters=dqn.pnet.parameters()
    )

    train(
        dqn,
        env_config=env_config,
        training_config=training_config,
        replay_memory=replay_memory,
        outdir=outdir,
    )
