import logging
import os
import pathlib
import sys

from ddqn.configure import process_cli_args
from ddqn.structures import DoubleDQN, DoubleReplayMemory, StateChange
from ddqn.utils import run, train

from utils import make_env_config, build_training_config


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(), stream=sys.stdout
)
logging.getLogger("ddqn").setLevel(logging.DEBUG)


if __name__ == "__main__":
    cli_args = process_cli_args()
    env_config = make_env_config(cli_args)

    outdir = pathlib.Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        run(
            ddqn,
            env=env_config.env,
            weights_dir=outdir,
            save_path=(
                pathlib.Path(cli_args.save_path) if cli_args.save else None
            ),
        )
    else:
        training_config = build_training_config(ddqn, cli_args, env_config)

        logging.info("Training with config: %s", training_config)

        train(
            ddqn,
            env_config=env_config,
            training_config=training_config,
            outdir=outdir,
        )
