import logging
import os
import sys

from ddqn.configure import process_cli_args
from ddqn.ddqn import DoubleDQN
from ddqn.parser import ddqn_parser
from ddqn.replay_buffer.experience_replay import ExperienceReplay
from ddqn.utils import train
from ddqn.watcher import (
    MovingAverageWatcher,
    ValueWatcher,
    WatcherTarget,
    WatcherType,
)

from utils import (
    build_env_config,
    build_run_config,
    build_training_config,
    custom_eval,
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout)
logging.getLogger("ddqn").setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


if __name__ == "__main__":
    cli_args = process_cli_args(ddqn_parser)
    env_config = build_env_config(cli_args)

    # Configure some logging
    watchers = [
        ValueWatcher(
            target=WatcherTarget.LOSS,
            watcher_type=WatcherType.GLOBAL_STEP,
            freq=100,
        ),
        ValueWatcher(
            target=WatcherTarget.EPSILON,
            watcher_type=WatcherType.GLOBAL_STEP,
            freq=100,
        ),
        MovingAverageWatcher(
            capacity=1_000,
            target=WatcherTarget.EPISODE_DURATIONS,
            watcher_type=WatcherType.EPISODES,
            freq=100,
        ),
        MovingAverageWatcher(
            capacity=100,
            target=WatcherTarget.REWARD,
            watcher_type=WatcherType.GLOBAL_STEP,
            freq=50,
        ),
        MovingAverageWatcher(
            capacity=1_000,
            target=WatcherTarget.REWARD,
            watcher_type=WatcherType.GLOBAL_STEP,
            freq=50,
        ),
    ]

    replay_buffer = ExperienceReplay(capacity=cli_args.buffer_size)
    ddqn = DoubleDQN(
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
            watchers=watchers,
        )
