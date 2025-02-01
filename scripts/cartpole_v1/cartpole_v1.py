import logging
import os
import sys

from ddqn.configure import process_cli_args
from ddqn.ddqn import DoubleDQN
from ddqn.replay_buffer.double_experience_replay import DoubleExperienceReplay
from ddqn.utils import run, train
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
    ddqn_parser,
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout)
logging.getLogger("ddqn").setLevel(logging.INFO)


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
            capacity=100,
            target=WatcherTarget.EPISODE_DURATIONS,
            watcher_type=WatcherType.EPISODES,
            freq=100,
        ),
        MovingAverageWatcher(
            capacity=1_000,
            target=WatcherTarget.REWARD,
            watcher_type=WatcherType.EPISODES,
            freq=50,
        ),
        MovingAverageWatcher(
            capacity=250,
            target=WatcherTarget.REWARD,
            watcher_type=WatcherType.EPISODES,
            freq=50,
        ),
    ]

    replay_buffer = DoubleExperienceReplay(
        capacity=cli_args.buffer_size,
        warmup_episodes=cli_args.warmup_episodes,
        frac_warmup=0.05,
    )
    ddqn = DoubleDQN(
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval:
        run_config = build_run_config(cli_args)
        run(ddqn, env_config=env_config, run_config=run_config)
    else:
        training_config = build_training_config(ddqn, cli_args, env_config)
        train(
            ddqn,
            replay_buffer,
            env_config=env_config,
            training_config=training_config,
            watchers=watchers,
        )
