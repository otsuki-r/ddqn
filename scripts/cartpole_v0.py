import argparse
import logging
import os
import pathlib
import sys

import gymnasium as gym
from ddqn.configure import EnvConfig, build_training_config, process_cli_args
from ddqn.structures import DoubleDQN, ReplayMemory, StateChange
from ddqn.utils import run, train


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(), stream=sys.stdout
)
logging.getLogger("ddqn").setLevel(logging.DEBUG)


def make_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    render_mode = "human" if cli_args.eval else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    initial_state, _ = env.reset(seed=42)

    return EnvConfig(
        env=env,
        action_space_size=int(env.action_space.n),  # type: ignore
        state_space_size=len(initial_state),
    )


if __name__ == "__main__":
    cli_args = process_cli_args()
    env_config = make_env_config(cli_args)

    outdir = pathlib.Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ddqn = DoubleDQN(
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval:
        run(ddqn, env=env_config.env, weights_dir=cli_args.outdir)
    else:
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
