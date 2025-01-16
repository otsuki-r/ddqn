import argparse
import functools
import logging
import os
import pathlib
import sys

import gymnasium as gym
import torch.optim as optim
import torch.nn as nn
from ddqn.configure import EnvConfig, build_training_config, process_cli_args
from ddqn.structures import DoubleDQN, DoubleReplayMemory, StateChange
from ddqn.utils import run, train
from ddqn.early_stop import winsorised_durations_early_return


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(), stream=sys.stdout
)
logging.getLogger("ddqn").setLevel(logging.DEBUG)


class CenteredCartPole(gym.Env):
    """
    Modified CartPole env with a penalty to encourage it to stay in
    the middle of the viewport.
    """

    def __init__(self, render_mode: str | None = None) -> None:
        self.env = gym.make("CartPole-v1", render_mode=render_mode)

        def penalise_non_centered(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                next_state, reward, *g = fn(*args, **kwargs)

                # Penalise if cart position is too far from center
                # n.b. -2.4 <= cart_position <= 2.4
                reward -= min(next_state[0] ** 2 * 10, 1)

                return next_state, reward, *g

            return inner

        setattr(self.env, "step", penalise_non_centered(self.env.step))


def make_env_config(cli_args: argparse.Namespace) -> EnvConfig:
    render_mode: str | None
    if cli_args.eval:
        render_mode = "human"
    elif cli_args.save:
        render_mode = "rgb_array"
    else:
        render_mode = None

    cart_pole = CenteredCartPole(render_mode=render_mode)
    initial_state, _ = cart_pole.env.reset(seed=42)

    return EnvConfig(
        env=cart_pole.env,
        action_space_size=int(cart_pole.env.action_space.n),  # type: ignore
        state_space_size=len(initial_state),
    )


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
        save_path = pathlib.Path(cli_args.save_path) if cli_args.save else None
        run(
            ddqn,
            env=env_config.env,
            weights_dir=outdir,
            save_path=cli_args.save_path,
        )
    else:
        training_config = build_training_config(
            cli_args,
            env_config,
            optimiser=optim.AdamW(
                ddqn.pnet.parameters(), lr=cli_args.alpha, amsgrad=True
            ),
            loss_fn=nn.SmoothL1Loss(),  # Generalisation of robust Huber loss
            early_return_fn=winsorised_durations_early_return(
                last_n=20, clip_lower=5, clip_upper=0, score_threshold=470.0
            ),
        )

        logging.info("Training with config: %s", training_config)

        train(
            ddqn,
            env_config=env_config,
            training_config=training_config,
            outdir=outdir,
        )
