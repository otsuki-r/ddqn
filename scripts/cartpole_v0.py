import pathlib

import gymnasium as gym
import torch
from ddqn.logger import logger
from ddqn.configure import (
    build_training_config,
    get_env_config,
    process_cli_args,
)
from ddqn.structures import DoubleDQN, ReplayMemory, StateChange
from ddqn.utils import train

if __name__ == "__main__":
    # Configuration
    cli_args = process_cli_args()
    env_config = get_env_config(cli_args)
    logger.info("Running with args %s", vars(cli_args))

    outdir = pathlib.Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Neural net to train
    ddqn = DoubleDQN(
        n_observations=env_config.state_space_size,
        n_actions=env_config.action_space_size,
    )

    if cli_args.eval:
        dqn.load_state_dict(outdir)
        dqn.eval()

        import time
        import random

        while True:
            env = gym.make("CartPole-v1", render_mode="human")
            this_seed = random.randint(1, 1_000_000)
            logger.debug("Seed: %s", this_seed)
            _state, _ = env.reset(seed=this_seed)
            state = torch.tensor(_state, dtype=torch.float32)

            completed = False
            this_duration = 0
            while not completed:
                action = dqn.pnet(state).max(0).indices.item()
                next_state, _, terminated, truncated, _ = env.step(action)
                state = torch.tensor(next_state)
                completed = terminated or truncated
                this_duration += 1
            logger.debug("This episode duration: %s", this_duration)

            env.close()
            time.sleep(1)
    else:
        # Set up training
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
