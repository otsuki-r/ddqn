import itertools
import pathlib
import time

import torch
import torch.optim as optim
import tqdm

from .base_logger import logger
from .configure import EnvConfig, TrainingConfig
from .dqn import DoubleDQN
from .plot import plot_episode_durations, plot_losses, plot_rewards
from .replay_memory import StateChange, StateChanges, ReplayMemory


def train(
    dqn: DoubleDQN,
    *,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    replay_memory: ReplayMemory[StateChange],
    outdir: pathlib.Path,
) -> None:
    dqn = dqn.to(training_config.device)

    state: torch.Tensor
    next_state: None | torch.Tensor
    episode_durations: list[int] = []
    completed: bool
    losses: list[float] = []
    rewards: list[int] = []

    logger.debug("Starting training...")
    start = time.time()
    for episode_num in tqdm.tqdm(range(training_config.num_episodes)):
        _state, _ = env_config.env.reset()
        state = torch.tensor(
            _state, dtype=torch.float32, device=training_config.device
        ).unsqueeze(0)

        this_episode_reward = 0
        for step_number in itertools.count():
            action = training_config.action_selector(
                step_number, state, dqn.pnet
            )

            _next_state, _reward, terminated, truncated, _ = (
                env_config.env.step(action.item())
            )
            # Penalise if cart position is too far from center
            # n.b. -4.8 <= cart_position <= 4.8
            _reward -= abs(_next_state[0]) ** 2 / 20

            this_episode_reward += int(
                float(_reward)
            )  # Weird cast for type checker

            reward = torch.tensor(
                [[_reward]], dtype=torch.float32, device=training_config.device
            )

            completed = terminated or truncated

            if completed:
                next_state = None
            else:
                next_state = torch.tensor(
                    _next_state,
                    dtype=torch.float32,
                    device=training_config.device,
                ).unsqueeze(0)

            exp = StateChange(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )
            replay_memory.append(exp)

            # Optimize the *policy network* by one step
            this_loss = optimize_one_step(
                dqn,
                replay_memory=replay_memory,
                batch_size=training_config.batch_size,
                discount=training_config.discount,
                loss_fn=training_config.loss_fn,
                optimizer=training_config.optimiser,
                device=training_config.device,
            )

            # Update the *target network* by one step
            update_one_step(dqn, update_rate=training_config.update_rate)

            if completed:
                episode_durations.append(step_number + 1)
                if this_loss is not None:
                    losses.append(this_loss)
                rewards.append(this_episode_reward)
                break

            else:
                assert next_state is not None
                state = next_state

    plot_episode_durations(episode_durations, outdir / "episode_durations.png")
    plot_losses(losses, outdir / "losses.png")
    plot_rewards(rewards, outdir / "rewards.png")
    dqn.save(outdir)

    logger.debug("Finished training")
    logger.debug(f"Time taken: {time.time() - start:.3f}s")
    return


def optimize_one_step(
    ddqn: DoubleDQN,
    *,
    replay_memory: ReplayMemory,
    batch_size: int,
    discount: float,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float | None:
    if len(replay_memory) < batch_size:
        return None

    sample_state_changes = replay_memory.sample(batch_size)
    this_batch = StateChanges(sample_state_changes)

    non_final_mask = torch.tensor(
        list(map(lambda s: s is not None, this_batch.next_states)),
        dtype=torch.bool,
        device=device,
    )

    non_final_next_states = torch.cat(
        [s for s in this_batch.next_states if s is not None]
    )

    state_batch = torch.cat(this_batch.states)
    action_batch = torch.cat(this_batch.actions)
    reward_batch = torch.cat(this_batch.rewards)

    state_action_values = ddqn.pnet(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            ddqn.tnet(non_final_next_states).max(1).values
        )

    expected_state_action_values = (
        next_state_values * discount + reward_batch.squeeze(1)
    )

    this_loss = loss_fn(
        state_action_values.squeeze(1), expected_state_action_values
    )

    optimizer.zero_grad()
    this_loss.backward()

    torch.nn.utils.clip_grad_value_(ddqn.pnet.parameters(), 100)
    optimizer.step()

    return this_loss.item()


def update_one_step(ddqn: DoubleDQN, *, update_rate: float) -> None:
    """
    Updates each parameter according to:
    θ′ <- τ * θ + (1 - τ) * θ′
    """
    tnet_state = ddqn.tnet.state_dict()
    pnet_state = ddqn.pnet.state_dict()
    for key in pnet_state.keys():
        tnet_state[key] = pnet_state[key] * update_rate + tnet_state[key] * (
            1.0 - update_rate
        )
    ddqn.tnet.load_state_dict(tnet_state)
    return
