import itertools
import logging
import time
from collections.abc import Collection

import numpy as np
import torch
import torch.optim as optim

from ..action_selection.epsilon_greedy import ExponentialDecay
from ..configure import DEVICE, EnvConfig, TrainingConfig
from ..step_handler import StepHandler, StepSummary
from ..ddqn import DoubleDQN
from ..replay_buffer import ReplayBuffer, StateChange, StateChanges
from ..watcher import Watcher

logger = logging.getLogger(__name__)


def train(
    ddqn: DoubleDQN,
    replay_buffer: ReplayBuffer[StateChange],
    *,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    watchers: Collection[Watcher] | None = None,
) -> None:
    """
    Main training loop. Trains `ddqn` in the environment set up in
    `env_config` with training parameters specified in
    `training_config`. Saves results to `outdir`.

    Parameters
    ----------
    ddqn : DoubleDQN
        Model to train.
    env_config : EnvConfig
        RL environment configuration.
    training_config :
        Training configuration.
    """

    # Ensure output dir exists
    outdir = training_config.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Adapt envs to return torch tensors everywhere
    env_reset = env_config.make_adapted_reset()
    env_step = env_config.make_adapted_step()

    # Alias for convenience
    action_selection = training_config.action_selector_fn

    ddqn = ddqn.to(DEVICE)

    # Configure what happens at end of steps/episodes/trainig
    step_handler = StepHandler(
        outdir=outdir,
        num_episodes=training_config.num_episodes,
        buffering_episodes=10,
        watchers=watchers or [],
        early_return_fn=training_config.early_return_fn,
        checkpoint_episodes=training_config.checkpoint_episodes,
    )
    step_handler.register_handler(ddqn)

    # Configure ε compuption for use in ε-greedy action selection
    epsilon_schedule = ExponentialDecay(
        start=training_config.epsilon_start,
        end=training_config.epsilon_end,
        decay=training_config.epsilon_decay,
        exploration_steps=training_config.epsilon_exploration_steps,
    )

    state: torch.Tensor
    next_state: torch.Tensor | None
    episode_duration: int | None
    global_step_number = 0
    episode_completed: bool
    training_completed: bool = False

    logger.debug("Starting training...")
    start = time.time()
    for episode_number in range(1, training_config.num_episodes + 1):
        if training_completed:
            logger.info("Early return condition met.")
            break

        episode_reward: float = 0.0

        state, _ = env_reset()

        for step_number in itertools.count(1):
            global_step_number += 1

            # Step
            epsilon = epsilon_schedule(global_step_number)
            action = action_selection(epsilon, state)
            next_state, reward, termd, truncd, _ = env_step(action.item())

            episode_reward += reward.item()  # type:ignore

            episode_completed = termd or truncd
            if episode_completed:
                next_state = None
                episode_duration = step_number
                this_episode_reward = episode_reward
            else:
                episode_duration = None
                this_episode_reward = None

            # Optimize *policy network* by one step
            loss = _optimize_one_step(
                ddqn,
                replay_buffer=replay_buffer,
                batch_size=training_config.batch_size,
                gamma=training_config.gamma,
                loss_fn=training_config.loss_fn,
                optimizer=training_config.optimiser,
            )

            # Update *target network* by one step
            _update_one_step(ddqn, tau=training_config.tau)

            step_summary = StepSummary(
                state=state,
                action=action,
                reward=reward,  # step reward
                next_state=next_state,
                epsilon=epsilon,
                loss=loss,
                step_number=step_number,
                episode_number=episode_number,
                global_step_number=global_step_number,
                episode_duration=episode_duration,
                episode_reward=this_episode_reward,
            )

            step_handler.step_end(step_summary, replay_buffer)

            if episode_completed:
                training_completed = step_handler.episode_end(
                    step_summary, ddqn
                )
                break

            else:
                state = next_state

    step_handler.train_end(ddqn)

    logger.debug("Finished training")
    logger.debug(f"Time taken: {time.time() - start:.3f}s")

    return


def _optimize_one_step(
    ddqn: DoubleDQN,
    replay_buffer: ReplayBuffer,
    *,
    batch_size: int,
    gamma: float,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
) -> float | None:
    r"""
    Optimize the policy network based on experience sampled from the
    target network's replay memory i.e. replay the target network's
    experience to the policy network ino order to train it.

    Since the batch is sampled from the target network's experience,
    the target network 'knows' what happened at t+1 (what action was
    taken, and what the next state was), whereas the policy state does
    not. This asymmetry means that the target network should have a
    more accurate prediction of the true state-action value function
    (Q-function) $Q^\ast$, and we use this to train the policy network.

    The predictions being fed into the loss function are thus the
    state-action values according to the policy network
    $Q^{\text{policy}}(s_t, a_t)$, derived from the policy
    network's estimate $Q^{\text{policy}}(s_t)$ and a
    concrete action $a_t$ taken at time t (sampled from the replay).

    The targets of the loss function are the state-action values
    according to the target network $Q^{\text{target}}(s_t, a_t)$.
    It is given by the sum of the concrete reward $r_t$ (awarded to
    the choice of action $a_t$) plus the estimated discounted value of
    the next state:
    $$
    Q^{\text{target}}(s_t, a_t) = r_t + \gamma V^{\text{target}}(s_t),
    $$
    where $V^{\text{target}}(s_t) = \max_a Q^{\text{target}}(s_t, a)$
    is the state value, according to the target network.

    Since DQN methods are notoriously unstable, we use the Huber loss
    to try and reduce sensitivity to extreme values and use gradient
    descent to minimise the loss.

    Note: the two networks in `ddqn` encode all of the state-action
    values at the same time. The networks maintain $Q(s_t)$ and the
    state-action value function $Q(s_t, a_i)$ of action i is given by
    the ith column of $Q(s_t)$.

    Parameters
    ----------
    ddqn : DoubleDQN
        Double DQN instance being trained.
    replay_buffer : ReplayBuffer
        Replay buffer to sample experiences from.
    batch_size : int
        Num. experiences to sample from `replay_buffer` per step.
    gamma : float
        Discount rate of future rewards.
    loss_fn : torch.nn.modules.loss._Loss,
        Loss function to train with.
    optimizer : optim.Optimizer
        Optimiser to train with.

    Returns
    -------
    float | None
        Loss of this step if `replay_buffer` has sufficient
        experience to sample from. If `replay_buffer` does not have
        sufficient experience to sample from, no optimisation is
        conducted on this step, and `None` is returned.
    """

    sample_state_changes = replay_buffer.sample(batch_size)
    if sample_state_changes is None:
        return None

    this_batch = StateChanges(sample_state_changes)

    with torch.no_grad():
        state_batch = torch.from_numpy(
            np.array([t.numpy() for t in this_batch.states], dtype=np.float32)
        )  # (batch_size, dim(state_space))
        action_batch = torch.from_numpy(
            np.array([t.numpy() for t in this_batch.actions], dtype=np.int64)
        )  # (batch_size, 1)
        reward_batch = torch.from_numpy(
            np.array([t.numpy() for t in this_batch.rewards], dtype=np.float32)
        )  # (batch_size, 1)

    # Compute predictions $Q^{\text{policy}}(s_t, a_t)$.
    predicted_state_action_values = ddqn.pnet(state_batch).gather(
        1, action_batch
    )  # (batch_size, dim(action_space) -> (batch_size, 1)

    # Compute target values
    # $Q^{\text{target}}(s_t, a_t) = r_t + \gamma V^{\text{target}}(s_{t+1})$.
    # Any experiences with `next_state = None` were terminated and so
    # automatically have a value of zero.
    with torch.no_grad():
        target_state_action_values = reward_batch  # (batch_size, 1)
        non_final_mask = torch.tensor(
            [ns is not None for ns in this_batch.next_states],
            dtype=torch.bool,
            device=DEVICE,
        )  # (batch_size,)
        non_final_next_states = torch.from_numpy(
            np.array(
                [
                    t.numpy()
                    for t, mask in zip(this_batch.next_states, non_final_mask)
                    if mask
                ],
                dtype=np.float32,
            )
        )  # (VAR, dim(state_space))
        if non_final_next_states.nelement() != 0:
            argmax_actions = (
                ddqn.pnet(non_final_next_states).argmax(1)  # (VAR, 1)
            ).unsqueeze(1)
            target_state_action_values[non_final_mask] += gamma * ddqn.tnet(
                non_final_next_states
            ).gather(1, argmax_actions)  # (VAR, 1)

    this_loss = loss_fn(
        predicted_state_action_values, target_state_action_values
    )  # SCALAR

    optimizer.zero_grad()
    this_loss.backward()

    torch.nn.utils.clip_grad_value_(ddqn.pnet.parameters(), 100)
    optimizer.step()

    return this_loss.item()


def _update_one_step(ddqn: DoubleDQN, *, tau: float) -> None:
    """
    Soft updates of target network with parameters of policy network
    to prevent too much drift.

    Copy over parameters *slowly* to mitigate the "moving targets"
    problem where the weights being learned by the policy network
    are trying to converge to a moving target. Doing it slowly enough
    allows us to treat the target as essentially fixed.

    The update rate is controlled by the convex combination
                    θ′ <- τ * θ + (1 - τ) * θ′
    where τ is the update rate.

    Modifies the target network weights in situ.

    Parameters
    ----------
    ddqn : DoubleDQN
        Double DQN being trained.
    tau : float
        Update rate τ controlling how fast the target network changes.
    """

    tnet_state = ddqn.tnet.state_dict()
    pnet_state = ddqn.pnet.state_dict()
    for theta in pnet_state.keys():
        tnet_state[theta] = (
            tau * pnet_state[theta] + (1.0 - tau) * tnet_state[theta]
        )
    ddqn.tnet.load_state_dict(tnet_state)
    return
