import itertools
import logging
import time
from collections.abc import Collection

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

    step_handler = StepHandler(
        outdir,
        num_episodes=training_config.num_episodes,
        buffering_episodes=10,
        watchers=watchers or [],
        early_return_fn=training_config.early_return_fn,
    )

    ddqn = ddqn.to(DEVICE)

    state: torch.Tensor
    next_state: torch.Tensor | None
    completed: bool
    episode_duration: int | None
    global_step_number = 0

    epsilon_schedule = ExponentialDecay(
        eps_start=training_config.epsilon_start,
        eps_end=training_config.epsilon_end,
        eps_decay=training_config.epsilon_decay,
    )

    logger.debug("Starting training...")
    start = time.time()
    for episode_number in range(1, training_config.num_episodes + 1):
        episode_reward: float = 0.0

        _state, _ = env_config.env.reset()
        state = env_config.state_space_adaptor(
            _state,
            dtype=env_config.state_space_dtype,
            device=DEVICE,
        )

        for step_number in itertools.count(1):
            global_step_number += 1
            epsilon = epsilon_schedule(global_step_number)

            action = training_config.action_selector_fn(epsilon, state)

            _next_state, _reward, terminated, truncated, _ = (
                env_config.env.step(action.item())
            )
            episode_reward += _reward  # type:ignore

            reward = torch.tensor(
                [_reward], dtype=env_config.reward_dtype, device=DEVICE
            )

            completed = terminated or truncated

            if completed:
                next_state = None
                episode_duration = step_number
                this_episode_reward = episode_reward
            else:
                next_state = env_config.state_space_adaptor(
                    _next_state,
                    dtype=env_config.state_space_dtype,
                    device=DEVICE,
                )
                episode_duration = None
                this_episode_reward = None

            # Optimize the *policy network* by one step
            loss = _optimize_one_step(
                ddqn,
                replay_buffer=replay_buffer,
                batch_size=training_config.batch_size,
                gamma=training_config.gamma,
                loss_fn=training_config.loss_fn,
                optimizer=training_config.optimiser,
            )

            # Update the *target network* by one step
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

            if completed:
                step_handler.episode_end(step_summary)
                break

            else:
                state = next_state

        # Check if early return has been satisfied
        training_config.early_return_fn.update(episode_reward)
        if (
            episode_number % 50 == 0
            and training_config.early_return_fn.evaluate(episode_number)
        ):
            logger.info("Early return condition met.")
            break

    # Write out remaining values
    step_handler.flush()

    ddqn.save(outdir)

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

    state_batch = torch.stack(
        this_batch.states
    )  # (batch_size, dim(state_space))
    action_batch = torch.stack(this_batch.actions)  # (batch_size, 1)
    reward_batch = torch.stack(this_batch.rewards)  # (batch_size, 1)

    # Compute predictions $Q^{\text{policy}}(s_t, a_t)$.
    predicted_state_action_values = ddqn.pnet(state_batch).gather(
        1, action_batch
    )  # (batch_size, dim(action_space) -> (batch_size, 1)

    # Compute target values
    # $Q^{\text{target}}(s_t, a_t) = r_t + \gamma V^{\text{target}}(s_{t+1})$.
    # Any experiences with `next_state = None` were terminated and so
    # automatically have a value of zero.
    target_next_state_values = torch.zeros(
        (batch_size, 1), dtype=torch.float32, device=DEVICE
    )  # (batch_size, 1)
    non_final_mask = torch.tensor(
        list(map(lambda s: s is not None, this_batch.next_states)),
        dtype=torch.bool,
        device=DEVICE,
    )  # (batch_size,)
    _non_final_next_states = [
        s for s in this_batch.next_states if s is not None
    ]
    if _non_final_next_states:
        with torch.no_grad():
            non_final_next_states = torch.stack(
                _non_final_next_states
            )  # (VAR, dim(state_space))
            argmax_actions = (
                ddqn.pnet(non_final_next_states).argmax(1)  # (VAR,)
            )
            target_next_state_values[non_final_mask] = ddqn.tnet(
                non_final_next_states
            ).gather(1, argmax_actions.unsqueeze(1))  # (batch_size,)

    target_state_action_values = (
        reward_batch + gamma * target_next_state_values
    )  # (batch_size, 1)

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
