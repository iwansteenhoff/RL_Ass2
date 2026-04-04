#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
from gymnasium.vector import SyncVectorEnv
import matplotlib.pyplot as plt
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQN_Agent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=128,
        use_target_network=False,
        target_update_freq=1000,
        device=None
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = self.loss_fn = nn.MSELoss()

        # Only create a target network if requested
        if self.use_target_network:
            self.target_network = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
            self.target_network.load_state_dict(self.q_net.state_dict())
            self.target_network.eval()
        else:
            self.target_network = None

    def _to_tensor(self, s):
        return torch.from_numpy(np.asarray(s, dtype=np.float32)).to(self.device)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        s_tensor = self._to_tensor(s)

        single_state = False
        if s_tensor.ndim == 1:
            s_tensor = s_tensor.unsqueeze(0)
            single_state = True

        with torch.no_grad():
            q_values = self.q_net(s_tensor)

        greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()

        if policy == 'greedy':
            return greedy_actions[0] if single_state else greedy_actions

        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            batch_size = s_tensor.shape[0]
            rand_actions = np.random.randint(0, self.n_actions, size=batch_size)
            mask = np.random.rand(batch_size) < epsilon
            a = np.where(mask, rand_actions, greedy_actions)

            return a[0] if single_state else a

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            probs = torch.softmax(q_values / temp, dim=1)
            a = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()

            return a[0] if single_state else a

        else:
            raise ValueError(f"Unknown policy: {policy}")

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s,a) from online network
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap target
        with torch.no_grad():
            if self.use_target_network:
                next_q_values = self.target_network(next_states).max(1).values
            else:
                next_q_values = self.q_net(next_states).max(1).values

            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.use_target_network:
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_net.state_dict())

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=500):
        returns = []
        for _ in range(n_eval_episodes):
            s, _ = eval_env.reset()
            R_ep = 0

            for _ in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, terminated, truncated, _ = eval_env.step(a)
                done = terminated or truncated
                R_ep += r

                if done:
                    break

                s = s_prime

            returns.append(R_ep)

        return np.mean(returns)


def DQN_run(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy='egreedy',
    epsilon=None,
    temp=None,
    hidden_dim=128,
    env_steps_per_update=100,
    plot=False,
    eval_interval=500,
    n_eval_episodes=10,
    use_replay_buffer=False,
    min_replay_size=1000,
    use_target_network=False,
    target_update_freq=1000,
    seed=None
):
    """
    Runs a single repetition of a DQN agent.
    Returns:
        eval_returns: array of evaluation returns
        eval_timesteps: array of timesteps at which evaluation happened
    """

    replay_buffer = deque(maxlen=100000)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def make_env():
        return gym.make("CartPole-v1")

    env = gym.make("CartPole-v1", render_mode="human" if plot else None)
    eval_env = gym.make("CartPole-v1")

    num_envs = 20
    envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    pi = DQN_Agent(
        obs_dim,
        n_actions,
        learning_rate,
        gamma,
        hidden_dim=hidden_dim,
        use_target_network=use_target_network,
        target_update_freq=target_update_freq
    )

    batch = []
    batch_size = env_steps_per_update

    eval_timesteps = []
    eval_returns = []

    timestep = 0
    s, _ = envs.reset(seed=seed)

    while timestep < n_timesteps:
        a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

        s_next, r, terminated, truncated, _ = envs.step(a)
        done = terminated | truncated

        for i in range(num_envs):
            transition = (s[i], a[i], r[i], s_next[i], done[i])

            if use_replay_buffer:
                replay_buffer.append(transition)
            else:
                batch.append(transition)

        if use_replay_buffer and len(replay_buffer) >= min_replay_size:
            sampled_batch = random.sample(replay_buffer, batch_size)
            pi.update_batch(sampled_batch)

        if not use_replay_buffer and len(batch) >= batch_size:
            pi.update_batch(batch[:batch_size])
            batch = batch[batch_size:]

        s = s_next
        timestep += num_envs

        if timestep % eval_interval == 0:
            mean_return = pi.evaluate(
                eval_env,
                n_eval_episodes=n_eval_episodes,
                max_episode_length=max_episode_length
            )
            eval_timesteps.append(timestep)
            eval_returns.append(mean_return)

    envs.close()
    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1_000_000
    max_episode_length = 500
    gamma = 0.99
    learning_rate = 1e-3

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    use_replay_buffer = True
    use_target_network = True
    target_update_freq = 1000

    plot = True

    eval_returns, eval_timesteps = DQN_run(
        n_timesteps=n_timesteps,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        gamma=gamma,
        policy=policy,
        epsilon=epsilon,
        temp=temp,
        plot=plot,
        use_replay_buffer=use_replay_buffer,
        min_replay_size=1000,
        use_target_network=use_target_network,
        target_update_freq=target_update_freq
    )

    print("Evaluation timesteps:", eval_timesteps)
    print("Evaluation returns:", eval_returns)

    plt.plot(eval_timesteps, eval_returns)
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation return")
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    test()
    print(time.time() - t0)