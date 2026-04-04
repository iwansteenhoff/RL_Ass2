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
from DQN import QNetwork

class DQN_Agent():

    def __init__(self, obs_dim, n_actions, learning_rate, gamma, hidden_dim=128, device=None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # target network
        self.target_network = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_net.state_dict())
        self.target_network.eval()

        self.target_update_freq = 1000
        self.step_count = 0

    def _to_tensor(self, s):
        return torch.from_numpy(s).to(self.device)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        s_tensor = self._to_tensor(s)

        if s_tensor.ndim == 1:
            s_tensor = s_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(s_tensor) # vectorized (shape N env, n actions)

        greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        if policy == 'greedy':
            return greedy_actions if len(greedy_actions) > 1 else greedy_actions[0]
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            rand_actions = np.random.randint(0, self.n_actions, size=len(s))
            mask = np.random.rand(len(s)) < epsilon

            a = np.where(mask, rand_actions, greedy_actions)

            return a if len(greedy_actions) > 1 else a[0]

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            probs = torch.softmax(q_values / temp, dim=1)
            a = torch.multinomial(probs, 1).squeeze(1)

            return a.cpu().numpy() if len(a) > 1 else a.item()

        else:
            raise ValueError(f"Unknown policy: {policy}")

    # batch update trial
    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states)  # (B, n_actions)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1).values # switch to update with target network instead of q network
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        self.step_count += 1

        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_net.state_dict())

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=500):
        returns = []
        for i in range(n_eval_episodes):
            s, _ = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, terminated, truncated, _ = eval_env.step(a)
                done = terminated or truncated
                R_ep += r
                if done:
                    break
                s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


def DQN_TN(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None,
                hidden_dim=128, env_steps_per_update=100, plot=False, eval_interval=500,
                n_eval_episodes=10, seed=None):
    '''
    Runs a single repetition of a Monte Carlo RL agent.
    Returns:
        eval_returns: array of evaluation returns
        eval_timesteps: array of timesteps at which evaluation happened
    '''

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def make_env():
        return gym.make("CartPole-v1")
    
    env = gym.make("CartPole-v1", render_mode="human" if plot else None)
    eval_env = gym.make("CartPole-v1")

    num_envs = 20

    envs = SyncVectorEnv([make_env for _ in range(num_envs)])

    obs_dim = env.observation_space.shape[0]   # CartPole observation is a 4D vector
    n_actions = env.action_space.n             # CartPole has a discrete action space

    pi = DQN_Agent(obs_dim, n_actions, learning_rate, gamma, hidden_dim=hidden_dim)

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
            batch.append((s[i], a[i], r[i], s_next[i], done[i]))

        if len(batch) >= batch_size:
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
    eval_env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps)


def test_TN():
    n_timesteps = 1000000
    max_episode_length = 500
    gamma = 0.99
    learning_rate = 5e-4

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    plot = False

    eval_returns, eval_timesteps = DQN_TN(
        n_timesteps=n_timesteps,
        max_episode_length=max_episode_length,
        learning_rate=learning_rate,
        gamma=gamma,
        policy=policy,
        epsilon=epsilon,
        temp=temp,
        plot=plot
    )

    print("Evaluation timesteps:", eval_timesteps)
    print("Evaluation returns:", eval_returns)

    plt.plot(eval_timesteps, eval_returns)
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    test_TN()
    print(time.time() - t0)
