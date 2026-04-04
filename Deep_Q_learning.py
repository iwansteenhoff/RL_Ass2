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

class DQN_Agent():

    def __init__(self, obs_dim, n_actions, learning_rate, gamma, device=None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _to_tensor(self, s):
        s = np.array(s, dtype=np.float32)
        return torch.tensor(s, dtype=torch.float32, device=self.device)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        s_tensor = self._to_tensor(s).unsqueeze(0)  # shape: (1, obs_dim)

        with torch.no_grad():
            q_values = self.q_net(s_tensor).squeeze(0)  # shape: (n_actions,)

        if policy == 'greedy':
            a = int(torch.argmax(q_values).item())

        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)
            else:
                a = int(torch.argmax(q_values).item())

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            probs = torch.softmax(q_values / temp, dim=0).cpu().numpy()
            a = np.random.choice(self.n_actions, p=probs)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        return a

    def update(self, s, a, r, s_next, done):
        s = self._to_tensor(s).unsqueeze(0)
        s_next = self._to_tensor(s_next).unsqueeze(0)

        q_value = self.q_net(s)[0, a]

        with torch.no_grad():
            next_q_value = torch.max(self.q_net(s_next), dim=1).values[0]
            target = r if done else r + self.gamma * next_q_value.item()

        target = torch.tensor(target, dtype=torch.float32, device=self.device)

        loss = self.loss_fn(q_value, target)

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


def DQL_run(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None,
                plot=True, eval_interval=10000):
    '''
    Runs a single repetition of a Monte Carlo RL agent.
    Returns:
        eval_returns: array of evaluation returns
        eval_timesteps: array of timesteps at which evaluation happened
    '''

    env = gym.make("CartPole-v1", render_mode="human" if plot else None)
    eval_env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]   # CartPole observation is a 4D vector
    n_actions = env.action_space.n             # CartPole has a discrete action space

    pi = DQN_Agent(obs_dim, n_actions, learning_rate, gamma)

    eval_timesteps = []
    eval_returns = []

    timestep = 0

    while timestep < n_timesteps:
        s, _ = env.reset()

        done = False
        t = 0

        while not done and t < max_episode_length and timestep < n_timesteps:
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            pi.update(s, a, r, s_next, done)

            s = s_next
            t += 1
            timestep += 1

            if timestep % eval_interval == 0:
                mean_return = pi.evaluate(
                    eval_env,
                    n_eval_episodes=10,
                    max_episode_length=max_episode_length
                )
                eval_timesteps.append(timestep)
                eval_returns.append(mean_return)

                print(timestep)

    env.close()
    eval_env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000000
    max_episode_length = 500
    gamma = 0.99
    learning_rate = 1e-3

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    plot = True

    eval_returns, eval_timesteps = DQL_run(
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


if __name__ == '__main__':
    test()