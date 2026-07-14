"""
Q-Micro :: strategies.rl_execution_agent
--------------------------------------------
DQN agent for optimal order execution (PyTorch).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from simulation.execution_env import ExecutionEnv, ExecutionEnvConfig


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNExecutionAgent:
    def __init__(self, state_dim: int = 6, n_actions: int = 4, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_actions = n_actions

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(q_values.argmax(dim=1).item())

    def optimize(self, batch_size: int = 64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train(n_episodes: int = 200, target_update_every: int = 10,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.97):
    env_cfg = ExecutionEnvConfig()
    agent = DQNExecutionAgent()
    epsilon = eps_start
    episode_rewards = []

    for ep in range(n_episodes):
        env = ExecutionEnv(env_cfg)
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.optimize()
            state = next_state
            total_reward += reward

        if ep % target_update_every == 0:
            agent.update_target()

        epsilon = max(eps_end, epsilon * eps_decay)
        episode_rewards.append(total_reward)

        if ep % 20 == 0:
            print(f"episode={ep} reward={total_reward:.2f} epsilon={epsilon:.3f}")

    return agent, episode_rewards


if __name__ == "__main__":
    train()