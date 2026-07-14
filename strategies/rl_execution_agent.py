"""
RL Execution Agent for Q-Micro.
Implements DQN and PPO agents for optimal execution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    input_dim: int = 6          # State dimension
    output_dim: int = 9        # Action space size
    hidden_dim: int = 64       # Hidden layer size
    lr: float = 0.001          # Learning rate
    gamma: float = 0.99        # Discount factor
    epsilon: float = 1.0       # Exploration rate
    epsilon_min: float = 0.01 # Minimum exploration rate
    epsilon_decay: float = 0.995 # Exploration decay rate
    batch_size: int = 64       # Batch size
    memory_size: int = 10000   # Replay buffer size
    target_update: int = 10    # Target network update frequency


class DQNNetwork(nn.Module):
    """Neural network for DQN agent."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    DQN agent for execution optimization.
    Uses experience replay and target network for stability.
    """
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNNetwork(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.target_net = DQNNetwork(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        
        # Replay buffer
        self.memory = deque(maxlen=config.memory_size)
        
        # Exploration
        self.epsilon = config.epsilon
        
        # Training stats
        self.losses = []
        self.rewards = []
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.config.output_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """Train on a batch of experiences from the replay buffer."""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update stats
        self.losses.append(loss.item())
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        
        return loss.item()
    
    def update_target(self) -> None:
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    input_dim: int = 6          # State dimension
    output_dim: int = 9        # Action space size
    hidden_dim: int = 64       # Hidden layer size
    lr: float = 0.0003         # Learning rate
    gamma: float = 0.99        # Discount factor
    clip_epsilon: float = 0.2 # PPO clip parameter
    batch_size: int = 64       # Batch size
    epochs: int = 4            # Number of epochs per update
    gae_lambda: float = 0.95  # GAE lambda parameter


class PPONetwork(nn.Module):
    """Neural network for PPO agent (actor-critic)."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Actor
        self.actor_fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, output_dim)
        
        # Critic
        self.critic_fc1 = nn.Linear(input_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
    
    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))
        return F.softmax(self.actor_head(x), dim=-1)
    
    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.critic_fc1(x))
        x = F.relu(self.critic_fc2(x))
        return self.critic_head(x)


class PPOAgent:
    """
    PPO agent for execution optimization.
    Uses Generalized Advantage Estimation (GAE) and clipped objective.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network
        self.net = PPONetwork(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select an action and compute log prob and value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Actor
        action_probs = self.net.forward_actor(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Critic
        value = self.net.forward_critic(state_tensor).squeeze()
        
        return action.item(), log_prob.item(), value.item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, log_prob: float, value: float, done: bool) -> None:
        """Store experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def update(self) -> float:
        """Update the agent using PPO."""
        if len(self.states) < self.config.batch_size:
            return 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Optimize for multiple epochs
        for _ in range(self.config.epochs):
            # Forward pass
            action_probs = self.net.forward_actor(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            values = self.net.forward_critic(states).squeeze()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute losses
            policy_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages,
            ).mean()
            
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.net.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        self.net.load_state_dict(torch.load(path))


class RLExecutionAgent:
    """
    High-level RL execution agent.
    Supports DQN and PPO algorithms.
    """
    
    def __init__(self, algorithm: str = "DQN", config: Optional[Dict] = None):
        self.algorithm = algorithm
        
        if algorithm == "DQN":
            if config is None:
                config = DQNConfig()
            self.agent = DQNAgent(config)
        elif algorithm == "PPO":
            if config is None:
                config = PPOConfig()
            self.agent = PPOAgent(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the RL agent."""
        if self.algorithm == "DQN":
            return self.agent.select_action(state)
        elif self.algorithm == "PPO":
            action, _, _ = self.agent.select_action(state)
            return action
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience."""
        if self.algorithm == "DQN":
            self.agent.remember(state, action, reward, next_state, done)
        elif self.algorithm == "PPO":
            if self.algorithm == "PPO":
                # For PPO, we need to store log_prob and value
                # This is handled in the PPOAgent's select_action method
                pass
    
    def update(self) -> Optional[float]:
        """Update the RL agent."""
        if self.algorithm == "DQN":
            loss = self.agent.replay()
            if self.agent.current_step % self.agent.config.target_update == 0:
                self.agent.update_target()
            return loss
        elif self.algorithm == "PPO":
            return self.agent.update()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(
        self,
        env,
        episodes: int = 100,
        max_steps: int = 1000,
        render: bool = False,
    ) -> List[float]:
        """
        Train the RL agent on the given environment.
        
        Args:
            env: TradingEnvironment instance.
            episodes: Number of training episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.
        
        Returns:
            List of episode rewards.
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            for step in range(max_steps):
                if render:
                    env.render()
                
                # Select action
                action = self.select_action(state)
                
                # Execute action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store experience
                if self.algorithm == "DQN":
                    self.agent.remember(state, action, reward, next_state, done)
                elif self.algorithm == "PPO":
                    # For PPO, we need to store additional info
                    action, log_prob, value = self.agent.select_action(state)
                    self.agent.remember(state, action, reward, log_prob, value, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
            
            # Update agent
            if self.algorithm == "DQN":
                self.agent.replay()
                if episode % self.agent.config.target_update == 0:
                    self.agent.update_target()
            elif self.algorithm == "PPO":
                self.agent.update()
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")
        
        return episode_rewards
    
    def save(self, path: str) -> None:
        """Save the agent's model."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent's model."""
        self.agent.load(path)
