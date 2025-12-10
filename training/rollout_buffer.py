"""Rollout buffer for storing trajectories and computing advantages."""

import torch
from typing import List, Dict, Any


class RolloutBuffer:
    """Stores trajectories and computes GAE advantages."""
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize rollout buffer.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage
        self.states: List[Dict[str, Any]] = []
        self.actions_type: List[int] = []
        self.actions_element: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        
    def add(self, state: Dict[str, Any], action_type: int, action_element: int,
            reward: float, done: bool, log_prob: float, value: float):
        """Add a single step to the buffer."""
        self.states.append(state)
        self.actions_type.append(action_type)
        self.actions_element.append(action_element)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages(self, last_value: float = 0.0) -> torch.Tensor:
        """
        Compute GAE advantages and returns.
        
        Args:
            last_value: Value estimate for the last state (if episode didn't terminate)
        
        Returns:
            advantages: [buffer_size] tensor
            returns: [buffer_size] tensor
        """
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = last_value
        
        # Compute backwards from the end
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * next_value * (1 - self.dones[step]) - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            
            # Return = advantage + value
            returns.insert(0, gae + self.values[step])
            
            next_value = self.values[step]
        
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return advantages_tensor, returns_tensor
    
    def get_batch(self, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Get all data as tensors for training.
        
        Returns:
            Dictionary with all tensors
        """
        return {
            'states': self.states,
            'actions_type': torch.tensor(self.actions_type, dtype=torch.long),
            'actions_element': torch.tensor(self.actions_element, dtype=torch.long),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.bool),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'values': torch.tensor(self.values, dtype=torch.float32),
            'advantages': advantages,
            'returns': returns,
        }
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions_type.clear()
        self.actions_element.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def merge(self, other: 'RolloutBuffer'):
        """Merge another buffer into this one."""
        self.states.extend(other.states)
        self.actions_type.extend(other.actions_type)
        self.actions_element.extend(other.actions_element)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.log_probs.extend(other.log_probs)
        self.values.extend(other.values)
    
    def __len__(self):
        return len(self.states)

