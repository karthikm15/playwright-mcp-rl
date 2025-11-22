"""Simple MLP policy for behavior cloning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Simple MLP policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128]):
        """
        Initialize MLP policy.
        
        Args:
            state_dim: Dimension of state encoding
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns action logits.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            logits: Action logits [batch_size, action_dim]
        """
        features = self.encoder(state)
        logits = self.policy_head(features)
        return logits

