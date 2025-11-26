"""MLP policy for behavior cloning with compositional action space."""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    MLP policy network with compositional action space.
    
    Instead of producing logits over a flat (type, element_ref) vocabulary,
    this policy produces:
      - action_type_logits: logits over a small fixed set of action types
      - element_logits: logits over positions in the current element list
    
    Training code is responsible for providing appropriate targets:
      - action_type_targets: [batch_size]
      - element_index_targets: [batch_size]
    """
    
    def __init__(
        self,
        state_dim: int,
        num_action_types: int,
        max_elements: int,
        hidden_dims: list = [256, 128],
    ):
        """
        Initialize MLP policy.
        
        Args:
            state_dim: Dimension of state encoding
            num_action_types: Number of discrete action types (e.g., click/type/check/submit/wait)
            max_elements: Maximum number of elements considered in the state encoding
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.num_action_types = num_action_types
        self.max_elements = max_elements
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for action type and element selection
        self.action_type_head = nn.Linear(prev_dim, num_action_types)
        # We produce logits for max_elements positions; any unused tail positions
        # can be masked out by the caller if needed.
        self.element_head = nn.Linear(prev_dim, max_elements)
    
    def forward(self, state: torch.Tensor):
        """
        Forward pass: returns compositional action logits.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            action_type_logits: [batch_size, num_action_types]
            element_logits: [batch_size, max_elements]
        """
        features = self.encoder(state)
        action_type_logits = self.action_type_head(features)
        element_logits = self.element_head(features)
        return action_type_logits, element_logits

