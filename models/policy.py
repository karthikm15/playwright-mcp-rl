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
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ):
        """
        Initialize MLP policy.
        
        Args:
            state_dim: Dimension of state encoding
            num_action_types: Number of discrete action types (e.g., click/type/check/submit/wait)
            max_elements: Maximum number of elements considered in the state encoding
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.num_action_types = num_action_types
        self.max_elements = max_elements
        self.use_layer_norm = use_layer_norm
        
        # Build encoder with optional layer norm and dropout
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add layer normalization for better training stability
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.ReLU())
            
            # Add dropout for regularization (important with limited data)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Separate heads for action type and element selection
        # Use slightly larger hidden dim for heads to allow more capacity
        head_hidden = prev_dim // 2 if prev_dim > 64 else prev_dim
        
        # Action type head with intermediate layer for better representation
        self.action_type_mlp = nn.Sequential(
            nn.Linear(prev_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5) if dropout > 0 else nn.Identity(),
        )
        self.action_type_head = nn.Linear(head_hidden, num_action_types)
        
        # Element selection head with intermediate layer
        self.element_mlp = nn.Sequential(
            nn.Linear(prev_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5) if dropout > 0 else nn.Identity(),
        )
        # We produce logits for max_elements positions; any unused tail positions
        # can be masked out by the caller if needed.
        self.element_head = nn.Linear(head_hidden, max_elements)
    
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
        
        # Process through separate heads
        action_type_features = self.action_type_mlp(features)
        action_type_logits = self.action_type_head(action_type_features)
        
        element_features = self.element_mlp(features)
        element_logits = self.element_head(element_features)
        
        return action_type_logits, element_logits

