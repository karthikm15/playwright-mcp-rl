"""Behavior cloning trainer."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any


class TrajectoryDataset(Dataset):
    """Dataset for behavior cloning from trajectories."""
    
    def __init__(self, state_encoder, action_encoder, trajectories):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.states: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        
        # Extract (state, action) pairs
        for traj in trajectories:
            observations = traj.get('observations', [])
            actions = traj.get('actions', [])
            
            for i, action in enumerate(actions):
                if i < len(observations):
                    state = observations[i]
                    self.states.append(state)
                    self.actions.append(action)
        
        print(f"Created dataset with {len(self.states)} (state, action) pairs")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        """
        Return encoded state and compositional action targets.
        
        - state_tensor: encoded state
        - action_type_idx: index into fixed action type set
        - element_index: index into current state's element list
        """
        raw_state = self.states[idx]
        raw_action = self.actions[idx]
        
        # Encode state
        state_tensor = self.state_encoder.encode_snapshot(raw_state)
        
        # Use raw snapshot elements for element indexing
        elements = raw_state.get('elements', [])
        action_type_idx, element_index = self.action_encoder.encode(
            raw_action, elements
        )
        
        return state_tensor, torch.tensor(action_type_idx, dtype=torch.long), torch.tensor(element_index, dtype=torch.long)


class BCTrainer:
    """Behavior cloning trainer."""
    
    def __init__(self, policy, state_encoder, action_encoder, config):
        """
        Initialize BC trainer.
        
        Args:
            policy: Policy network
            state_encoder: State encoder
            action_encoder: Action encoder
            config: Training config dict
        """
        self.policy = policy
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.config = config
        
        # Combine parameters from policy and state encoder
        params = list(policy.parameters()) + list(state_encoder.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.get('learning_rate', 1e-3)
        )
        # Separate losses for action type and element index
        self.action_type_criterion = nn.CrossEntropyLoss()
        self.element_criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.state_encoder.to(self.device)
    
    def train(self, trajectories: List[Dict[str, Any]]):
        """
        Train policy on trajectories.
        
        Args:
            trajectories: List of trajectory dicts
        """
        # Action vocabulary should already be built before creating trainer
        # Don't rebuild it here to avoid index mismatches
        
        # Create dataset
        dataset = TrajectoryDataset(
            self.state_encoder, self.action_encoder, trajectories
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        num_epochs = self.config.get('num_epochs', 100)
        
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for states, action_type_targets, element_targets in dataloader:
                states = states.to(self.device)
                action_type_targets = action_type_targets.to(self.device)
                element_targets = element_targets.to(self.device)
                
                # Forward pass
                action_type_logits, element_logits = self.policy(states)
                loss_type = self.action_type_criterion(
                    action_type_logits, action_type_targets
                )
                loss_element = self.element_criterion(
                    element_logits, element_targets
                )
                loss = loss_type + loss_element
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        print("Training complete!")

