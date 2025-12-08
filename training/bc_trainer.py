"""Behavior cloning trainer."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from tqdm import tqdm


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
        self.policy_params = list(policy.parameters())
        self.encoder_params = list(state_encoder.parameters())
        params = self.policy_params + self.encoder_params
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.get('learning_rate', 1e-3),
            weight_decay=1e-5  # Small weight decay for regularization
        )
        self.gradient_clip = config.get('gradient_clip', None)
        self.freeze_encoder_epochs = config.get('freeze_encoder_epochs', 0)
        self.adaptive_unfreeze = config.get('adaptive_unfreeze', False)
        self.unfreeze_loss_threshold = config.get('unfreeze_loss_threshold', 1.5)
        self.encoder_frozen = True  # Track if encoder is currently frozen
        
        # Learning rate scheduler - reduce LR when loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
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
        print(f"Dataset size: {len(dataset)} samples, Batch size: {self.config.get('batch_size', 32)}")
        print(f"Total batches per epoch: {len(dataloader)}\n")
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        
        # Freeze/unfreeze encoder based on config
        for param in self.encoder_params:
            param.requires_grad = (self.freeze_encoder_epochs == 0)
        
        for epoch in epoch_pbar:
            # Unfreeze encoder logic
            should_unfreeze = False
            unfreeze_reason = ""
            
            if self.encoder_frozen:
                # Check if we should unfreeze based on epoch count
                if epoch >= self.freeze_encoder_epochs:
                    should_unfreeze = True
                    unfreeze_reason = f"reached epoch {epoch + 1}"
                # Or check if we should unfreeze adaptively based on loss
                elif self.adaptive_unfreeze and epoch > 0:
                    # We'll check loss after computing it
                    pass
            
            total_loss = 0.0
            total_loss_type = 0.0
            total_loss_element = 0.0
            num_batches = 0
            total_loss = 0.0
            total_loss_type = 0.0
            total_loss_element = 0.0
            num_batches = 0
            
            # Progress bar for batches within each epoch
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False, unit="batch")
            
            for states, action_type_targets, element_targets in batch_pbar:
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
                
                # Gradient clipping to prevent explosion
                if self.gradient_clip is not None:
                    # Get all parameters that require gradients
                    all_params = [p for p in self.policy.parameters() if p.requires_grad]
                    all_params += [p for p in self.state_encoder.parameters() if p.requires_grad]
                    if all_params:
                        torch.nn.utils.clip_grad_norm_(all_params, self.gradient_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_loss_type += loss_type.item()
                total_loss_element += loss_element.item()
                num_batches += 1
                
                # Update batch progress bar with current loss
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_loss_type = total_loss_type / num_batches if num_batches > 0 else 0.0
            avg_loss_element = total_loss_element / num_batches if num_batches > 0 else 0.0
            
            # Check adaptive unfreeze based on loss
            if self.encoder_frozen and self.adaptive_unfreeze and epoch > 0:
                if avg_loss < self.unfreeze_loss_threshold:
                    should_unfreeze = True
                    unfreeze_reason = f"loss dropped to {avg_loss:.4f} < {self.unfreeze_loss_threshold}"
            
            # Unfreeze encoder if conditions are met
            if should_unfreeze and self.encoder_frozen:
                for param in self.encoder_params:
                    param.requires_grad = True
                # Recreate optimizer with encoder params
                params = self.policy_params + self.encoder_params
                self.optimizer = torch.optim.Adam(
                    params,
                    lr=self.config.get('learning_rate', 1e-3),
                    weight_decay=1e-5
                )
                # Recreate scheduler for new optimizer
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
                )
                self.encoder_frozen = False
                print(f"\n✓ Unfreezing state encoder at epoch {epoch + 1} ({unfreeze_reason})")
            
            # Update learning rate scheduler
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update epoch progress bar
            frozen_indicator = " [FROZEN]" if self.encoder_frozen else ""
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'loss_type': f'{avg_loss_type:.4f}',
                'loss_element': f'{avg_loss_element:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            epoch_pbar.set_description(f"Training{frozen_indicator}")
        
        epoch_pbar.close()
        print("\n✓ Training complete!")

