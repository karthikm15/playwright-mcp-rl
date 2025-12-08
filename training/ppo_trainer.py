"""PPO trainer for policy gradient RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(
        self,
        policy: nn.Module,
        state_encoder: nn.Module,
        action_encoder: Any,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Policy network (with value head)
            state_encoder: State encoder
            action_encoder: Action encoder
            optimizer: Optimizer for policy
            device: Device to run on
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
        """
        self.policy = policy
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.optimizer = optimizer
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy.to(device)
        self.state_encoder.to(device)
    
    def update(self, buffer: RolloutBuffer, num_epochs: int = 4, batch_size: int = 64, last_value: float = 0.0):
        """
        Update policy using PPO on collected rollouts.
        
        Args:
            buffer: RolloutBuffer with collected trajectories
            num_epochs: Number of update epochs
            batch_size: Batch size for updates
            last_value: Value estimate for state after last step (if episode didn't terminate)
        """
        if len(buffer) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = buffer.compute_advantages(last_value=last_value)
        batch_data = buffer.get_batch(advantages, returns)
        
        # Move to device
        actions_type = batch_data['actions_type'].to(self.device)
        actions_element = batch_data['actions_element'].to(self.device)
        old_log_probs = batch_data['log_probs'].to(self.device)
        advantages = batch_data['advantages'].to(self.device)
        returns = batch_data['returns'].to(self.device)
        
        # Encode states (detach to avoid backprop through graph multiple times)
        states_tensor = []
        with torch.no_grad():
            for state in batch_data['states']:
                # Extract elements from snapshot
                elements = self._extract_elements(state)
                snapshot = {'elements': elements}
                state_vec = self.state_encoder.encode_snapshot(snapshot)
                states_tensor.append(state_vec)
        
        # Stack and detach - states are treated as constants during PPO updates
        states = torch.stack(states_tensor).to(self.device).detach()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Multiple epochs of updates
        for epoch in range(num_epochs):
            # Shuffle for each epoch
            perm = torch.randperm(len(states))
            states_epoch = states[perm]
            actions_type_epoch = actions_type[perm]
            actions_element_epoch = actions_element[perm]
            old_log_probs_epoch = old_log_probs[perm]
            advantages_epoch = advantages[perm]
            returns_epoch = returns[perm]
            
            # Mini-batch updates
            for i in range(0, len(states), batch_size):
                batch_states = states_epoch[i:i+batch_size]
                batch_actions_type = actions_type_epoch[i:i+batch_size]
                batch_actions_element = actions_element_epoch[i:i+batch_size]
                batch_old_log_probs = old_log_probs_epoch[i:i+batch_size]
                batch_advantages = advantages_epoch[i:i+batch_size]
                batch_returns = returns_epoch[i:i+batch_size]
                
                # Forward pass
                action_type_logits, element_logits, values = self.policy(batch_states)
                values = values.squeeze(-1)
                
                # Compute action probabilities
                action_type_probs = F.softmax(action_type_logits, dim=-1)
                element_probs = F.softmax(element_logits, dim=-1)
                
                # Get log probs for selected actions
                action_type_log_probs = F.log_softmax(action_type_logits, dim=-1)
                element_log_probs = F.log_softmax(element_logits, dim=-1)
                
                log_probs_type = action_type_log_probs.gather(1, batch_actions_type.unsqueeze(1)).squeeze(1)
                log_probs_element = element_log_probs.gather(1, batch_actions_element.unsqueeze(1)).squeeze(1)
                
                # Combined log prob (sum of type and element)
                new_log_probs = log_probs_type + log_probs_element
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_type = -(action_type_probs * action_type_log_probs).sum(dim=-1).mean()
                entropy_element = -(element_probs * element_log_probs).sum(dim=-1).mean()
                entropy = entropy_type + entropy_element
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        num_updates = num_epochs * (len(states) // batch_size + (1 if len(states) % batch_size else 0))
        
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def _extract_elements(self, snapshot: Dict[str, Any]) -> list:
        """Extract elements from snapshot (same logic as run_policy)."""
        elements = []
        
        def collect(node):
            if not isinstance(node, dict):
                return
            if 'ref' in node:
                elements.append(node)
            for value in node.values():
                if isinstance(value, dict):
                    collect(value)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
        
        collect(snapshot)
        return elements

