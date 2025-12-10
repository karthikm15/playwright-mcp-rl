"""PPO trainer for policy gradient RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from pathlib import Path
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
        
        # Loss tracking for plotting
        self.loss_history: List[float] = []
        self.policy_loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.update_count = 0
        
        # Action type probability tracking
        self.action_type_probs_history: List[Dict[str, float]] = []  # List of dicts: {action_type: prob}
        
        # Element choice tracking (top N elements by selection frequency)
        self.element_choices_history: List[Dict[str, int]] = []  # List of dicts: {element_ref: count}
        
        # Additional metrics tracking
        self.avg_reward_history: List[float] = []  # Average reward/return per update
        self.kl_divergence_history: List[float] = []  # KL divergence per update
        self.entropy_history: List[float] = []  # Entropy per update
        
        # Plot file paths
        self.loss_file = Path('loss.png')
        self.policy_loss_file = Path('policy_loss.png')
        self.value_loss_file = Path('value_loss.png')
        self.action_type_probs_file = Path('action_type_probs.png')
        self.element_choices_file = Path('element_choices.png')
        self.avg_reward_file = Path('avg_reward.png')
        self.kl_divergence_file = Path('kl_divergence.png')
        self.entropy_file = Path('entropy.png')
    
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
        total_kl_div = 0.0
        
        # Track average return from buffer
        avg_return = returns.mean().item() if len(returns) > 0 else 0.0
        
        # Track action type probabilities and element choices for this update
        action_type_probs_sum = torch.zeros(len(self.action_encoder.action_types), device=self.device)
        action_type_count = 0
        element_choice_counts: Dict[str, int] = {}
        
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
                
                # Track average action type probabilities
                action_type_probs_sum += action_type_probs.mean(dim=0)
                action_type_count += 1
                
                # Track element choices (selected elements in this batch)
                selected_elements = batch_actions_element.cpu().numpy()
                for elem_idx in selected_elements:
                    # Track by element index (shows which positions are selected most)
                    elem_key = f"Element {int(elem_idx)}"
                    element_choice_counts[elem_key] = element_choice_counts.get(elem_key, 0) + 1
                
                # Get log probs for selected actions
                action_type_log_probs = F.log_softmax(action_type_logits, dim=-1)
                element_log_probs = F.log_softmax(element_logits, dim=-1)
                
                log_probs_type = action_type_log_probs.gather(1, batch_actions_type.unsqueeze(1)).squeeze(1)
                log_probs_element = element_log_probs.gather(1, batch_actions_element.unsqueeze(1)).squeeze(1)
                
                # Combined log prob (sum of type and element)
                new_log_probs = log_probs_type + log_probs_element
                
                # Compute KL divergence: KL(old || new) = E[log(old) - log(new)]
                # Using old_log_probs and new_log_probs
                kl_div = (batch_old_log_probs - new_log_probs).mean()
                total_kl_div += kl_div.item()
                
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
        
        # Average losses and metrics
        avg_loss = total_loss / num_updates
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_kl_div = total_kl_div / num_updates
        
        # Track losses and metrics
        self.loss_history.append(avg_loss)
        self.policy_loss_history.append(avg_policy_loss)
        self.value_loss_history.append(avg_value_loss)
        self.entropy_history.append(avg_entropy)
        self.kl_divergence_history.append(avg_kl_div)
        self.avg_reward_history.append(avg_return)
        self.update_count += 1
        
        # Track action type probabilities (average over all batches in this update)
        if action_type_count > 0:
            avg_action_type_probs = (action_type_probs_sum / action_type_count).detach().cpu().numpy()
            action_type_probs_dict = {
                action_type: float(avg_action_type_probs[i])
                for i, action_type in enumerate(self.action_encoder.action_types)
            }
            self.action_type_probs_history.append(action_type_probs_dict)
        
        # Track element choices (top elements selected in this update)
        self.element_choices_history.append(element_choice_counts.copy())
        
        # Update all plots
        self._plot_all_losses()
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl_div,
            'avg_reward': avg_return,
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
    
    def _plot_all_losses(self):
        """Plot and save all loss and statistics curves."""
        if len(self.loss_history) == 0:
            return
        
        updates = range(1, len(self.loss_history) + 1)
        
        # 1. Total Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(updates, self.loss_history, 'red', linewidth=1.5)
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.loss_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Policy Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(updates, self.policy_loss_history, 'red', linewidth=1.5)
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.policy_loss_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. Value Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(updates, self.value_loss_history, 'red', linewidth=1.5)
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.value_loss_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 4. Action Type Probabilities
        if len(self.action_type_probs_history) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            for action_type in self.action_encoder.action_types:
                probs = [hist.get(action_type, 0.0) for hist in self.action_type_probs_history]
                ax.plot(updates[:len(probs)], probs, label=action_type, linewidth=1.5)
            ax.set_xlabel('Update')
            ax.set_ylabel('Probability')
            ax.set_title('Action Type Probabilities')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(self.action_type_probs_file, dpi=100, bbox_inches='tight')
            plt.close()
        
        # 5. Element Choices (Top N elements over time)
        if len(self.element_choices_history) > 0:
            # Collect all unique element keys
            all_element_keys = set()
            for hist in self.element_choices_history:
                all_element_keys.update(hist.keys())
            
            # Get top N most frequently selected elements overall
            element_totals = {}
            for hist in self.element_choices_history:
                for elem_key, count in hist.items():
                    element_totals[elem_key] = element_totals.get(elem_key, 0) + count
            
            # Sort by total count and take top 10
            top_elements = sorted(element_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            top_element_keys = [elem[0] for elem in top_elements]
            
            if top_element_keys:
                fig, ax = plt.subplots(figsize=(10, 6))
                for elem_key in top_element_keys:
                    counts = [hist.get(elem_key, 0) for hist in self.element_choices_history]
                    ax.plot(updates[:len(counts)], counts, label=elem_key, linewidth=1.5, alpha=0.7)
                ax.set_xlabel('Update')
                ax.set_ylabel('Selection Count')
                ax.set_title('Top Element Choices Over Time')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.element_choices_file, dpi=100, bbox_inches='tight')
                plt.close()
        
        # 6. Average Reward/Return
        if len(self.avg_reward_history) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(updates, self.avg_reward_history, 'red', linewidth=1.5)
            ax.set_xlabel('Update')
            ax.set_ylabel('Average Return')
            ax.set_title('Average Return Over Time')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.avg_reward_file, dpi=100, bbox_inches='tight')
            plt.close()
        
        # 7. KL Divergence
        if len(self.kl_divergence_history) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(updates, self.kl_divergence_history, 'red', linewidth=1.5)
            ax.set_xlabel('Update')
            ax.set_ylabel('KL Divergence')
            ax.set_title('KL Divergence')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.kl_divergence_file, dpi=100, bbox_inches='tight')
            plt.close()
        
        # 8. Entropy
        if len(self.entropy_history) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(updates, self.entropy_history, 'red', linewidth=1.5)
            ax.set_xlabel('Update')
            ax.set_ylabel('Entropy')
            ax.set_title('Entropy')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.entropy_file, dpi=100, bbox_inches='tight')
            plt.close()

