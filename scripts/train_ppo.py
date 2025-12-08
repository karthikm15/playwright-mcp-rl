"""Train policy using PPO on browser environment."""

import asyncio
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from models.policy import PPOPolicy
from utils.transformer_state_encoder import TransformerStateEncoder
from utils.action_encoder import ActionEncoder
from env.browser_env import BrowserEnv
from training.rollout_buffer import RolloutBuffer
from training.ppo_trainer import PPOTrainer


def extract_elements_from_snapshot(snapshot: Dict[str, Any]) -> list:
    """Extract interactive elements from snapshot."""
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


def build_validity_mask(action_type: str, elements: list, max_elements: int, device: str):
    """Build mask for valid elements given action type."""
    valid_mask = []
    for elem in elements:
        elem_type = str(elem.get('type', '')).lower()
        elem_name = str(elem.get('name', '')).lower()
        
        is_valid = False
        if action_type == 'type':
            is_valid = elem_type == 'textbox'
        elif action_type == 'check':
            is_valid = elem_type in ('radio', 'checkbox')
        elif action_type == 'submit':
            is_valid = (elem_type == 'button' and ('submit' in elem_name or 'send' in elem_name))
        elif action_type == 'click':
            is_valid = elem_type in ('button', 'link', 'textbox')
        else:  # Unknown action type (shouldn't happen with current action_types)
            is_valid = True
        valid_mask.append(is_valid)
    
    num_elements = max(len(elements), 1)
    valid_mask_tensor = torch.tensor(
        valid_mask if len(valid_mask) > 0 else [True],
        dtype=torch.bool,
        device=device,
    )
    if valid_mask_tensor.numel() < num_elements:
        pad = torch.ones(num_elements - valid_mask_tensor.numel(), dtype=torch.bool, device=device)
        valid_mask_tensor = torch.cat([valid_mask_tensor, pad], dim=0)
    
    return valid_mask_tensor[:num_elements]


async def collect_rollout(
    env: BrowserEnv,
    policy: PPOPolicy,
    state_encoder: TransformerStateEncoder,
    action_encoder: ActionEncoder,
    buffer: RolloutBuffer,
    device: str,
    max_steps: int = 50,
):
    """Collect a single rollout episode."""
    state = await env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        print(f"Step {step+1}/{max_steps}")
        # Extract elements and encode state
        elements = extract_elements_from_snapshot(state)
        snapshot = {'elements': elements}
        state_tensor = state_encoder.encode_snapshot(snapshot).unsqueeze(0).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action_type_logits, element_logits, value = policy(state_tensor)
            value = value.item()
            print(f"Value: {value}")
            # Sample action type
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            action_type_idx = torch.multinomial(action_type_probs, 1).item()
            action_type = action_encoder.action_types[action_type_idx]
            
            # Build validity mask and sample element
            valid_mask = build_validity_mask(action_type, elements, len(elements), device)
            num_elements = max(len(elements), 1)
            masked_element_logits = element_logits[:, :num_elements].clone()
            masked_element_logits[:, ~valid_mask] = -1e9
            
            element_probs = F.softmax(masked_element_logits, dim=-1)
            element_idx = torch.multinomial(element_probs, 1).item()
            
            # Compute log prob
            action_type_log_probs = F.log_softmax(action_type_logits, dim=-1)
            element_log_probs = F.log_softmax(masked_element_logits, dim=-1)
            log_prob = action_type_log_probs[0, action_type_idx].item() + element_log_probs[0, element_idx].item()
        
        # Decode action
        action_dict = action_encoder.decode(action_type_idx, element_idx, elements)
        print(f"Action: {action_dict}")
        # Add text for type actions
        if action_type == 'type' and elements[element_idx].get('type', '').lower() == 'textbox':
            action_dict['text'] = 'Test User'  # Simple default text
        
        # Store current state in buffer BEFORE executing action (standard RL: store s_t, a_t)
        buffer.add(
            state=state,
            action_type=action_type_idx,
            action_element=element_idx,
            reward=0.0,  # Placeholder, will be updated after step
            done=False,  # Placeholder, will be updated after step
            log_prob=log_prob,
            value=value,
        )
        
        # Step environment - this executes the action and returns (s_{t+1}, r_t, done, info)
        next_state, reward, done, info = await env.step(action_dict)
        
        # Update the reward and done for the last stored step (r_t corresponds to action a_t)
        if len(buffer.rewards) > 0:
            buffer.rewards[-1] = reward
            buffer.dones[-1] = done
        
        # Update state to the new state returned from environment
        state = next_state
        step += 1
    
    # Get final value if episode didn't terminate
    if not done:
        elements = extract_elements_from_snapshot(state)
        snapshot = {'elements': elements}
        state_tensor = state_encoder.encode_snapshot(snapshot).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, final_value = policy(state_tensor)
            final_value = final_value.item()
    else:
        final_value = 0.0
    
    return step, done, info.get('success', False), final_value


async def train_ppo(
    task_config_path: str,
    num_rollouts: int,
    num_updates: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_elements: int,
    device: str,
    max_steps: int,
    headless: bool,
    save_path: str,
):
    """Train policy using PPO."""
    # Load task
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Initialize components
    state_encoder = TransformerStateEncoder(max_elements=max_elements, d_model=128)
    action_encoder = ActionEncoder()
    state_dim = state_encoder.get_state_dim()
    
    policy = PPOPolicy(
        state_dim=state_dim,
        num_action_types=action_encoder.get_num_action_types(),
        max_elements=max_elements,
        hidden_dims=[256, 128],
    )
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    trainer = PPOTrainer(
        policy=policy,
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        optimizer=optimizer,
        device=device,
    )
    
    # Create environment
    env = BrowserEnv(task_config, headless=headless)
    
    buffer = RolloutBuffer()
    
    print(f"Training PPO on task: {task_config['url']}")
    print(f"Device: {device}, Max elements: {max_elements}")
    print(f"Rollouts per update: {num_rollouts}, Updates: {num_updates}")
    print("-" * 60)
    
    for update in range(num_updates):
        # Collect rollouts
        total_steps = 0
        total_success = 0
        buffer.clear()
        
        for rollout_idx in range(num_rollouts):
            print(f"Collecting rollout {rollout_idx+1}/{num_rollouts}")
            steps, done, success, final_value = await collect_rollout(
                env, policy, state_encoder, action_encoder, buffer, device,
                max_steps=max_steps,
            )
            total_steps += steps
            print(f"Steps: {steps}")
            if success:
                total_success += 1
            
            # For non-terminated episodes, update the last stored value with bootstrap value
            # This is a simplification - ideally we'd track per-rollout, but works for minimal impl
            if not done and len(buffer.values) > 0:
                buffer.values[-1] = final_value
        
        # Update policy
        if len(buffer) > 0:
            # Use 0.0 as last_value since we've already updated buffer.values for non-terminated episodes
            metrics = trainer.update(buffer, num_epochs=num_epochs, batch_size=batch_size, last_value=0.0)
            
            print(f"Update {update+1}/{num_updates}: "
                  f"Success rate: {total_success}/{num_rollouts} ({100*total_success/num_rollouts:.1f}%), "
                  f"Avg steps: {total_steps/num_rollouts:.1f}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Policy loss: {metrics['policy_loss']:.4f}, "
                  f"Value loss: {metrics['value_loss']:.4f}")
        else:
            print(f"Update {update+1}/{num_updates}: No data collected")
        
        # Save checkpoint
        if (update + 1) % 10 == 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'state_encoder_state_dict': state_encoder.state_dict(),
                'state_dim': state_dim,
                'num_action_types': action_encoder.get_num_action_types(),
                'max_elements': max_elements,
                'action_encoder': {'action_types': action_encoder.action_types},
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    await env.close()
    
    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'state_encoder_state_dict': state_encoder.state_dict(),
        'state_dim': state_dim,
        'num_action_types': action_encoder.get_num_action_types(),
        'max_elements': max_elements,
        'action_encoder': {'action_types': action_encoder.action_types},
    }, save_path)
    print(f"\nTraining complete! Saved final model to {save_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='data/tasks/example_single_field.json')
    parser.add_argument('--num-rollouts', type=int, default=20)
    parser.add_argument('--num-updates', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=15)    
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-elements', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in visible mode (default is headless)')
    parser.add_argument('--save-path', type=str, default='models/checkpoints/ppo_policy.pt')
    
    args = parser.parse_args()
    
    # Default to headless=True, only set to False if --no-headless is provided
    headless = not args.no_headless
    
    asyncio.run(train_ppo(
        task_config_path=args.task,
        num_rollouts=args.num_rollouts,
        num_updates=args.num_updates,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_elements=args.max_elements,
        device=args.device,
        max_steps=args.max_steps,
        headless=headless,
        save_path=args.save_path,
    ))

