"""Run trained policy in real environment."""

import asyncio
import json
import sys
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from models.policy import MLPPolicy, PPOPolicy
from utils.transformer_state_encoder import TransformerStateEncoder
from utils.action_encoder import ActionEncoder
from env.browser_env import BrowserEnv


def load_model(model_path: str, policy_type: str = 'bc'):
    """
    Load trained model and encoders.
    
    Args:
        model_path: Path to model checkpoint
        policy_type: 'bc' for Behavior Cloning (MLPPolicy) or 'ppo' for PPO (PPOPolicy)
    
    Returns:
        policy, state_encoder, action_encoder
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Reconstruct encoders
    # Use saved max_elements if available; fallback to 50
    max_elements = checkpoint.get('max_elements', 50)
    state_encoder = TransformerStateEncoder(max_elements=max_elements, d_model=128)
    state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
    state_encoder.eval()
    
    action_encoder = ActionEncoder()
    # If custom action types were saved, use them (otherwise defaults are fine)
    saved_action_types = checkpoint.get('action_encoder', {}).get('action_types')
    if saved_action_types:
        action_encoder.action_types = saved_action_types
        action_encoder.type_to_idx = {
            t: i for i, t in enumerate(saved_action_types)
        }
        action_encoder.idx_to_type = {
            i: t for i, t in enumerate(saved_action_types)
        }
    
    # Reconstruct policy based on type
    state_dim = checkpoint['state_dim']
    num_action_types = checkpoint.get(
        'num_action_types', action_encoder.get_num_action_types()
    )
    max_elements = checkpoint.get('max_elements', max_elements)
    
    if policy_type.lower() == 'ppo':
        policy = PPOPolicy(
            state_dim=state_dim,
            num_action_types=num_action_types,
            max_elements=max_elements,
            hidden_dims=[256, 128],
        )
    else:  # default to BC
        policy = MLPPolicy(
            state_dim=state_dim,
            num_action_types=num_action_types,
            max_elements=max_elements,
            hidden_dims=[256, 128],
        )
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    return policy, state_encoder, action_encoder


def extract_elements_from_snapshot(snapshot: Dict[str, Any]) -> list:
    """
    Extract interactive elements from a BrowserEnv snapshot.
    
    The live env returns a root node (e.g., WebArea) with its content
    under 'children', not 'elements'. We treat every node that has a
    'ref' as an element and flatten the tree.
    """
    elements = []
    
    def collect(node):
        if not isinstance(node, dict):
            return
        # Any node with a 'ref' is considered an element
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


def display_action(action: Dict[str, Any], step: int):
    """Display action in readable format."""
    action_type = action.get('type', 'unknown')
    element_ref = action.get('element_ref', '')
    name = action.get('name', '')
    text = action.get('text', '')
    
    print(f"\n[Step {step}] Action: {action_type.upper()}")
    if name:
        print(f"  Element: {name}")
    if element_ref:
        print(f"  Ref: {element_ref}")
    if text:
        print(f"  Text: '{text}'")


async def run_policy(task_config_path: str, model_path: str, headless: bool = True, policy_type: str = 'bc'):
    """
    Run trained policy in environment.
    
    Args:
        task_config_path: Path to task configuration JSON
        model_path: Path to model checkpoint
        headless: Whether to run browser in headless mode
        policy_type: 'bc' for Behavior Cloning or 'ppo' for PPO
    """
    # Load task config
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Load model
    print(f"Loading {policy_type.upper()} model from {model_path}...")
    policy, state_encoder, action_encoder = load_model(model_path, policy_type=policy_type)
    print(f"Model loaded ({policy_type.upper()}) with compositional action space:")
    print(f"  Action types: {action_encoder.action_types}")
    
    # Create environment
    env = BrowserEnv(task_config, headless=headless)
    
    try:
        # Reset environment
        print(f"\nNavigating to: {task_config['url']}")
        state = await env.reset()
        
        step = 0
        done = False
        total_reward = 0.0
        last_action_ref = None
        
        print("\n" + "="*80)
        print("RUNNING POLICY")
        print("="*80)
        
        # Show available actions
        print("\nAvailable action types:")
        for i, t in enumerate(action_encoder.action_types):
            print(f"  [{i}] {t}")
        print()
        print("State:", state)
        
        while not done and step < task_config.get('max_steps', 50):
            # Show current state
            elements = extract_elements_from_snapshot(state)
            print(f"\n[Step {step + 1}] Current page has {len(elements)} interactive elements")
            print(f"  Sample elements: {[(e.get('type'), e.get('name', '')[:40], e.get('ref')) for e in elements[:3]]}")
            
            # Encode state (format as snapshot with elements)
            snapshot = {'elements': elements}
            state_tensor = state_encoder.encode_snapshot(snapshot)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            
            # Get action from policy (PPO returns 3 values, BC returns 2)
            with torch.no_grad():
                policy_output = policy(state_tensor)
                if policy_type.lower() == 'ppo':
                    action_type_logits, element_logits, _ = policy_output
                else:
                    action_type_logits, element_logits = policy_output
                action_type_probs = torch.softmax(action_type_logits, dim=1)
                
                # First choose action type
                action_type_idx = torch.argmax(action_type_logits, dim=1).item()
                action_type = action_encoder.action_types[action_type_idx]
                
                # Build a mask over elements based on action type to ensure
                # we only select semantically compatible refs.
                valid_mask = []
                for elem in elements:
                    elem_type = str(elem.get('type', '')).lower()
                    elem_name = str(elem.get('name', '')).lower()
                    
                    if action_type == 'type':
                        # Only textboxes for typing
                        is_valid = elem_type == 'textbox'
                    elif action_type == 'check':
                        # Radios and checkboxes
                        is_valid = elem_type in ('radio', 'checkbox')
                    elif action_type == 'submit':
                        # Submit-like buttons
                        is_valid = (
                            elem_type == 'button'
                            and ('submit' in elem_name or 'send' in elem_name)
                        )
                    elif action_type == 'click':
                        # Generic click: buttons, links, maybe textboxes
                        is_valid = elem_type in ('button', 'link', 'textbox')
                    else:
                        # wait or others: allow any element
                        is_valid = True
                    
                    valid_mask.append(is_valid)
                
                num_elements = max(len(elements), 1)
                valid_mask_tensor = torch.tensor(
                    valid_mask if len(valid_mask) > 0 else [True],
                    dtype=torch.bool,
                    device=element_logits.device,
                )
                # Truncate/extend mask to match available logits slice
                valid_mask_tensor = valid_mask_tensor[:num_elements]
                if valid_mask_tensor.numel() < num_elements:
                    pad = torch.ones(num_elements - valid_mask_tensor.numel(), dtype=torch.bool, device=element_logits.device)
                    valid_mask_tensor = torch.cat([valid_mask_tensor, pad], dim=0)
                
                # Apply mask: set invalid element logits to a very negative value
                masked_element_logits = element_logits[:, :num_elements].clone()
                masked_element_logits[:, ~valid_mask_tensor] = -1e9
                
                element_probs = torch.softmax(masked_element_logits, dim=1)
                element_idx = torch.argmax(masked_element_logits, dim=1).item()

            # Show model predictions
            print("\nModel action type probabilities:")
            for i, t in enumerate(action_encoder.action_types):
                prob = action_type_probs[0][i].item()
                marker = " <-- SELECTED" if i == action_type_idx else ""
                print(f"  [{i}] {t}: {prob:.3f}{marker}")
            
            print("\nModel element selection probabilities:")
            for i in range(len(elements)):
                elem = elements[i]
                prob = element_probs[0][i].item()
                marker = " <-- SELECTED" if i == element_idx else ""
                print(f"  [{i}] {elem.get('type')} | {elem.get('name', '')[:40]} | {elem.get('ref')}: {prob:.3f}{marker}")
            
            # Decode action using current elements
            action_dict = action_encoder.decode(
                action_type_idx, element_idx, elements
            )
            
            # Get element name from current snapshot for display
            elements = extract_elements_from_snapshot(state)
            element_name = next(
                (e['name'] for e in elements if e.get('ref') == action_dict.get('element_ref', '')),
                action_dict.get('element_ref', 'unknown')
            )
            action_dict['name'] = element_name
            action_dict['description'] = element_name
            
            # For type actions, we need to provide text
            # For now, use a placeholder - in practice, you'd need a text generation model
            # or use the most common text from training data
            if action_dict['type'] == 'type':
                # Simple placeholder - in real system, would predict text
                action_dict['text'] = 'test'
                print("  Note: Using placeholder text 'test' for type action")
            
            # Display action
            display_action(action_dict, step + 1)
            
            # Execute action
            try:
                next_state, reward, done, info = await env.step(action_dict)
                total_reward += reward
                
                print(f"  Reward: {reward:.2f}, Done: {done}")
                if info.get('success'):
                    print("  ✓ Success!")
                
                # Check if we're stuck (same action repeatedly)
                if step > 0 and action_dict.get('element_ref') == last_action_ref:
                    print(f"  ⚠ Warning: Repeated action detected (same element {last_action_ref})")
                    # Try to break out by selecting a different element (cyclic)
                    if len(elements) > 1:
                        print("  Trying alternative element...")
                        element_idx = (element_idx + 1) % len(elements)
                        action_dict = action_encoder.decode(
                            action_type_idx, element_idx, elements
                        )
                        element_name = next(
                            (e['name'] for e in elements if e.get('ref') == action_dict.get('element_ref', '')),
                            action_dict.get('element_ref', 'unknown')
                        )
                        action_dict['name'] = element_name
                        action_dict['description'] = element_name
                        if action_dict['type'] == 'type':
                            action_dict['text'] = 'test'
                        print(f"  Alternative action: {action_dict}")
                        # Retry with new action
                        next_state, reward, done, info = await env.step(action_dict)
                        total_reward += reward
                        print(f"  Alternative action - Reward: {reward:.2f}, Done: {done}")
                
                last_action_ref = action_dict.get('element_ref', '')
                state = next_state
                step += 1
                
            except Exception as e:
                print(f"  Error executing action: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Small delay
            await asyncio.sleep(0.5)
        
        # Final results
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {done and info.get('success', False)}")
        print("="*80)
        
    finally:
        await env.close()


if __name__ == '__main__':
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trained policy in browser environment')
    parser.add_argument('task_config', type=str, help='Path to task configuration JSON')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--policy-type', type=str, default='bc', choices=['bc', 'ppo'],
                        help='Type of policy: "bc" for Behavior Cloning or "ppo" for PPO (default: bc)')
    parser.add_argument('--no-headless', action='store_true',
                        help='Run browser in visible mode (default is headless)')
    
    args = parser.parse_args()
    
    headless = not args.no_headless
    
    asyncio.run(run_policy(
        task_config_path=args.task_config,
        model_path=args.model_path,
        headless=headless,
        policy_type=args.policy_type
    ))

