"""Run trained policy in real environment."""

import asyncio
import json
import sys
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from models.policy import MLPPolicy
from utils.state_encoder import StateEncoder
from utils.action_encoder import ActionEncoder
from env.browser_env import BrowserEnv


def load_model(model_path: str):
    """Load trained model and encoders."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Reconstruct encoders
    state_encoder = StateEncoder(max_elements=50, element_dim=64)
    state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
    state_encoder.eval()
    
    action_encoder = ActionEncoder()
    action_encoder.action_to_idx = checkpoint['action_encoder']['action_to_idx']
    action_encoder.idx_to_action = checkpoint['action_encoder']['idx_to_action']
    
    # Reconstruct policy
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=[256, 128])
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    return policy, state_encoder, action_encoder


def extract_elements_from_snapshot(snapshot: Dict[str, Any]) -> list:
    """Extract elements from snapshot for display."""
    elements = snapshot.get('elements', [])
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


async def run_policy(task_config_path: str, model_path: str, headless: bool = True):
    """Run trained policy in environment."""
    # Load task config
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Load model
    print(f"Loading model from {model_path}...")
    policy, state_encoder, action_encoder = load_model(model_path)
    print(f"Model loaded: {action_encoder.get_action_dim()} actions")
    
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
        print("\nAvailable actions in vocabulary:")
        for i, (action_type, element_ref) in enumerate(action_encoder.idx_to_action):
            print(f"  [{i}] {action_type} on {element_ref}")
        print()
        
        while not done and step < task_config.get('max_steps', 50):
            # Show current state
            elements = extract_elements_from_snapshot(state)
            print(f"\n[Step {step + 1}] Current page has {len(elements)} interactive elements")
            print(f"  Sample elements: {[(e.get('type'), e.get('name', '')[:40], e.get('ref')) for e in elements[:3]]}")
            
            # Encode state
            state_tensor = state_encoder.encode_snapshot(state)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            
            # Get action from policy
            with torch.no_grad():
                logits = policy(state_tensor)
                probs = torch.softmax(logits, dim=1)
                action_idx = torch.argmax(logits, dim=1).item()
            
            # Show model predictions
            print(f"\nModel action probabilities:")
            for i in range(min(action_encoder.get_action_dim(), 10)):  # Show top 10
                if i < len(action_encoder.idx_to_action):
                    action_type, element_ref = action_encoder.idx_to_action[i]
                    prob = probs[0][i].item()
                    marker = " <-- SELECTED" if i == action_idx else ""
                    print(f"  [{i}] {action_type} on {element_ref}: {prob:.3f}{marker}")
            
            # Decode action
            action_dict = action_encoder.decode(action_idx)
            
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
                print(f"  Note: Using placeholder text 'test' for type action")
            
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
                    # Try to break out by selecting a different action
                    if action_idx < action_encoder.get_action_dim() - 1:
                        print(f"  Trying alternative action...")
                        action_idx = (action_idx + 1) % action_encoder.get_action_dim()
                        action_dict = action_encoder.decode(action_idx)
                        element_name = next(
                            (e['name'] for e in elements if e.get('ref') == action_dict.get('element_ref', '')),
                            action_dict.get('element_ref', 'unknown')
                        )
                        action_dict['name'] = element_name
                        action_dict['description'] = element_name
                        if action_dict['type'] == 'type':
                            action_dict['text'] = 'test'
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
    
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_policy.py <task_config_path> <model_path> [--no-headless]")
        sys.exit(1)
    
    task_config_path = sys.argv[1]
    model_path = sys.argv[2]
    headless = '--no-headless' not in sys.argv
    
    asyncio.run(run_policy(task_config_path, model_path, headless))

