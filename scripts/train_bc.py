"""Train behavior cloning baseline."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

# Add parent directory to path BEFORE importing local modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.transformer_state_encoder import TransformerStateEncoder
from models.policy import MLPPolicy
from utils.action_encoder import ActionEncoder
from training.bc_trainer import BCTrainer


def load_trajectories(trajectory_paths: List[str]) -> List[Dict[str, Any]]:
    """Load trajectories from JSON files."""
    trajectories = []
    for path in trajectory_paths:
        with open(path, 'r') as f:
            traj = json.load(f)
            trajectories.append(traj)
        print(f"Loaded {path}: {len(traj.get('actions', []))} actions")
    return trajectories


def main():
    """Main training function."""
    # Trajectory files
    demo_dir = Path('data/trajectories')
    trajectory_files = list(demo_dir.glob('*.json'))
    
    # Check which files exist
    existing_files = [f for f in trajectory_files if f.exists()]
    if not existing_files:
        print(f"No trajectory files found in {demo_dir}")
        print(f"Looking for: {[f.name for f in trajectory_files]}")
        return
    
    print(f"Loading {len(existing_files)} trajectory files...")
    trajectories = load_trajectories([str(f) for f in existing_files])
    
    # Initialize encoders
    max_elements = 50
    state_encoder = TransformerStateEncoder(max_elements=max_elements, d_model=128)
    action_encoder = ActionEncoder()
    
    # Initialize policy
    state_dim = state_encoder.get_state_dim()
    num_action_types = action_encoder.get_num_action_types()
    
    print("\nPolicy architecture:")
    print(f"  State dimension: {state_dim}")
    print(f"  Num action types: {num_action_types}")
    print(f"  Max elements: {max_elements}")
    
    policy = MLPPolicy(
        state_dim=state_dim,
        num_action_types=num_action_types,
        max_elements=max_elements,
        hidden_dims=[256, 128],
    )
    
    # Training config
    # With limited data (62 samples), use smaller batch size and lower learning rate
    config = {
        'learning_rate': 1e-3,  # Reduced from 1e-2 for more stable training
        'batch_size': 16,  # Smaller batch size for better gradient estimates with limited data
        'num_epochs': 400,
        'gradient_clip': 1.0,  # Clip gradients to prevent explosion
        'freeze_encoder_epochs': 25,  # Freeze state encoder for first N epochs (25 = ~100 batches with 62 samples)
        'adaptive_unfreeze': True,  # Unfreeze when policy loss drops below threshold
        'unfreeze_loss_threshold': 1.5,  # Unfreeze when avg_loss < this value
    }
    
    # Train
    trainer = BCTrainer(policy, state_encoder, action_encoder, config)
    trainer.train(trajectories)
    
    # Save model
    model_dir = Path('models/checkpoints')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'bc_policy_compositional.pt'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'state_encoder_state_dict': state_encoder.state_dict(),
        'action_encoder': {
            # Save action type list for readability / potential extension
            'action_types': action_encoder.action_types,
        },
        'state_dim': state_dim,
        'num_action_types': num_action_types,
        'max_elements': max_elements,
    }, model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()

