"""Train behavior cloning baseline."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

sys.path.append(str(Path(__file__).parent.parent))

from models.policy import MLPPolicy
from utils.state_encoder import StateEncoder
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
    state_encoder = StateEncoder(max_elements=50, element_dim=64)
    action_encoder = ActionEncoder()
    
    # Build action vocabulary first
    action_encoder.build_vocab(trajectories)
    
    # Initialize policy
    state_dim = state_encoder.get_state_dim()
    action_dim = action_encoder.get_action_dim()
    
    print(f"\nPolicy architecture:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    
    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=[256, 128])
    
    # Training config
    config = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 200
    }
    
    # Train
    trainer = BCTrainer(policy, state_encoder, action_encoder, config)
    trainer.train(trajectories)
    
    # Save model
    model_dir = Path('models/checkpoints')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'bc_policy.pt'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'state_encoder_state_dict': state_encoder.state_dict(),
        'action_encoder': {
            'action_to_idx': action_encoder.action_to_idx,
            'idx_to_action': action_encoder.idx_to_action
        },
        'state_dim': state_dim,
        'action_dim': action_dim
    }, model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()

