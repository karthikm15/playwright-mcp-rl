"""Action encoder for converting actions to indices."""

from typing import Dict, Any, List
import json


class ActionEncoder:
    """Maps actions to discrete indices."""
    
    def __init__(self):
        """Initialize action encoder."""
        self.action_to_idx = {}
        self.idx_to_action = []
        self.next_idx = 0
    
    def build_vocab(self, trajectories: List[Dict[str, Any]]):
        """
        Build action vocabulary from trajectories.
        
        Args:
            trajectories: List of trajectory dicts
        """
        seen_actions = set()
        
        for traj in trajectories:
            for action in traj.get('actions', []):
                # Create action key: type + element_ref
                action_key = (action.get('type', ''), action.get('element_ref', ''))
                if action_key not in seen_actions:
                    seen_actions.add(action_key)
                    self.action_to_idx[action_key] = self.next_idx
                    self.idx_to_action.append(action_key)
                    self.next_idx += 1
        
        print(f"Built action vocabulary with {len(self.action_to_idx)} unique actions")
    
    def encode(self, action: Dict[str, Any]) -> int:
        """
        Encode action to index.
        
        Args:
            action: Action dict
        
        Returns:
            action_idx: Integer index (0 to action_dim-1)
        """
        action_key = (action.get('type', ''), action.get('element_ref', ''))
        idx = self.action_to_idx.get(action_key, 0)
        # Ensure index is within bounds
        if idx >= len(self.idx_to_action):
            return 0
        return idx
    
    def decode(self, action_idx: int) -> Dict[str, Any]:
        """
        Decode index to action dict.
        
        Args:
            action_idx: Integer index
        
        Returns:
            action: Action dict
        """
        if action_idx < len(self.idx_to_action):
            action_type, element_ref = self.idx_to_action[action_idx]
            return {
                'type': action_type,
                'element_ref': element_ref
            }
        return {'type': 'wait', 'element_ref': ''}
    
    def get_action_dim(self) -> int:
        """Get number of possible actions."""
        return len(self.action_to_idx)

