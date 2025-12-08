"""Action encoder for compositional actions (action_type, element_index)."""

from typing import Dict, Any, List, Tuple


class ActionEncoder:
    """
    Encodes actions in a compositional way:
      - action_type: one of a small fixed set of types
      - element_index: index into the current state's element list
    
    This avoids baking element_ref IDs into a fixed vocabulary and
    allows the policy to generalize to new forms with different refs.
    """
    
    def __init__(self):
        """Initialize action encoder."""
        # Fixed action types
        self.action_types: List[str] = ['click', 'type', 'check', 'submit']
        self.type_to_idx: Dict[str, int] = {
            t: i for i, t in enumerate(self.action_types)
        }
        self.idx_to_type: Dict[int, str] = {
            i: t for i, t in enumerate(self.action_types)
        }
    
    def get_num_action_types(self) -> int:
        """Return number of discrete action types."""
        return len(self.action_types)
    
    def encode(
        self,
        action: Dict[str, Any],
        current_elements: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Encode an action into (action_type_idx, element_index).
        
        Args:
            action: Action dict with 'type' and 'element_ref'
            current_elements: List of elements from current snapshot
        
        Returns:
            (action_type_idx, element_index)
        """
        action_type = action.get('type', 'click')
        action_type_idx = self.type_to_idx.get(action_type, self.type_to_idx['click'])
        
        # Map element_ref to index in current_elements
        element_ref = action.get('element_ref', '')
        element_index = 0  # default to first element if not found / empty
        for idx, elem in enumerate(current_elements):
            if elem.get('ref') == element_ref:
                element_index = idx
                break
        
        return action_type_idx, element_index
    
    def decode(
        self,
        action_type_idx: int,
        element_index: int,
        current_elements: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Decode (action_type_idx, element_index) back to action dict.
        
        Args:
            action_type_idx: Index of action type
            element_index: Index into current_elements
            current_elements: List of elements from current snapshot
        
        Returns:
            action dict with 'type' and 'element_ref'
        """
        action_type = self.idx_to_type.get(
            action_type_idx, 'click'
        )
        
        element_ref = ''
        if 0 <= element_index < len(current_elements):
            element_ref = current_elements[element_index].get('ref', '')
        
        return {
            'type': action_type,
            'element_ref': element_ref,
        }

