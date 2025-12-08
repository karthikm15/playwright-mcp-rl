"""
Script to update all trajectory JSON files to match the new reward structure
and state representation changes.

Changes:
1. Recompute rewards based on new structure:
   - Base: -0.1 per step
   - +0.2 for each field filled
   - -0.2 for redundant actions
   - +1.0 on success, -1.0 on timeout
2. Add 'checked' field to radio/checkbox elements
3. Filter out irrelevant nodes from observations
4. Remove 'wait' actions
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path


# Irrelevant UI elements to filter out (from browser_env.py)
IRRELEVANT_PATTERNS = [
    'sign in to google',
    'learn more',
    'google forms',
    'Google  Forms',
    'help and feedback',
    'does this form look suspicious',
    'clear form',
    'never submit passwords',
    'report',
    'privacy policy',
    'terms of service',
    'to save your progress',
]


def is_irrelevant_node(node: Dict[str, Any]) -> bool:
    """Check if a node should be filtered out."""
    name = node.get('name', '').lower()
    role = node.get('type', '').lower()
    
    # Check against irrelevant patterns
    for pattern in IRRELEVANT_PATTERNS:
        if pattern.lower() in name:
            return True
    
    # Filter out certain non-interactive roles
    if role in ['generic', 'none'] and not name:
        return True
    
    return False


def filter_irrelevant_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out irrelevant elements from a list."""
    filtered = []
    for elem in elements:
        if not is_irrelevant_node(elem):
            filtered.append(elem)
    return filtered


def add_checked_field(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add 'checked' field to radio/checkbox elements based on their value."""
    updated = []
    for elem in elements:
        elem_type = elem.get('type', '').lower()
        value = str(elem.get('value', '')).strip()
        
        # Add checked field for radio/checkbox
        if elem_type in ['radio', 'checkbox']:
            checked = value.lower() == 'true'
            elem['checked'] = checked
            # Ensure value is "true" or ""
            if checked:
                elem['value'] = 'true'
            else:
                elem['value'] = ''
        
        updated.append(elem)
    return updated


def count_filled_fields(elements: List[Dict[str, Any]]) -> int:
    """Count number of filled fields (textboxes with text, checked radios/checkboxes)."""
    count = 0
    for elem in elements:
        elem_type = elem.get('type', '').lower()
        value_str = str(elem.get('value', '')).strip()
        checked = elem.get('checked', False)
        
        if elem_type == 'textbox' and value_str:
            count += 1
        elif elem_type in ['radio', 'checkbox'] and (checked or value_str.lower() == 'true'):
            count += 1
    return count


def find_element_by_ref(elements: List[Dict[str, Any]], element_ref: str) -> Optional[Dict[str, Any]]:
    """Find an element by its ref in the elements list."""
    for elem in elements:
        if elem.get('ref') == element_ref:
            return elem
    return None


def recompute_rewards(
    observations: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    dones: List[bool],
    task_config: Dict[str, Any]
) -> List[float]:
    """
    Recompute rewards based on the new reward structure.
    
    Reward structure:
    - Base: -0.1 per step
    - +0.2 for each field filled (comparing prev_state to current_state)
    - -0.2 for redundant actions (typing into filled textbox, checking already-checked radio)
    - +1.0 on success (when done=True and success=True)
    - -1.0 on timeout (when done=True but not success)
    """
    rewards = []
    success_condition = task_config.get('success_condition', '')
    
    for i in range(len(actions)):
        reward = -0.1  # Base step penalty
        
        # Get previous and current states
        prev_obs = observations[i] if i < len(observations) else None
        curr_obs = observations[i + 1] if i + 1 < len(observations) else None
        
        if prev_obs and curr_obs:
            prev_elements = prev_obs.get('elements', [])
            curr_elements = curr_obs.get('elements', [])
            
            # Count filled fields
            prev_filled = count_filled_fields(prev_elements)
            curr_filled = count_filled_fields(curr_elements)
            
            # Reward for filling a new field
            if curr_filled > prev_filled:
                reward += 0.2 * (curr_filled - prev_filled)
            
            # Penalty for redundant actions
            action = actions[i]
            action_type = action.get('type', '')
            element_ref = action.get('element_ref', '')
            
            if action_type == 'type':
                # Check if we tried to type into an already-filled textbox
                prev_elem = find_element_by_ref(prev_elements, element_ref)
                if prev_elem and prev_elem.get('type', '').lower() == 'textbox':
                    prev_value = str(prev_elem.get('value', '')).strip()
                    if prev_value:
                        reward -= 0.2  # Penalty for redundant typing
            elif action_type == 'check':
                # Check if we tried to check an already-checked radio/checkbox
                prev_elem = find_element_by_ref(prev_elements, element_ref)
                if prev_elem:
                    prev_checked = prev_elem.get('checked', False) or str(prev_elem.get('value', '')).lower() == 'true'
                    if prev_checked:
                        reward -= 0.2  # Penalty for redundant checking
        
        # Check for success or timeout
        if dones[i]:
            # Check if current observation contains success condition
            # The observation after the action should contain the success message
            if curr_obs:
                # Check if success condition appears anywhere in the observation
                # (matching browser_env.py logic which checks the snapshot string)
                obs_str = json.dumps(curr_obs).lower()
                success = success_condition.lower() in obs_str if success_condition else False
                
                if success:
                    reward = 1.0
                else:
                    # Timeout (reached max steps without success)
                    reward = -1.0
            else:
                # No observation after action - likely timeout
                reward = -1.0
        
        rewards.append(reward)
    
    return rewards


def update_trajectory(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Update a single trajectory with new reward structure and state representation."""
    # Filter irrelevant elements from all observations
    observations = trajectory.get('observations', [])
    if not observations:
        print("  Warning: No observations found, skipping trajectory")
        return trajectory
    
    for obs in observations:
        elements = obs.get('elements', [])
        # Filter irrelevant elements
        filtered_elements = filter_irrelevant_elements(elements)
        # Add checked field to radio/checkbox elements
        updated_elements = add_checked_field(filtered_elements)
        obs['elements'] = updated_elements
    
    # Remove 'wait' actions
    actions = trajectory.get('actions', [])
    dones = trajectory.get('dones', [])
    
    # Ensure dones has the same length as actions
    while len(dones) < len(actions):
        dones.append(False)
    while len(dones) > len(actions):
        dones.pop()
    
    # Find indices of 'wait' actions to remove
    indices_to_remove = []
    for i, action in enumerate(actions):
        if action.get('type', '') == 'wait':
            indices_to_remove.append(i)
    
    # Remove wait actions and corresponding dones (in reverse order to maintain indices)
    # Note: observations[i+1] is the state after action[i]
    # When removing action[i], we should remove observation[i+1] (the state after wait)
    for idx in reversed(indices_to_remove):
        actions.pop(idx)
        if idx < len(dones):
            dones.pop(idx)
        # Remove the observation after the wait action (since wait doesn't change state meaningfully)
        # observation[0] is initial state, observation[i+1] is state after action[i]
        if idx + 1 < len(observations):
            observations.pop(idx + 1)
    
    # Ensure observations length matches (should be len(actions) + 1)
    # If we removed observations, we might need to adjust
    expected_obs_len = len(actions) + 1
    if len(observations) > expected_obs_len:
        # Remove extra observations from the end (keep initial state)
        observations = observations[:expected_obs_len]
    elif len(observations) < expected_obs_len:
        # This shouldn't happen, but if it does, pad with last observation
        while len(observations) < expected_obs_len:
            observations.append(observations[-1] if observations else {})
    
    # Recompute rewards
    task_config = trajectory.get('task', {})
    rewards = recompute_rewards(observations, actions, dones, task_config)
    
    # Ensure rewards length matches actions length
    while len(rewards) < len(actions):
        rewards.append(-0.1)
    while len(rewards) > len(actions):
        rewards.pop()
    
    # Update trajectory
    trajectory['observations'] = observations
    trajectory['actions'] = actions
    trajectory['rewards'] = rewards
    trajectory['dones'] = dones
    
    return trajectory


def main():
    """Update all trajectory files in data/trajectories/."""
    trajectories_dir = Path(__file__).parent.parent / 'data' / 'trajectories'
    
    if not trajectories_dir.exists():
        print(f"Error: Trajectories directory not found: {trajectories_dir}")
        return
    
    trajectory_files = sorted(trajectories_dir.glob('trajectory_*.json'))
    
    if not trajectory_files:
        print(f"No trajectory files found in {trajectories_dir}")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files to update")
    
    updated_count = 0
    error_count = 0
    
    for traj_file in trajectory_files:
        try:
            print(f"Processing {traj_file.name}...")
            
            # Load trajectory
            with open(traj_file, 'r') as f:
                trajectory = json.load(f)
            
            # Update trajectory
            updated_trajectory = update_trajectory(trajectory)
            
            # Save updated trajectory
            with open(traj_file, 'w') as f:
                json.dump(updated_trajectory, f, indent=2)
            
            updated_count += 1
            print(f"  ✓ Updated {traj_file.name}")
            
        except Exception as e:
            error_count += 1
            print(f"  ✗ Error processing {traj_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nUpdate complete!")
    print(f"  Updated: {updated_count} files")
    print(f"  Errors: {error_count} files")


if __name__ == '__main__':
    main()

