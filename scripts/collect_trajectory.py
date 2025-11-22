import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from env.browser_env import BrowserEnv


def extract_elements_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all interactive elements from snapshot tree."""
    elements = []
    
    def traverse(node):
        if not isinstance(node, dict):
            return
        
        # Check if this is an interactive element
        role = node.get('type', node.get('role', '')).lower()
        ref = node.get('ref', '')
        name = node.get('name', '')
        
        if ref and role in ['textbox', 'button', 'radio', 'checkbox', 'link']:
            elements.append({
                'ref': ref,
                'type': role,
                'name': name,
                'value': node.get('value', '')
            })
        
        # Traverse children
        for value in node.values():
            if isinstance(value, (dict, list)):
                if isinstance(value, list):
                    for item in value:
                        traverse(item)
                else:
                    traverse(value)
    
    traverse(snapshot)
    return elements


def display_state(snapshot: Dict[str, Any], step: int, done: bool):
    """Display current page state and available actions."""
    elements = extract_elements_from_snapshot(snapshot)
    
    print("\n" + "="*80)
    print(f"STEP {step} - Current Page State")
    print("="*80)
    
    if done:
        print("✓ Task completed!")
        return
    
    if not elements:
        print("No interactive elements found on page.")
        return
    
    # Group elements by type
    textboxes = [e for e in elements if e['type'] == 'textbox']
    buttons = [e for e in elements if e['type'] == 'button']
    radios = [e for e in elements if e['type'] == 'radio']
    checkboxes = [e for e in elements if e['type'] == 'checkbox']
    links = [e for e in elements if e['type'] == 'link']
    
    print("\n📝 AVAILABLE ACTIONS:\n")
    
    action_num = 1
    action_map = {}
    
    # Textboxes - offer both click and type options
    if textboxes:
        print("TEXT INPUTS:")
        for elem in textboxes:
            name = elem['name'][:60] if elem['name'] else f"Textbox {elem['ref']}"
            value = elem.get('value', '')
            value_display = f" (current: '{value}')" if value else ""
            # Option to click (focus) the input
            print(f"  [{action_num}] Click: {name}{value_display}")
            action_map[action_num] = {
                'type': 'click',
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'textbox'
            }
            action_num += 1
            # Option to type in the input
            print(f"  [{action_num}] Type in: {name}{value_display}")
            action_map[action_num] = {
                'type': 'type',
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'textbox'
            }
            action_num += 1
        print()
    
    # Buttons
    if buttons:
        print("BUTTONS:")
        for elem in buttons:
            name = elem['name'][:60] if elem['name'] else f"Button {elem['ref']}"
            action_type = 'submit' if 'submit' in name.lower() else 'click'
            print(f"  [{action_num}] {action_type.capitalize()}: {name}")
            action_map[action_num] = {
                'type': action_type,
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'button'
            }
            action_num += 1
        print()
    
    # Radio buttons
    if radios:
        print("RADIO BUTTONS:")
        for elem in radios:
            name = elem['name'][:60] if elem['name'] else f"Radio {elem['ref']}"
            print(f"  [{action_num}] Select: {name}")
            action_map[action_num] = {
                'type': 'check',
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'radio'
            }
            action_num += 1
        print()
    
    # Checkboxes
    if checkboxes:
        print("CHECKBOXES:")
        for elem in checkboxes:
            name = elem['name'][:60] if elem['name'] else f"Checkbox {elem['ref']}"
            print(f"  [{action_num}] Toggle: {name}")
            action_map[action_num] = {
                'type': 'check',
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'checkbox'
            }
            action_num += 1
        print()
    
    # Links
    if links:
        print("LINKS:")
        for elem in links:
            name = elem['name'][:60] if elem['name'] else f"Link {elem['ref']}"
            print(f"  [{action_num}] Click: {name}")
            action_map[action_num] = {
                'type': 'click',
                'element_ref': elem['ref'],
                'element_name': elem['name'],
                'element_type': 'link'
            }
            action_num += 1
        print()
    
    # Special actions
    print("OTHER ACTIONS:")
    print(f"  [{action_num}] Wait (do nothing)")
    action_map[action_num] = {'type': 'wait'}
    action_num += 1
    
    print(f"  [{action_num}] Finish and save trajectory")
    action_map[action_num] = {'type': 'finish'}
    action_num += 1
    
    print()
    print("="*80)
    
    return action_map


async def get_user_action(action_map: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get action from user input."""
    while True:
        try:
            choice = input("\nEnter action number: ").strip()
            if not choice:
                continue
            
            action_num = int(choice)
            if action_num in action_map:
                action = action_map[action_num].copy()
                
                # If it's a type action, get the text to type
                if action['type'] == 'type':
                    text = input(f"Enter text to type in '{action['element_name']}': ").strip()
                    if not text:
                        print("No text entered, skipping...")
                        continue
                    action['text'] = text
                
                return action
            else:
                print(f"Invalid choice. Please enter a number between 1 and {max(action_map.keys())}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            return {'type': 'finish'}


async def collect_trajectory(task_config_path: str, output_path: str = None, headless: bool = True):
    """
    Collect a trajectory using command line interface.
    
    Args:
        task_config_path: Path to task configuration JSON file
        output_path: Path to save trajectory (default: data/demos/trajectory_XXX.json)
        headless: Whether to run browser in headless mode (default: True)
    """
    # Load task config
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Create environment (headless)
    env = BrowserEnv(task_config, headless=headless)
    
    # Initialize trajectory storage
    observations = []
    actions = []
    rewards = []
    dones = []
    
    try:
        # Reset environment
        print("Initializing browser environment...")
        initial_state = await env.reset()
        observations.append({
            'type': 'snapshot',
            'url': env.current_url or task_config['url'],
            'elements': extract_elements_from_snapshot(initial_state)
        })
        
        print(f"\n{'='*80}")
        print("TRAJECTORY COLLECTION")
        print(f"{'='*80}")
        print(f"Task URL: {task_config.get('url', 'N/A')}")
        print(f"Max steps: {task_config.get('max_steps', 50)}")
        print(f"Success condition: {task_config.get('success_condition', 'N/A')}")
        print(f"{'='*80}\n")
        
        step_count = 0
        done = False
        
        while step_count < task_config.get('max_steps', 50) and not done:
            # Get current snapshot
            snapshot = await env.render()
            
            # Display state and get available actions
            action_map = display_state(snapshot, step_count + 1, done)
            
            if not action_map:
                print("No actions available. Finishing...")
                break
            
            # Get user action
            user_action = await get_user_action(action_map)
            
            if not user_action:
                continue
            
            if user_action['type'] == 'finish':
                print("\nFinishing trajectory collection...")
                break
            
            if user_action['type'] == 'wait':
                print("Waiting...")
                await asyncio.sleep(1.0)
                # Still record this as an action with minimal reward
                actions.append({
                    'type': 'wait',
                    'element_ref': '',
                    'name': 'wait',
                    'description': 'wait'
                })
                rewards.append(-0.1)
                dones.append(False)
                observations.append({
                    'type': 'snapshot',
                    'url': env.current_url or task_config.get('url', ''),
                    'elements': extract_elements_from_snapshot(snapshot)
                })
                step_count += 1
                continue
            
            # Build action dict for environment
            action_dict = {
                'type': user_action['type'],
                'element_ref': user_action.get('element_ref', ''),
                'description': user_action.get('element_name', ''),
            }
            
            if 'text' in user_action:
                action_dict['text'] = user_action['text']
            
            # Execute action
            print(f"\nExecuting: {user_action['type']} on {user_action.get('element_name', 'element')}...")
            
            try:
                state_after, reward, done, info = await env.step(action_dict)
                
                # Override reward for failed submit actions
                # If user clicked submit but success condition is not met, give -1.0 reward
                # (This matches the pattern in trajectory_002.json where failed submits get -1.0)
                if (user_action['type'] == 'submit' or 
                    (user_action['type'] == 'click' and 'submit' in user_action.get('element_name', '').lower())):
                    success = info.get('success', False)
                    if not success:
                        # Submit failed - give -1.0 reward (unless it already succeeded)
                        if reward != 1.0:
                            reward = -1.0
                            print(f"  ⚠ Submit action failed - setting reward to -1.0")
                
            except Exception as e:
                print(f"Error executing action: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Store trajectory data
            actions.append({
                'type': user_action['type'],
                'element_ref': user_action.get('element_ref', ''),
                'name': user_action.get('element_name', ''),
                'description': user_action.get('element_name', ''),
                **({'text': user_action['text']} if 'text' in user_action else {})
            })
            rewards.append(reward)
            dones.append(done)
            
            # Store observation after action
            observations.append({
                'type': 'snapshot',
                'url': env.current_url or task_config.get('url', ''),
                'elements': extract_elements_from_snapshot(state_after)
            })
            
            step_count += 1
            
            # Display result
            print(f"\n✓ Action completed - Reward: {reward}, Done: {done}")
            if done:
                print("✓ Task completed successfully!")
                break
            
            # Small delay before next step
            await asyncio.sleep(0.3)
        
        # Save trajectory
        trajectory = {
            'task': task_config,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }
        
        # Determine output path
        if output_path is None:
            demo_dir = Path('data/trajectories')
            demo_dir.mkdir(parents=True, exist_ok=True)
            existing = list(demo_dir.glob('trajectory_*.json'))
            if existing:
                numbers = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1
            output_path = demo_dir / f'trajectory_{next_num:03d}.json'
        
        with open(output_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"\n{'='*80}")
        print("TRAJECTORY SAVED")
        print(f"{'='*80}")
        print(f"Path: {output_path}")
        print(f"Steps: {step_count}")
        print(f"Actions recorded: {len(actions)}")
        print(f"Success: {dones[-1] if dones else False}")
        print(f"Total reward: {sum(rewards):.2f}")
        print(f"{'='*80}\n")
        
    finally:
        await env.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/collect_trajectory.py <task_config_path> [output_path] [--no-headless]")
        sys.exit(1)
    
    task_config_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    headless = '--no-headless' not in sys.argv  # Default to headless
    
    asyncio.run(collect_trajectory(task_config_path, output_path, headless))
