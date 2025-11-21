import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from env.browser_env import BrowserEnv
from playwright.async_api import Page


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


def find_element_ref_by_selector(snapshot: Dict[str, Any], selector: str, role: str) -> Optional[str]:
    """Find element ref by matching selector/role in snapshot."""
    def traverse(node):
        if not isinstance(node, dict):
            return None
        
        node_role = node.get('type', node.get('role', '')).lower()
        node_ref = node.get('ref', '')
        
        # Try to match by role
        if node_ref and node_role == role.lower():
            return node_ref
        
        # Traverse children
        for value in node.values():
            if isinstance(value, (dict, list)):
                if isinstance(value, list):
                    for item in value:
                        result = traverse(item)
                        if result:
                            return result
                else:
                    result = traverse(value)
                    if result:
                        return result
        return None
    
    return traverse(snapshot)


async def setup_interaction_monitoring(page: Page, env: BrowserEnv, actions: List, observations: List, 
                                      rewards: List, dones: List, task_config: Dict[str, Any]):
    """Set up JavaScript monitoring to capture user interactions."""
    
    # Queue to store captured actions from JavaScript
    action_queue = asyncio.Queue()
    
    # Expose function for JavaScript to call when user interacts
    async def record_action(action_data: Dict[str, Any]):
        """Called from JavaScript when user interacts with page."""
        await action_queue.put(action_data)
    
    await page.expose_function("recordAction", record_action)
    
    # Inject monitoring script - must be added after page loads
    async def inject_monitoring():
        await page.evaluate("""
        (function() {
            if (window.__trajectoryMonitoring) {
                return; // Already injected
            }
            window.__trajectoryMonitoring = true;
            
            let lastInputValue = {};
            let typingTimeout = null;
            let lastClickTime = 0;
            
            // Helper to find the actual interactive element (not just a child span/div)
            function findInteractiveElement(element) {
                let current = element;
                // Walk up the DOM tree to find the actual interactive element
                while (current && current !== document.body) {
                    const tag = current.tagName;
                    const role = current.getAttribute('role');
                    const type = current.type;
                    
                    // Check if this is an interactive element
                    if (role === 'button' || role === 'radio' || role === 'checkbox' || role === 'textbox' || role === 'link') {
                        return current;
                    }
                    if (tag === 'BUTTON' || tag === 'A' || tag === 'INPUT' || tag === 'TEXTAREA') {
                        return current;
                    }
                    if (type === 'button' || type === 'submit' || type === 'radio' || type === 'checkbox') {
                        return current;
                    }
                    
                    current = current.parentElement;
                }
                return element; // Fallback to original
            }
            
            // Monitor clicks - use capture phase to catch all clicks
            document.addEventListener('click', async function(e) {
                const now = Date.now();
                // Debounce rapid clicks
                if (now - lastClickTime < 100) {
                    return;
                }
                lastClickTime = now;
                
                // Find the actual interactive element (not just a child)
                const target = findInteractiveElement(e.target);
                
                let role = target.getAttribute('role') || '';
                if (!role) {
                    // Try to infer role from tag/type
                    if (target.tagName === 'BUTTON' || target.type === 'button' || target.type === 'submit') {
                        role = 'button';
                    } else if (target.tagName === 'INPUT' && target.type === 'radio') {
                        role = 'radio';
                    } else if (target.tagName === 'INPUT' && target.type === 'checkbox') {
                        role = 'checkbox';
                    } else if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
                        role = 'textbox';
                    } else if (target.tagName === 'A') {
                        role = 'link';
                    } else {
                        // Skip non-interactive elements
                        return;
                    }
                }
                
                // Get name - try multiple sources, including walking up for text content
                let name = target.getAttribute('aria-label') || 
                          target.getAttribute('aria-labelledby') ||
                          target.getAttribute('placeholder') ||
                          target.getAttribute('name') || 
                          target.getAttribute('value') ||
                          '';
                
                // For buttons/links, get text content (including from children)
                if (!name && (role === 'button' || role === 'link' || target.tagName === 'BUTTON' || target.tagName === 'A')) {
                    name = target.textContent?.trim() || target.innerText?.trim() || '';
                }
                
                // For inputs, try to get label
                if (!name && target.id) {
                    const label = document.querySelector('label[for="' + target.id + '"]');
                    if (label) {
                        name = label.textContent?.trim() || '';
                    }
                }
                
                // Try aria-labelledby
                if (!name && target.getAttribute('aria-labelledby')) {
                    const labelId = target.getAttribute('aria-labelledby');
                    const labelEl = document.getElementById(labelId);
                    if (labelEl) {
                        name = labelEl.textContent?.trim() || '';
                    }
                }
                
                // Skip if we still don't have a name for interactive elements
                if (!name && role !== 'textbox') {
                    console.log('Skipping element without name:', role, target);
                    return;
                }
                
                // Determine action type
                let actionType = 'click';
                if (role === 'button' && (name.toLowerCase().includes('submit') || target.type === 'submit')) {
                    actionType = 'submit';
                } else if (role === 'radio' || (target.type === 'radio')) {
                    actionType = 'check';
                } else if (role === 'checkbox' || (target.type === 'checkbox')) {
                    actionType = 'check';
                }
                
                console.log('Recording action:', actionType, name, role, target.tagName);
                
                try {
                    await window.recordAction({
                        type: actionType,
                        element_selector: target.id || target.name || target.className || '',
                        element_name: name.substring(0, 200),
                        element_role: role,
                        timestamp: Date.now()
                    });
                } catch (err) {
                    console.error('Error recording action:', err);
                }
            }, true);
            
            // Monitor input/typing
            document.addEventListener('input', async function(e) {
                const target = e.target;
                if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || 
                    target.getAttribute('role') === 'textbox' || 
                    target.contentEditable === 'true') {
                    
                    const currentValue = target.value || target.textContent || '';
                    const previousValue = lastInputValue[target.id || target.name || target.className] || '';
                    
                    // Only record if value actually changed
                    if (currentValue !== previousValue) {
                        lastInputValue[target.id || target.name || target.className] = currentValue;
                        
                        // Clear previous timeout
                        if (typingTimeout) {
                            clearTimeout(typingTimeout);
                        }
                        
                        // Wait a bit to see if user is still typing
                        typingTimeout = setTimeout(async () => {
                            let role = target.getAttribute('role') || 'textbox';
                            let name = target.getAttribute('aria-label') || 
                                      target.getAttribute('placeholder') ||
                                      target.getAttribute('name') || 
                                      target.id || '';
                            
                            // Try to get label
                            if (!name && target.id) {
                                const label = document.querySelector('label[for="' + target.id + '"]');
                                if (label) {
                                    name = label.textContent?.trim() || '';
                                }
                            }
                            
                            console.log('Recording type action:', name, currentValue);
                            
                            try {
                                await window.recordAction({
                                    type: 'type',
                                    element_selector: target.id || target.name || '',
                                    element_name: name.substring(0, 200),
                                    element_role: role,
                                    text: currentValue,
                                    timestamp: Date.now()
                                });
                            } catch (err) {
                                console.error('Error recording type action:', err);
                            }
                        }, 800); // Wait 800ms after last keystroke
                    }
                }
            }, true);
            
            // Monitor form submissions
            document.addEventListener('submit', async function(e) {
                e.preventDefault(); // Don't actually submit, we'll handle it
                const form = e.target;
                console.log('Form submit detected');
                try {
                    await window.recordAction({
                        type: 'submit',
                        element_selector: form.id || 'form',
                        element_name: 'form submit',
                        element_role: 'form',
                        timestamp: Date.now()
                    });
                } catch (err) {
                    console.error('Error recording submit:', err);
                }
            }, true);
        })();
        """)
    
    # Inject after page loads
    await page.wait_for_load_state('networkidle')
    await inject_monitoring()
    
    return action_queue


async def collect_trajectory(task_config_path: str, output_path: str = None, headless: bool = False):
    """
    Collect a trajectory by monitoring direct browser interactions.
    
    Args:
        task_config_path: Path to task configuration JSON file
        output_path: Path to save trajectory (default: data/demos/trajectory_XXX.json)
        headless: Whether to run browser in headless mode (must be False for interaction)
    """
    if headless:
        print("Warning: headless mode disabled - browser must be visible for interaction")
        headless = False
    
    # Load task config
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Create environment
    env = BrowserEnv(task_config, headless=headless)
    
    # Initialize trajectory storage
    observations = []
    actions = []
    rewards = []
    dones = []
    
    try:
        # Reset environment
        print("Resetting environment...")
        initial_state = await env.reset()
        observations.append({
            'type': 'snapshot',
            'url': env.current_url or task_config['url'],
            'elements': extract_elements_from_snapshot(initial_state)
        })
        
        print(f"\nTask: {task_config.get('url', 'N/A')}")
        print(f"Max steps: {task_config.get('max_steps', 50)}")
        print(f"Success condition: {task_config.get('success_condition', 'N/A')}")
        print("\n" + "="*60)
        print("INTERACT WITH THE BROWSER DIRECTLY")
        print("All clicks, typing, and interactions will be recorded automatically")
        print("Press Ctrl+C or close the browser to finish and save trajectory")
        print("="*60 + "\n")
        
        # Set up interaction monitoring
        page = env.page
        action_queue = await setup_interaction_monitoring(page, env, actions, observations, rewards, dones, task_config)
        
        # Wait a moment for monitoring to be ready
        await asyncio.sleep(0.5)
        
        done = False
        step_count = 0
        last_action_time = 0
        
        # Monitor for actions and check for completion
        print("Waiting for interactions... (interact with the browser now)")
        
        while not done and step_count < task_config.get('max_steps', 50):
            try:
                # Get current snapshot FIRST (before action) to have correct elements for matching
                snapshot_before = await env.render()
                elements = extract_elements_from_snapshot(snapshot_before)
                
                # Wait for action with timeout to periodically check for success
                try:
                    js_action = await asyncio.wait_for(action_queue.get(), timeout=2.0)
                    print(f"Received action from browser: {js_action.get('type')} on '{js_action.get('element_name')}'")
                except asyncio.TimeoutError:
                    # Check for success condition periodically
                    success = await env._check_success()
                    if success:
                        print("\n✓ Success condition detected!")
                        done = True
                        break
                    # Refresh snapshot for next iteration
                    snapshot_before = await env.render()
                    elements = extract_elements_from_snapshot(snapshot_before)
                    continue
                
                # Try to find element ref by matching name/role
                element_ref = None
                element_name = js_action.get('element_name', '').strip()
                element_role = js_action.get('element_role', '').strip()
                
                print(f"Looking for element: name='{element_name}', role='{element_role}'")
                print(f"Available elements: {[(e['ref'], e['type'], e['name'][:30]) for e in elements[:5]]}")
                
                # Match element by name and role (exact match first)
                for elem in elements:
                    elem_name = elem['name'].strip()
                    elem_type = elem['type'].lower()
                    
                    # Try exact name match
                    if elem_name.lower() == element_name.lower():
                        if elem_type == element_role.lower() or not element_role:
                            element_ref = elem['ref']
                            element_name = elem['name']  # Use canonical name
                            print(f"Found exact match: {element_ref}")
                            break
                
                # If not found, try partial name match
                if not element_ref:
                    for elem in elements:
                        elem_name = elem['name'].strip()
                        elem_type = elem['type'].lower()
                        
                        if element_name and (element_name.lower() in elem_name.lower() or elem_name.lower() in element_name.lower()):
                            if elem_type == element_role.lower() or not element_role:
                                element_ref = elem['ref']
                                element_name = elem['name']
                                print(f"Found partial match: {element_ref}")
                                break
                
                # If still not found, try just by role (for inputs, try to match by position)
                if not element_ref and element_role:
                    candidates = [e for e in elements if e['type'].lower() == element_role.lower()]
                    if candidates:
                        # For textbox, prefer ones that match name partially
                        if element_role == 'textbox' and element_name:
                            for cand in candidates:
                                if element_name.lower() in cand['name'].lower():
                                    element_ref = cand['ref']
                                    element_name = cand['name']
                                    print(f"Found by role+name: {element_ref}")
                                    break
                        if not element_ref and candidates:
                            element_ref = candidates[0]['ref']
                            element_name = candidates[0]['name']
                            print(f"Found by role only: {element_ref}")
                
                # If still not found, skip this action but log it
                if not element_ref:
                    print(f"Warning: Could not map element '{element_name}' ({element_role}) to ref")
                    print(f"  Available elements: {[(e['ref'], e['type'], e['name'][:50]) for e in elements]}")
                    continue
                
                # Build action dict
                action_type = js_action.get('type', 'click')
                action_dict = {
                    'type': action_type,
                    'element_ref': element_ref,
                    'name': element_name,
                    'description': element_name
                }
                
                if action_type == 'type':
                    typed_text = js_action.get('text', '')
                    action_dict['text'] = typed_text
                
                # Check for duplicate actions (same action within 200ms)
                current_time = js_action.get('timestamp', 0)
                if current_time - last_action_time < 200:
                    print("Skipping duplicate action (within 200ms)")
                    continue
                last_action_time = current_time
                
                # Record the action (user already performed it in browser)
                print(f"Recording: {action_type} on {element_name} (ref: {element_ref})")
                
                # Wait a bit for page to settle after the action
                await asyncio.sleep(0.5)
                
                # Get current state AFTER the action (user already performed it)
                try:
                    state_after = await env.render()
                except Exception as e:
                    print(f"Warning: Could not get state after action: {e}")
                    # Page might have navigated, try to get new page state
                    await asyncio.sleep(1.0)
                    try:
                        state_after = await env.render()
                    except Exception:
                        # If still failing, use last known state
                        state_after = observations[-1] if observations else snapshot_before
                
                # Check for success condition
                success = await env._check_success()
                
                # Compute reward and done
                reward = -0.01  # Default step penalty
                done = False
                
                if success:
                    reward = 1.0
                    done = True
                elif step_count + 1 >= task_config.get('max_steps', 50):
                    reward = -1.0
                    done = True
                
                # Store trajectory data IN ORDER
                actions.append(action_dict)
                rewards.append(reward)
                dones.append(done)
                
                # Store observation after action
                observations.append({
                    'type': 'snapshot',
                    'url': env.current_url or task_config.get('url', ''),
                    'elements': extract_elements_from_snapshot(state_after)
                })
                
                step_count += 1
                print(f"✓ Step {step_count}: {action_type} on {element_name} - Reward: {reward}, Done: {done}")
                
                if success:
                    print("✓ Success condition met!")
                    done = True
                    break
                
                print()  # Blank line for readability
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving trajectory...")
                break
            except Exception as e:
                print(f"Error processing action: {e}")
                import traceback
                traceback.print_exc()
                # Don't continue - we want to see what went wrong
                await asyncio.sleep(0.5)
                continue
        
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
            demo_dir = Path('data/demos')
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
        
        print(f"\n✓ Trajectory saved to: {output_path}")
        print(f"  Steps: {step_count}")
        print(f"  Actions recorded: {len(actions)}")
        print(f"  Success: {dones[-1] if dones else False}")
        print(f"  Total reward: {sum(rewards)}")
        
    finally:
        await env.close()


async def collect_trajectory_programmatic(task_config_path: str, action_sequence: List[Dict[str, Any]], 
                                         output_path: str = None, headless: bool = True):
    """
    Collect a trajectory programmatically with a predefined action sequence.
    
    Args:
        task_config_path: Path to task configuration JSON file
        action_sequence: List of action dictionaries to execute
        output_path: Path to save trajectory
        headless: Whether to run browser in headless mode
    """
    # Load task config
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # Create environment
    env = BrowserEnv(task_config, headless=headless)
    
    # Initialize trajectory storage
    observations = []
    actions = []
    rewards = []
    dones = []
    
    try:
        # Reset environment
        initial_state = await env.reset()
        snapshot_elements = extract_elements_from_snapshot(initial_state)
        observations.append({
            'type': 'snapshot',
            'url': env.current_url or task_config['url'],
            'elements': snapshot_elements
        })
        
        done = False
        step_count = 0
        
        for action_dict in action_sequence:
            if done or step_count >= task_config.get('max_steps', 50):
                break
            
            # Get element name from current snapshot if not provided
            if 'element_ref' in action_dict and 'name' not in action_dict:
                snapshot = await env.render()
                elements = extract_elements_from_snapshot(snapshot)
                element_name = next((e['name'] for e in elements if e['ref'] == action_dict['element_ref']), '')
                action_dict['name'] = element_name
                if 'description' not in action_dict:
                    action_dict['description'] = element_name
            
            # Execute action
            state, reward, done, info = await env.step(action_dict)
            
            # Store trajectory data
            actions.append(action_dict)
            rewards.append(reward)
            dones.append(done)
            
            # Store observation after action
            observations.append({
                'type': 'snapshot',
                'url': env.current_url or task_config['url'],
                'elements': extract_elements_from_snapshot(state)
            })
            
            step_count += 1
            
            if done:
                break
        
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
            demo_dir = Path('data/demos')
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
        
        print(f"Trajectory saved to: {output_path}")
        return trajectory
        
    finally:
        await env.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/collect_trajectory.py <task_config_path> [output_path] [--headless]")
        sys.exit(1)
    
    task_config_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    headless = '--headless' in sys.argv
    
    asyncio.run(collect_trajectory(task_config_path, output_path, headless))