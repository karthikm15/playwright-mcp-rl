"""Test browser environment with MCP server."""

import asyncio
import json
from utils.mcp_client import MCPClient
from env.browser_env import BrowserEnv


async def main():
    """Test browser environment interaction."""
    # Load task config
    with open('data/tasks/example_single_field.json', 'r') as f:
        task_config = json.load(f)
    
    # Create Playwright browser client
    print("Initializing Playwright browser...")
    mcp_client = await MCPClient.create()
    
    # Create browser environment
    env = BrowserEnv(task_config, mcp_client)
    
    try:
        # Reset environment
        print("\n=== RESET ===")
        state = await env.reset()
        print(f"Initial state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
        print(f"State preview: {str(state)[:200]}...")
        
        # Get snapshot to see available elements
        snapshot = await env.render()
        print(f"\nSnapshot type: {type(snapshot)}")
        print(f"Snapshot preview: {str(snapshot)}")
        
        # Find input field (textbox), radio buttons, and submit button
        input_ref = None
        radio_ref = None
        submit_ref = None
        
        assert isinstance(snapshot, dict), f"Snapshot is not a dict: {type(snapshot)}"
        # Handle both dict 
        def find_elements_in_tree(node, input_refs, radio_refs, submit_refs):
            if not isinstance(node, dict):
                return
            role = node.get('type', node.get('role', '')).lower()
            name = node.get('name', '').lower()
            ref = node.get('ref', '')
            
            if 'textbox' in role or ('input' in role and 'name' in str(node).lower()):
                if ref:
                    input_refs.append((ref, node))
            if role == 'radio':
                if ref:
                    radio_refs.append((ref, node))
            if 'button' in role and 'submit' in name:
                if ref:
                    submit_refs.append((ref, node))
            
            for value in node.values():
                if isinstance(value, (dict, list)):
                    if isinstance(value, list):
                        for item in value:
                            find_elements_in_tree(item, input_refs, radio_refs, submit_refs)
                    else:
                        find_elements_in_tree(value, input_refs, radio_refs, submit_refs)
        
        input_candidates = []
        radio_candidates = []
        submit_candidates = []
        find_elements_in_tree(snapshot, input_candidates, radio_candidates, submit_candidates)
        
        if input_candidates:
            input_ref = input_candidates[0][0]
            print(f"Found input field: {input_ref}")
        if radio_candidates:
            # Select the first radio button (e.g., "Freshman")
            radio_ref = radio_candidates[0][0]
            radio_name = radio_candidates[0][1].get('name', '')
            print(f"Found radio button: {radio_ref} ({radio_name})")
        if submit_candidates:
            submit_ref = submit_candidates[0][0]
            print(f"Found submit button: {submit_ref}")
        # else:
        #     # String-based snapshot parsing (fallback)
        #     breakpoint()
        #     snapshot_str = str(snapshot)
        #     element_refs = re.findall(r'\[ref=([^\]]+)\]', snapshot_str)
        #     element_refs = list(set(element_refs))
        #     print(f"\nFound element refs: {element_refs}")
            
        #     lines = snapshot_str.split('\n')
        #     for i, line in enumerate(lines):
        #         line_lower = line.lower()
        #         # Look for textbox/input field
        #         if ('textbox' in line_lower or ('input' in line_lower and 'name' in line_lower)) and '[ref=' in line:
        #             ref_match = re.search(r'\[ref=([^\]]+)\]', line)
        #             if ref_match:
        #                 input_ref = ref_match.group(1)
        #                 print(f"Found input field: {input_ref} in line: {line.strip()}")
                
        #         # Look for submit button
        #         if re.match(r'\s*-\s*button\s+"submit"\s+\[ref=[^\]]+\]', line, re.IGNORECASE):
        #             ref_match = re.search(r'\[ref=([^\]]+)\]', line)
        #             if ref_match:
        #                 submit_ref = ref_match.group(1)
        #                 print(f"Found submit button: {submit_ref} in line: {line.strip()}")
            
        #     # If not found by keywords, try to find by context
        #     if not input_ref:
        #         for line in lines:
        #             if 'name=' in line.lower() and '[ref=' in line:
        #                 ref_match = re.search(r'\[ref=([^\]]+)\]', line)
        #                 if ref_match:
        #                     input_ref = ref_match.group(1)
        #                     break
            
        #     submit_ref = submit_ref or "e89"  # Fallback for testing

        print(f"\nSelected refs - Input: {input_ref}, Radio: {radio_ref}, Submit: {submit_ref}")
        # Execute actions to complete form
        if input_ref:
            # Step 1: Click input field
            print(f"\n=== STEP 1: Click input field ({input_ref}) ===")
            action = {
                'type': 'click',
                'element_ref': input_ref,
                'description': 'input field'
            }
            state, reward, done, info = await env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            
            # Step 2: Type text into input
            print("\n=== STEP 2: Type text into input ===")
            action = {
                'type': 'type',
                'element_ref': input_ref,
                'description': 'input field',
                'text': 'John Doe'
            }
            state, reward, done, info = await env.step(action)
            print(f"Reward: {reward}, Done: {done}")
        else:
            print("No input field found in snapshot")
        
        # Step 3: Check radio button for year
        if radio_ref:
            print(f"\n=== STEP 3: Check radio button ({radio_ref}) ===")
            action = {
                'type': 'check',
                'element_ref': radio_ref,
                'description': 'year radio button'
            }
            state, reward, done, info = await env.step(action)
            print(f"Reward: {reward}, Done: {done}")
        else:
            print("No radio button found in snapshot")
            
        # Step 4: Submit form
        if submit_ref:
            print(f"\n=== STEP 4: Submit form ({submit_ref}) ===")
            action = {
                'type': 'submit',
                'element_ref': submit_ref,
                'description': 'submit button'
            }
            state, reward, done, info = await env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            print(f"Info: {info}")
            print(f"Success: {info.get('success', False)}")
            
            # Check final state
            final_snapshot = await env.render()
            print(f"\nFinal snapshot preview: {str(final_snapshot)[:500]}")
        else:
            print("No submit button found")
    finally:
        await env.close()
        print("\n=== Test complete ===")


if __name__ == '__main__':
    asyncio.run(main())

