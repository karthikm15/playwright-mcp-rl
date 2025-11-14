"""Test browser environment with MCP server."""

import asyncio
import json
import re
from utils.mcp_client import MCPClient
from env.browser_env import BrowserEnv


async def main():
    """Test browser environment interaction."""
    # Load task config
    with open('data/tasks/example_single_field.json', 'r') as f:
        task_config = json.load(f)
    
    # Create MCP client
    print("Connecting to MCP server...")
    mcp_client = await MCPClient.create("http://localhost:8931/mcp")
    
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
        print(f"Snapshot preview: {str(snapshot)[:800]}")
        
        # Parse snapshot string to extract element refs
        def extract_refs(snapshot_str):
            """Extract element references from snapshot string."""
            refs = []
            matches = re.findall(r'\[ref=([^\]]+)\]', snapshot_str)
            return list(set(matches))
        
        snapshot_str = str(snapshot)
        element_refs = extract_refs(snapshot_str)
        print(f"\nFound element refs: {element_refs}")
        
        # Find input field (textbox) and submit button
        input_ref = None
        submit_ref = None
        
        lines = snapshot_str.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Look for textbox/input field
            if ('textbox' in line_lower or ('input' in line_lower and 'name' in line_lower)) and '[ref=' in line:
                ref_match = re.search(r'\[ref=([^\]]+)\]', line)
                if ref_match:
                    input_ref = ref_match.group(1)
                    print(f"Found input field: {input_ref} in line: {line.strip()}")
            
            # Look for submit button
            if ('submit' in line_lower or ('button' in line_lower and 'type' in line_lower)) and '[ref=' in line:
                ref_match = re.search(r'\[ref=([^\]]+)\]', line)
                if ref_match:
                    submit_ref = ref_match.group(1)
                    print(f"Found submit button: {submit_ref} in line: {line.strip()}")
        
        # If not found by keywords, try to find by context
        if not input_ref:
            # Look for lines with "name=" which often indicates form inputs
            for line in lines:
                if 'name=' in line.lower() and '[ref=' in line:
                    ref_match = re.search(r'\[ref=([^\]]+)\]', line)
                    if ref_match:
                        input_ref = ref_match.group(1)
                        break
        
        if not submit_ref:
            # Look for button or submit in any form
            for line in lines:
                if ('button' in line.lower() or 'submit' in line.lower()) and '[ref=' in line:
                    ref_match = re.search(r'\[ref=([^\]]+)\]', line)
                    if ref_match:
                        submit_ref = ref_match.group(1)
                        break
        
        print(f"\nSelected refs - Input: {input_ref}, Submit: {submit_ref}")
        
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
            print(f"\n=== STEP 2: Type text into input ===")
            action = {
                'type': 'type',
                'element_ref': input_ref,
                'description': 'input field',
                'text': 'John Doe'
            }
            state, reward, done, info = await env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            
            # Step 3: Submit form
            if submit_ref:
                print(f"\n=== STEP 3: Submit form ({submit_ref}) ===")
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
        else:
            print("No input field found in snapshot")
            print("Available refs:", element_refs)
            
    finally:
        await mcp_client.close()
        print("\n=== Test complete ===")


if __name__ == '__main__':
    asyncio.run(main())

