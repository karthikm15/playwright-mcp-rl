import asyncio
import json
from env.browser_env import BrowserEnv


async def main():
    """Test browser environment interaction."""
    # Load task config
    with open('data/tasks/example_single_field.json', 'r') as f:
        task_config = json.load(f)
    
    env = BrowserEnv(task_config, headless=False)
    
    try:
        state = await env.reset()
        snapshot = await env.render()
        assert isinstance(snapshot, dict), f"Snapshot is not a dict: {type(snapshot)}"
        
        input_ref = None
        radio_ref = None
        submit_ref = None
        
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
            radio_ref = radio_candidates[0][0]
            radio_name = radio_candidates[0][1].get('name', '')
            print(f"Found radio button: {radio_ref} ({radio_name})")
        if submit_candidates:
            submit_ref = submit_candidates[0][0]
            print(f"Found submit button: {submit_ref}")

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

