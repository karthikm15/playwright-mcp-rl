"""
Script to test alternating between radio button selections on a Google Form.
Alternates between clicking "10" and "11" radio buttons 3 times each.
Does not type in textbox or submit the form.
"""

import asyncio
from env.browser_env import BrowserEnv


async def test_form_model_submission():
    """Test alternating radio button selections."""
    # Form URL
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeKu6H0jhrbhFpBePgaEmDooF9JwpAiEj_tkeWaQ0lX1GJLzQ/viewform"
    
    # Create task config
    task_config = {
        "url": form_url,
        "success_condition": "Your response has been recorded",
        "max_steps": 50
    }
    
    # Create environment (headless=False to see the browser)
    env = BrowserEnv(task_config, headless=False)
    
    try:
        print("=" * 80)
        print("TESTING FORM SUBMISSION")
        print("=" * 80)
        
        # Reset environment (navigates to form)
        print("\n[1] Navigating to form...")
        state = await env.reset()
        print(f"✓ Loaded form page")
        
        # Get initial snapshot
        snapshot = await env.render()
        
        # Helper function to find elements in snapshot
        def find_elements_in_tree(node, element_type=None, name_contains=None):
            """Find elements in the snapshot tree."""
            elements = []
            
            def traverse(n):
                if not isinstance(n, dict):
                    return
                
                elem_type = n.get('type', '').lower()
                elem_name = n.get('name', '').lower()
                elem_ref = n.get('ref', '')
                
                match = True
                if element_type and elem_type != element_type.lower():
                    match = False
                if name_contains and name_contains.lower() not in elem_name:
                    match = False
                
                if match and elem_ref:
                    elements.append({
                        'ref': elem_ref,
                        'type': n.get('type', ''),
                        'name': n.get('name', ''),
                        'value': n.get('value', ''),
                        'checked': n.get('checked', False)
                    })
                
                # Recursively check children
                for value in n.values():
                    if isinstance(value, dict):
                        traverse(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                traverse(item)
            
            traverse(node)
            return elements
        
        # Find age radio buttons
        print("\n[2] Finding age radio buttons...")
        radio_buttons = find_elements_in_tree(snapshot, element_type='radio')
        if not radio_buttons:
            print("✗ Could not find radio buttons")
            return
        
        # Filter for age-related radios (should have "10" or "11" in name)
        age_radios = [r for r in radio_buttons if '10' in r['name'] or '11' in r['name']]
        if not age_radios:
            print("✗ Could not find age radio buttons")
            print(f"  Available radios: {[r['name'] for r in radio_buttons]}")
            return
        
        # Separate "10" and "11" radio buttons
        radio_10 = None
        radio_11 = None
        
        for radio in age_radios:
            if '10' in radio['name'] and radio_10 is None:
                radio_10 = radio
            elif '11' in radio['name'] and radio_11 is None:
                radio_11 = radio
        
        if not radio_10 or not radio_11:
            print("✗ Could not find both '10' and '11' radio buttons")
            print(f"  Found radios: {[r['name'] for r in age_radios]}")
            return
        
        print(f"✓ Found radio '10': {radio_10['name']} (ref: {radio_10['ref']})")
        print(f"✓ Found radio '11': {radio_11['name']} (ref: {radio_11['ref']})")
        
        # Alternate between clicking "10" and "11" three times each (6 clicks total)
        print("\n[3] Alternating between radio buttons (3 times each)...")
        
        for i in range(3):
            # Click "10"
            print(f"\n  Click {i+1}: Selecting '10'...")
            action = {
                'type': 'check',
                'element_ref': radio_10['ref'],
                'description': radio_10['name']
            }
            state, reward, done, info = await env.step(action)
            print(f"    ✓ Selected '10' - Reward: {reward:.3f}, Done: {done}")
            
            # Wait a bit
            await asyncio.sleep(0.5)
            
            # Get updated snapshot
            snapshot = await env.render()
            
            # Re-find radios to get updated state
            radio_buttons = find_elements_in_tree(snapshot, element_type='radio')
            age_radios = [r for r in radio_buttons if '10' in r['name'] or '11' in r['name']]
            for radio in age_radios:
                if '10' in radio['name']:
                    radio_10 = radio
                elif '11' in radio['name']:
                    radio_11 = radio
            
            # Click "11"
            print(f"  Click {i+1}: Selecting '11'...")
            action = {
                'type': 'check',
                'element_ref': radio_11['ref'],
                'description': radio_11['name']
            }
            state, reward, done, info = await env.step(action)
            print(f"    ✓ Selected '11' - Reward: {reward:.3f}, Done: {done}")
            
            # Wait a bit
            await asyncio.sleep(0.5)
            
            # Get updated snapshot for next iteration
            snapshot = await env.render()
            
            # Re-find radios to get updated state
            radio_buttons = find_elements_in_tree(snapshot, element_type='radio')
            age_radios = [r for r in radio_buttons if '10' in r['name'] or '11' in r['name']]
            for radio in age_radios:
                if '10' in radio['name']:
                    radio_10 = radio
                elif '11' in radio['name']:
                    radio_11 = radio
        
        print("\n✓ Completed 6 clicks (3x '10', 3x '11')")
        print("  Note: Did not type in textbox or submit form")
        
        # Get final snapshot to show state
        print("\n[4] Final state...")
        final_snapshot = await env.render()
        
        # Check which radio is currently selected
        radio_buttons = find_elements_in_tree(final_snapshot, element_type='radio')
        age_radios = [r for r in radio_buttons if '10' in r['name'] or '11' in r['name']]
        
        for radio in age_radios:
            if radio.get('checked', False) or radio.get('value', '').lower() == 'true':
                print(f"  Currently selected: {radio['name']} (ref: {radio['ref']})")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        print("Summary:")
        print("  - Alternated between '10' and '11' radio buttons 3 times each")
        print("  - Total clicks: 6")
        print("  - Did not type in textbox")
        print("  - Did not submit form")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Keep browser open for a few seconds to see the result
        print("\nKeeping browser open for 5 seconds to view results...")
        await asyncio.sleep(5)
        await env.close()
        print("Browser closed.")


if __name__ == '__main__':
    asyncio.run(test_form_model_submission())

