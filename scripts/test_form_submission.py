"""
Script to test form submission on a Google Form using Playwright.
Fills in name, selects age, and submits the form.
"""

import asyncio
import json
from pathlib import Path
from env.browser_env import BrowserEnv


async def test_form_submission():
    """Test submitting a Google Form."""
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
        
        # Find the name textbox
        print("\n[2] Finding name textbox...")
        name_textboxes = find_elements_in_tree(snapshot, element_type='textbox', name_contains='name')
        if not name_textboxes:
            print("✗ Could not find name textbox")
            return
        
        name_textbox = name_textboxes[0]
        print(f"✓ Found name textbox: {name_textbox['name']} (ref: {name_textbox['ref']})")
        
        # Type name into textbox
        print("\n[3] Typing name...")
        action = {
            'type': 'type',
            'element_ref': name_textbox['ref'],
            'text': 'Test User',
            'description': name_textbox['name']
        }
        state, reward, done, info = await env.step(action)
        print(f"✓ Typed 'Test User' into name field")
        print(f"  Reward: {reward}, Done: {done}")
        
        # Wait a bit for the form to update
        await asyncio.sleep(0.5)
        
        # Get updated snapshot
        snapshot = await env.render()
        
        # Find age radio buttons
        print("\n[4] Finding age radio buttons...")
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
        
        # Select the first age option (e.g., "10")
        age_radio = age_radios[0]
        print(f"✓ Found age radio: {age_radio['name']} (ref: {age_radio['ref']})")
        
        # Check the radio button
        print("\n[5] Selecting age...")
        action = {
            'type': 'check',
            'element_ref': age_radio['ref'],
            'description': age_radio['name']
        }
        state, reward, done, info = await env.step(action)
        print(f"✓ Selected age: {age_radio['name']}")
        print(f"  Reward: {reward}, Done: {done}")
        
        # Wait a bit
        await asyncio.sleep(0.5)
        
        # Get updated snapshot
        snapshot = await env.render()
        
        # Find submit button
        print("\n[6] Finding submit button...")
        buttons = find_elements_in_tree(snapshot, element_type='button')
        submit_buttons = [b for b in buttons if 'submit' in b['name'].lower()]
        
        if not submit_buttons:
            print("✗ Could not find submit button")
            print(f"  Available buttons: {[b['name'] for b in buttons]}")
            return
        
        submit_button = submit_buttons[0]
        print(f"✓ Found submit button: {submit_button['name']} (ref: {submit_button['ref']})")
        
        # Submit the form
        print("\n[7] Submitting form...")
        action = {
            'type': 'submit',
            'element_ref': submit_button['ref'],
            'description': submit_button['name']
        }
        state, reward, done, info = await env.step(action)
        print(f"✓ Submitted form")
        print(f"  Reward: {reward}, Done: {done}")
        print(f"  Success: {info.get('success', False)}")
        
        # Wait a bit to see the result
        await asyncio.sleep(2)
        
        # Check final state
        print("\n[8] Checking final state...")
        final_snapshot = await env.render()
        final_snapshot_str = str(final_snapshot).lower()
        
        if 'response has been recorded' in final_snapshot_str or 'your response' in final_snapshot_str:
            print("✓ SUCCESS: Form submitted successfully!")
            print("  The success message was found in the page.")
        else:
            print("? Could not confirm success message, but form may have been submitted.")
            print(f"  Snapshot preview: {str(final_snapshot)[:200]}...")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during form submission: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Keep browser open for a few seconds to see the result
        print("\nKeeping browser open for 5 seconds to view results...")
        await asyncio.sleep(5)
        await env.close()
        print("Browser closed.")


if __name__ == '__main__':
    asyncio.run(test_form_submission())

