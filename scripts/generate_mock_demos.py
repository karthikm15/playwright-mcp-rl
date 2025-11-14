"""Generate mock expert demonstrations."""

import json
from pathlib import Path


def create_mock_snapshot(url, elements):
    """Create mock accessibility snapshot."""
    return {
        "type": "snapshot",
        "url": url,
        "elements": elements
    }


def generate_mock_demo(task_config, actions_data):
    """
    Generate a mock demonstration.
    
    Args:
        task_config: Task configuration dict
        actions_data: List of (action_type, element_ref, description, text) tuples
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    
    url = task_config['url']
    
    # Initial state
    initial_elements = [
        {"ref": "e1", "type": "textbox", "name": "Input", "value": ""},
        {"ref": "e2", "type": "button", "name": "Submit"}
    ]
    observations.append(create_mock_snapshot(url, initial_elements))
    
    # Execute actions
    for i, (action_type, element_ref, description, text) in enumerate(actions_data):
        # Action
        actions.append({
            "type": action_type,
            "element_ref": element_ref,
            "description": description,
            "text": text
        })
        
        # Update state after action
        if action_type == "type":
            # Update input value
            elements = initial_elements.copy()
            elements[0]["value"] = text
            observations.append(create_mock_snapshot(url, elements))
        elif action_type == "submit":
            # Success state
            success_elements = [
                {"ref": "e3", "type": "heading", "name": "Thank you", "value": "Thank you for submitting!"}
            ]
            observations.append(create_mock_snapshot(url + "/success", success_elements))
        else:
            # Click - state unchanged
            observations.append(create_mock_snapshot(url, initial_elements))
        
        # Rewards and dones
        if action_type == "submit":
            rewards.append(1.0)
            dones.append(True)
        else:
            rewards.append(0.0)
            dones.append(False)
    
    return {
        "task": task_config,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones
    }


def main():
    """Generate multiple mock demonstrations."""
    demo_dir = Path("data/demos")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    demos = [
        {
            "task": {
                "url": "https://example.com/form",
                "field_selector": "#name-input",
                "submit_selector": "#submit-btn",
                "success_condition": "Thank you",
                "max_steps": 50
            },
            "actions": [
                ("click", "e1", "Name input field", ""),
                ("type", "e1", "Name input field", "John Doe"),
                ("submit", "e2", "Submit button", "")
            ]
        },
        {
            "task": {
                "url": "https://example.com/form",
                "field_selector": "#email-input",
                "submit_selector": "#submit-btn",
                "success_condition": "Thank you",
                "max_steps": 50
            },
            "actions": [
                ("click", "e1", "Email input field", ""),
                ("type", "e1", "Email input field", "user@example.com"),
                ("submit", "e2", "Submit button", "")
            ]
        },
        {
            "task": {
                "url": "https://example.com/form",
                "field_selector": "#message-input",
                "submit_selector": "#submit-btn",
                "success_condition": "Thank you",
                "max_steps": 50
            },
            "actions": [
                ("click", "e1", "Message input field", ""),
                ("type", "e1", "Message input field", "Hello world"),
                ("submit", "e2", "Submit button", "")
            ]
        }
    ]
    
    for i, demo_data in enumerate(demos, 1):
        demo = generate_mock_demo(demo_data["task"], demo_data["actions"])
        output_path = demo_dir / f"demo_{i:03d}.json"
        with open(output_path, 'w') as f:
            json.dump(demo, f, indent=2)
        print(f"Generated {output_path}")


if __name__ == '__main__':
    main()

