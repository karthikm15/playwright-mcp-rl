"""Minimal browser environment wrapper for Playwright MCP."""

import json
from typing import Dict, Any, Tuple, Optional


class BrowserEnv:
    """Environment wrapper for browser form filling tasks using Playwright MCP."""
    
    def __init__(self, task_config: Dict[str, Any], mcp_client=None):
        """
        Initialize with task configuration.
        
        Args:
            task_config: dict with keys:
                - url: target URL
                - field_selector: CSS selector for input field (optional, for reference)
                - submit_selector: CSS selector for submit button (optional)
                - success_condition: text/selector to check for success
                - max_steps: maximum steps per episode
            mcp_client: MCP client instance with browser tools
        """
        self.task_config = task_config
        self.mcp_client = mcp_client
        self.current_step = 0
        self.max_steps = task_config.get('max_steps', 50)
        self.current_url = None
        self.last_snapshot = None
    
    async def _call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call MCP tool and return result."""
        if not self.mcp_client:
            return None
        try:
            print(f"Calling MCP tool {tool_name} with params: {params}")
            result = await self.mcp_client.call_tool(tool_name, params)
            print(f"MCP tool {tool_name} returned: {result}")
            return result
        except Exception as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _navigate(self, url: str):
        """Navigate to URL using MCP browser_navigate."""
        result = await self._call_mcp_tool('browser_navigate', {'url': url})
        self.current_url = url
        return result
    
    async def _get_snapshot(self) -> Dict[str, Any]:
        """Get accessibility snapshot of current page."""
        result = await self._call_mcp_tool('browser_snapshot', {})
        if result:
            self.last_snapshot = result
            return result
        return self.last_snapshot or {}
    
    async def _click(self, element_ref: str, description: str = ""):
        """Click element using MCP browser_click."""
        # Playwright MCP browser_click typically only needs the ref
        return await self._call_mcp_tool('browser_click', {
            'ref': element_ref
        })
    
    async def _type(self, element_ref: str, text: str, description: str = ""):
        """Type text into element using MCP browser_type."""
        return await self._call_mcp_tool('browser_type', {
            'element': description,
            'ref': element_ref,
            'text': text
        })
    
    async def _wait_for(self, text: Optional[str] = None, time: Optional[float] = None):
        """Wait for text to appear or time to pass."""
        params = {}
        if text:
            params['text'] = text
        if time:
            params['time'] = time
        return await self._call_mcp_tool('browser_wait_for', params)
    
    async def _check_success(self) -> bool:
        """Check if task is completed successfully."""
        snapshot = await self._get_snapshot()
        success_condition = self.task_config.get('success_condition')
        if not success_condition:
            return False
        
        # Handle both dict and string snapshots
        snapshot_str = str(snapshot)
        if isinstance(success_condition, str):
            return success_condition.lower() in snapshot_str.lower()
        return False
    
    async def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial state."""
        self.current_step = 0
        url = self.task_config['url']
        await self._navigate(url)
        # Wait and try to get a non-empty state dict; retry if empty.
        # Use exponential backoff for waiting, up to a max wait of 60 seconds total
        backoff = 0.5
        max_wait = 60.0

        await self._wait_for(time=backoff)

        state = await self._get_snapshot()
        while (not state or not isinstance(state, dict) or not state) and backoff < max_wait:
            print(f"line 102: State: {state}")
            backoff = min(backoff * 2, max_wait)
            if backoff <= 0:
                break
            await self._wait_for(time=backoff)
            state = await self._get_snapshot()
        return state
    
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute action and return (state, reward, done, info).
        
        Args:
            action: dict with keys:
                - type: 'click', 'type', 'submit', 'wait'
                - element_ref: reference to element (from snapshot)
                - text: text to type (if type is 'type')
                - description: human-readable element description
        
        Returns:
            state: accessibility snapshot
            reward: float reward
            done: bool whether episode is done
            info: dict with additional info
        """
        self.current_step += 1
        
        # Execute action
        action_type = action.get('type')
        element_ref = action.get('element_ref', '')
        text = action.get('text', '')
        description = action.get('description', '')
        
        if action_type == 'click':
            await self._click(element_ref, description)
        elif action_type == 'type':
            await self._type(element_ref, text, description)
        elif action_type == 'submit':
            await self._click(element_ref, description or 'submit button')
        elif action_type == 'wait':
            await self._wait_for(time=action.get('time', 0.5))
        
        await self._wait_for(time=0.3)
        state = await self._get_snapshot()
        
        done = False
        reward = -0.01
        success = await self._check_success()
        
        if success:
            reward = 1.0
            done = True
        elif self.current_step >= self.max_steps:
            reward = -1.0
            done = True
        
        info = {'step': self.current_step, 'success': success if done else False, 'action_type': action_type}
        return state, reward, done, info
    
    async def render(self) -> Dict[str, Any]:
        """Get current state snapshot."""
        return await self._get_snapshot()
    
    def close(self):
        """Clean up resources."""
        # Placeholder: close browser/page if needed
        pass

