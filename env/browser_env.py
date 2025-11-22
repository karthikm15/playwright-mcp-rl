"""Minimal browser environment wrapper for Playwright."""

import asyncio
from typing import Dict, Any, Tuple, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page


class BrowserEnv:
    """Environment wrapper for browser form filling tasks using Playwright."""
    
    def __init__(self, task_config: Dict[str, Any], headless: bool = True):
        """
        Initialize with task configuration.
        
        Args:
            task_config: dict with keys:
                - url: target URL
                - field_selector: CSS selector for input field (optional, for reference)
                - submit_selector: CSS selector for submit button (optional)
                - success_condition: text/selector to check for success
                - max_steps: maximum steps per episode
            headless: whether to run browser in headless mode (default: False)
        """
        self.task_config = task_config
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.current_step = 0
        self.max_steps = task_config.get('max_steps', 50)
        self.current_url = None
        self.last_snapshot = None
        self.ref_to_node: Dict[str, Dict[str, Any]] = {}  # Map element_ref to accessibility node
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize browser if not already initialized."""
        if not self._initialized:
            if async_playwright is None:
                raise ImportError("Playwright is not installed. Please install it with: pip install playwright && playwright install chromium")
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            self._initialized = True
    
    def _build_ref_mapping(self, node: Dict[str, Any], ref_counter: list) -> Dict[str, Any]:
        """Recursively build element reference mapping from accessibility tree."""
        if not node:
            return {}
        
        # Generate ref for this node
        ref = f"e{ref_counter[0]}"
        ref_counter[0] += 1
        
        # Store node info
        node_info = {
            'role': node.get('role', ''),
            'name': node.get('name', ''),
            'value': node.get('value', ''),
            'description': node.get('description', ''),
        }
        self.ref_to_node[ref] = node_info
        
        # Build result with ref
        result = {
            'ref': ref,
            'type': node.get('role', ''),
            'name': node.get('name', ''),
            'value': node.get('value', ''),
        }
        
        # Process children
        children = node.get('children', [])
        if children:
            child_results = []
            for child in children:
                if child:  # Skip None children
                    child_result = self._build_ref_mapping(child, ref_counter)
                    if child_result:
                        child_results.append(child_result)
            if child_results:
                result['children'] = child_results
        
        return result
    
    async def _navigate(self, url: str):
        """Navigate to URL using Playwright."""
        await self._ensure_initialized()
        # Wait for page to fully load, including network requests
        await self.page.goto(url, wait_until='networkidle', timeout=30000)
        self.current_url = url
        return {'url': url}
    
    async def _get_snapshot(self) -> Dict[str, Any]:
        """Get accessibility snapshot of current page."""
        await self._ensure_initialized()
        try:
            # Wait a bit for any dynamic content to render
            await asyncio.sleep(0.5)
            snapshot = await self.page.accessibility.snapshot()
            if snapshot:
                # Build ref mapping
                self.ref_to_node = {}
                ref_counter = [1]
                formatted_snapshot = self._build_ref_mapping(snapshot, ref_counter)
                self.last_snapshot = formatted_snapshot
                return formatted_snapshot
        except Exception as e:
            print(f"Error getting accessibility snapshot: {e}")
            import traceback
            traceback.print_exc()
        
        return self.last_snapshot or {}
    
    async def _find_locator_by_ref(self, element_ref: str):
        """Find Playwright locator from element reference."""
        await self._ensure_initialized()
        if element_ref not in self.ref_to_node:
            return None
        
        node_info = self.ref_to_node[element_ref]
        role = node_info.get('role', '')
        name = node_info.get('name', '')
        
        # Try to find element by role and name
        if role and name:
            try:
                return self.page.get_by_role(role, name=name)
            except Exception:
                pass
        
        # Fallback: try by role only
        if role:
            try:
                if role == 'textbox':
                    # For textboxes, try to find by placeholder or label
                    if name:
                        try:
                            return self.page.get_by_placeholder(name)
                        except Exception:
                            try:
                                return self.page.get_by_label(name)
                            except Exception:
                                pass
                elif role == 'button':
                    if name:
                        return self.page.get_by_role('button', name=name)
                elif role == 'radio':
                    if name:
                        return self.page.get_by_role('radio', name=name)
                elif role == 'checkbox':
                    if name:
                        return self.page.get_by_role('checkbox', name=name)
            except Exception:
                pass
        
        return None
    
    async def _click(self, element_ref: str, description: str = ""):
        """Click element using Playwright."""
        await self._ensure_initialized()
        locator = await self._find_locator_by_ref(element_ref)
        if locator:
            await locator.click()
            return {'clicked': element_ref}
        # Fallback: try description-based locator
        if description:
            try:
                if 'button' in description.lower() or 'submit' in description.lower():
                    await self.page.get_by_role('button', name=description).click()
                    return {'clicked': element_ref}
            except Exception:
                pass
        return {'clicked': element_ref}
    
    async def _check(self, element_ref: str, description: str = ""):
        """Check checkbox or radio button using Playwright."""
        await self._ensure_initialized()
        locator = await self._find_locator_by_ref(element_ref)
        if locator:
            # For radio buttons, use click instead of check (Google Forms uses custom radio buttons)
            node_info = self.ref_to_node.get(element_ref, {})
            role = node_info.get('role', '')
            if role == 'radio':
                await locator.click()
            else:
                try:
                    await locator.check()
                except Exception:
                    # If check fails, try click as fallback
                    await locator.click()
            return {'checked': element_ref}
        # Fallback: try description-based locator
        if description:
            try:
                # Try to find by role and name
                node_info = self.ref_to_node.get(element_ref, {})
                role = node_info.get('role', '')
                name = node_info.get('name', '')
                if role == 'radio' and name:
                    # Use click for radio buttons (Google Forms)
                    await self.page.get_by_role('radio', name=name).click()
                    return {'checked': element_ref}
                elif role == 'checkbox' and name:
                    try:
                        await self.page.get_by_role('checkbox', name=name).check()
                    except Exception:
                        await self.page.get_by_role('checkbox', name=name).click()
                    return {'checked': element_ref}
            except Exception:
                pass
        
        # Fallback: just click it
        return await self._click(element_ref, description)
    
    async def _type(self, element_ref: str, text: str, description: str = ""):
        """Type text into element using Playwright."""
        await self._ensure_initialized()
        locator = await self._find_locator_by_ref(element_ref)
        if locator:
            await locator.fill(text)
            return {'typed': element_ref, 'text': text}
        # Fallback: try description-based locator
        if description:
            try:
                await self.page.get_by_placeholder(description).fill(text)
                return {'typed': element_ref, 'text': text}
            except Exception:
                pass
        return {'typed': element_ref, 'text': text}
    
    async def _wait_for(self, text: Optional[str] = None, time: Optional[float] = None):
        """Wait for text to appear or time to pass."""
        await self._ensure_initialized()
        if time:
            await asyncio.sleep(time)
            return {'waited': True}
        elif text:
            try:
                await self.page.wait_for_selector(f"text={text}", timeout=5000)
                return {'waited': True}
            except Exception:
                pass
        return {'waited': True}
    
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
        await self._ensure_initialized()
        self.current_step = 0
        url = self.task_config['url']
        await self._navigate(url)
        
        # Wait for page to be fully interactive
        try:
            # Wait for body to be visible (indicates page loaded)
            await self.page.wait_for_selector('body', timeout=10000)
            # Additional wait for any dynamic content
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f"Warning: Timeout waiting for page load: {e}")
        
        # Wait and try to get a non-empty state dict with children; retry if empty.
        # Use exponential backoff for waiting, up to a max wait of 60 seconds total
        backoff = 0.5
        max_wait = 60.0

        state = await self._get_snapshot()
        
        # Check if state has children (actual form elements)
        def has_children(state_obj):
            if isinstance(state_obj, dict):
                if 'children' in state_obj and state_obj['children']:
                    return True
                # Check if any key contains children
                for value in state_obj.values():
                    if has_children(value):
                        return True
            elif isinstance(state_obj, list):
                for item in state_obj:
                    if has_children(item):
                        return True
            return False
        
        while (not state or not isinstance(state, dict) or not has_children(state)) and backoff < max_wait:
            print(f"Waiting for page to load... (backoff: {backoff}s)")
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
                - type: 'click', 'type', 'submit', 'wait', 'check'
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
        elif action_type == 'check':
            await self._check(element_ref, description)
        elif action_type == 'submit':
            await self._click(element_ref, description or 'submit button')
        elif action_type == 'wait':
            await self._wait_for(time=action.get('time', 0.5))
        
        await self._wait_for(time=0.3)
        state = await self._get_snapshot()
        
        done = False
        reward = -0.1
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
    
    async def close(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False

