"""Playwright browser client using Playwright Python library."""

from typing import Dict, Any, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page


class MCPClient:
    """Playwright browser client wrapper (maintains MCPClient interface for compatibility)."""
    
    def __init__(self, browser: Optional[Browser] = None, context: Optional[BrowserContext] = None, page: Optional[Page] = None):
        self.browser = browser
        self.context = context
        self.page = page
        self.playwright = None
        self.initialized = False
    
    @classmethod
    async def create(cls, url: str = None):
        """Create Playwright browser client."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        client = cls(browser, context, page)
        client.playwright = playwright
        client.initialized = True
        return client
    
    async def initialize(self):
        """Initialize browser (already done in create, but kept for compatibility)."""
        if not self.initialized:
            await self.create()
        self.initialized = True
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call browser tool (maps MCP tool names to Playwright methods)."""
        if not self.initialized or not self.page:
            await self.initialize()
        
        if tool_name == 'browser_navigate':
            url = params.get('url')
            if url:
                await self.page.goto(url, wait_until='domcontentloaded')
                return {'url': url}
        
        elif tool_name == 'browser_snapshot':
            # Get accessibility snapshot
            snapshot = await self.page.accessibility.snapshot()
            return snapshot
        
        elif tool_name == 'browser_click':
            ref = params.get('ref')
            if ref:
                # Find element by ref and click
                # We'll need to maintain a ref-to-locator mapping
                # For now, this is a placeholder - will be handled in BrowserEnv
                return {'clicked': ref}
        
        elif tool_name == 'browser_type':
            ref = params.get('ref')
            text = params.get('text', '')
            if ref and text:
                # Find element by ref and type
                # Will be handled in BrowserEnv
                return {'typed': ref, 'text': text}
        
        elif tool_name == 'browser_wait_for':
            if 'time' in params:
                import asyncio
                await asyncio.sleep(params['time'])
            elif 'text' in params:
                await self.page.wait_for_selector(f"text={params['text']}", timeout=5000)
            return {'waited': True}
        
        return None
    
    async def close(self):
        """Close browser and cleanup."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()