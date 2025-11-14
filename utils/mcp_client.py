"""MCP client using urllib requests."""

import json
import asyncio
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from typing import Dict, Any, Optional


class MCPClient:
    """MCP client wrapper for Playwright browser tools."""
    
    def __init__(self, url: str, session_id: Optional[str] = None):
        self.url = url
        self.session_id = session_id
        self.initialized = False
    
    @classmethod
    async def create(cls, url: str = "http://localhost:8931/mcp"):
        """Create MCP client connected to Playwright server."""
        client = cls(url)
        await client.initialize()
        return client
    
    async def _post_json(self, payload: Dict[str, Any]) -> tuple:
        """Post JSON-RPC request and parse SSE response."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        
        req = Request(self.url, data=data, headers=headers, method="POST")
        
        try:
            resp = urlopen(req)
            body = resp.read().decode("utf-8")
            status = resp.status
            resp_headers = dict(resp.headers)
        except HTTPError as e:
            body = e.read().decode("utf-8")
            status = e.code
            resp_headers = dict(e.headers)
        
        # Parse SSE response
        msg = None
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str:
                    try:
                        msg = json.loads(data_str)
                    except json.JSONDecodeError:
                        pass
                    break
        
        # Extract session ID
        if not self.session_id:
            self.session_id = (
                resp_headers.get("mcp-session-id")
                or resp_headers.get("Mcp-Session-Id")
                or resp_headers.get("MCP-SESSION-ID")
            )
        
        return status, msg
    
    async def initialize(self):
        """Initialize MCP session."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "playwright-rl", "version": "0.1.0"},
            },
        }
        status, result = await self._post_json(payload)
        if status != 200:
            raise Exception(f"Initialize failed: {status}")
        
        # Send initialized notification
        await self._post_json({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        })
        self.initialized = True
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call MCP tool."""
        if not self.initialized:
            await self.initialize()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
        }
        status, result = await self._post_json(payload)
        if status != 200 or not result:
            return None
        
        # Extract result content
        if "result" in result:
            result_data = result["result"]
            if "content" in result_data:
                content = result_data["content"]
                if content and len(content) > 0:
                    text = content[0].get("text", "")
                    try:
                        return json.loads(text)
                    except:
                        return text
            return result_data
        return None
    
    async def close(self):
        """Close client."""
        pass