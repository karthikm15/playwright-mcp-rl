import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

MCP_URL = "http://localhost:8931/mcp"

def post_json(payload, session_id=None):
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req = Request(MCP_URL, data=data, headers=headers, method="POST")

    try:
        resp = urlopen(req)
        body = resp.read().decode("utf-8")
        status = resp.status
        headers = dict(resp.headers)
    except HTTPError as e:
        body = e.read().decode("utf-8")
        status = e.code
        headers = dict(e.headers)

    print(f"Status: {status}")
    print(f"Response: {body}")
    print()

    msg = None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data:
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    print(f"line 40: Failed to parse data as JSON: {data}")
                    return
                break
    return status, headers, msg


def main():
    # 1. initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "minimal-python-client",
                "version": "0.1.0",
            },
        },
    }

    print("### INITIALIZE ###")
    status, headers, init_result = post_json(init_payload)

    if status != 200:
        print("Initialize failed, stop here.")
        return

    session_id = (
        headers.get("mcp-session-id")
        or headers.get("Mcp-Session-Id")
        or headers.get("MCP-SESSION-ID")
    )

    print(f"Session ID: {session_id}")
    if not session_id:
        print("No session id returned, stop here.")
        return

    # 2. send initialized notification (no id for notification)
    initialized_payload = {
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {},
    }

    print("### INITIALIZED ###")
    post_json(initialized_payload, session_id=session_id)

    # 3. list tools
    list_tools_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }

    print("### TOOLS/LIST ###")
    status, _, tools_result = post_json(list_tools_payload, session_id=session_id)
    if status != 200:
        print("tools/list failed.")
        return

    tools = tools_result.get("result", {}).get("tools", [])
    print("Tools:")
    print(json.dumps(tools, indent=2))

    if not tools:
        print("No tools available to call.")
        return

    first_tool = "browser_navigate"
    print(f"Using tool: {first_tool}")

    # 4. call the first tool with empty arguments
    call_payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": first_tool,
            "arguments": {"url": "https://www.google.com"},
        },
    }

    print("### TOOL CALL ###")
    post_json(call_payload, session_id=session_id)


if __name__ == "__main__":
    main()
