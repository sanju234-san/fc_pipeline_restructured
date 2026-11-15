"""Internal async MCP client bridge for calling the FastMCP server in-process."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastmcp import Client

from fc_pipeline.mcp_server.server import mcp


def _extract_result(res: Any) -> Any:
    """Normalize FastMCP CallToolResult to plain dict/str."""
    if hasattr(res, "data") and res.data is not None:
        return res.data
    content = getattr(res, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return text
    return res


async def call_mcp_tool(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    timeout_seconds: float = 600.0,
) -> Any:
    """Call an internal FastMCP tool concurrently over an isolated async Client session."""
    async with Client(mcp, timeout=timeout_seconds) as client:
        res = await client.call_tool(tool_name, arguments or {}, timeout=timeout_seconds)
        return _extract_result(res)


async def list_mcp_tools() -> list[str]:
    """List all tool names exposed by the internal FastMCP server."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        return [t.name for t in tools]

