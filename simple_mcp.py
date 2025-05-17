from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import asyncio
import os

@asynccontextmanager
async def simple_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """
    Simple lifespan for the MCP server.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        None: No context needed for this simple server
    """
    try:
        yield None
    finally:
        pass

# Initialize FastMCP server
mcp = FastMCP(
    "simple-mcp",
    description="A super simple MCP server that answers who is the best",
    lifespan=simple_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)        

@mcp.tool()
async def who_is_the_best(ctx: Context) -> str:
    """Answers who is the best.
    
    Call this tool when you need to know who is the best.

    Args:
        ctx: The MCP server provided context
    
    Returns:
        A string indicating who is the best
    """
    return "yoyoyo"

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())