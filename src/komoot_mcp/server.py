"""FastMCP entrypoint for the Komoot connector."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .client import KomootClient
from .tools import (
    register_tours,
    register_tour_search,
    register_exports,
    register_uploads,
    register_highlights,
    register_routing,
    register_streams,
    register_profile,
    register_trail_discovery,
)

mcp = FastMCP("komoot-mcp")
_client = KomootClient()

register_tours(mcp, _client)
register_tour_search(mcp, _client)
register_exports(mcp, _client)
register_uploads(mcp, _client)
register_highlights(mcp, _client)
register_routing(mcp, _client)
register_streams(mcp, _client)
register_profile(mcp, _client)
register_trail_discovery(mcp, _client)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()