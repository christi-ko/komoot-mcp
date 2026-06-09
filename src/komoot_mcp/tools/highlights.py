"""MCP tools for Komoot highlights, tips and tour images."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def get_highlight(highlight_id: int) -> dict[str, Any]:
        """Detail d'un highlight Komoot (point d'interet communautaire).

        Args:
            highlight_id: ID du highlight.
        """
        return await client.get(f"/highlights/{highlight_id}")

    @mcp.tool()
    async def get_highlight_tips(highlight_id: int) -> dict[str, Any]:
        """Conseils/tips de la communaute pour un highlight.

        Args:
            highlight_id: ID du highlight.
        """
        return await client.get(f"/highlights/{highlight_id}/tips/")

    @mcp.tool()
    async def get_tour_images(tour_id: int) -> dict[str, Any]:
        """Lister les images/photos d'un tour.

        Args:
            tour_id: ID du tour.
        """
        return await client.get(f"/tours/{tour_id}/cover_images/")
