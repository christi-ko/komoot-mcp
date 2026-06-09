"""MCP tools for Komoot tour sub-resources (coordinates, surfaces, way types, directions)."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def get_tour_coordinates(tour_id: int) -> dict[str, Any]:
        """Sequence de coordonnees d'un tour (lat, lng, alt, timestamp).

        Args:
            tour_id: ID du tour.
        """
        return await client.get(f"/tours/{tour_id}/coordinates")

    @mcp.tool()
    async def get_tour_surfaces(tour_id: int) -> dict[str, Any]:
        """Types de surface par segment d'un tour (asphalte, gravier, terre, etc.).

        Args:
            tour_id: ID du tour.
        """
        return await client.get(f"/tours/{tour_id}/surfaces")

    @mcp.tool()
    async def get_tour_way_types(tour_id: int) -> dict[str, Any]:
        """Types de voie par segment d'un tour (route, piste cyclable, sentier, etc.).

        Args:
            tour_id: ID du tour.
        """
        return await client.get(f"/tours/{tour_id}/way_types")

    @mcp.tool()
    async def get_tour_directions(tour_id: int) -> dict[str, Any]:
        """Instructions de navigation turn-by-turn d'un tour planifie.

        Args:
            tour_id: ID du tour.
        """
        return await client.get(f"/tours/{tour_id}/directions")
