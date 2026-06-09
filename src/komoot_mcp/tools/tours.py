"""MCP tools for Komoot tours — list, get, update, delete."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def list_tours(
        tour_type: str | None = None,
        sport_types: str | None = None,
        status: str | None = None,
        sort_field: str = "date",
        sort_direction: str = "desc",
        name: str | None = None,
        page: int = 0,
        limit: int = 30,
    ) -> dict[str, Any]:
        """Lister tes tours Komoot (planifies et enregistres).

        Args:
            tour_type: 'tour_planned', 'tour_recorded', ou None pour tous.
            sport_types: types de sport separes par virgule (ex: 'hike,touringbicycle,mtb,racebicycle,jogging').
            status: 'public', 'private', ou None pour tous.
            sort_field: champ de tri ('date', 'name', 'distance', 'duration').
            sort_direction: 'asc' ou 'desc'.
            name: filtrer par nom (recherche partielle).
            page: pagination (0-indexee).
            limit: nb max (defaut 30, max 100).
        """
        return await client.get(
            f"/users/{client.user_id}/tours/",
            type=tour_type,
            sport_types=sport_types,
            status=status,
            sort_field=sort_field,
            sort_direction=sort_direction,
            name=name,
            page=page,
            limit=min(limit, 100),
        )

    @mcp.tool()
    async def get_tour(tour_id: int) -> dict[str, Any]:
        """Detail complet d'un tour (distance, denivele, duree, sport, difficulte, surfaces, etc.).

        Args:
            tour_id: ID du tour Komoot.
        """
        return await client.get(f"/tours/{tour_id}")

    @mcp.tool()
    async def update_tour(
        tour_id: int,
        name: str | None = None,
        sport: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Modifier les metadonnees d'un tour.

        Args:
            tour_id: ID du tour.
            name: nouveau nom.
            sport: type de sport ('hike', 'touringbicycle', 'mtb', 'racebicycle',
                   'jogging', 'mountaineering', 'running', 'e_touringbicycle',
                   'e_mtb', 'nordic_walking', 'skitour', 'snowshoe').
            status: 'public' ou 'private'.
        """
        return await client.patch(
            f"/tours/{tour_id}",
            name=name,
            sport=sport,
            status=status,
        )

    @mcp.tool()
    async def delete_tour(tour_id: int) -> dict[str, str]:
        """Supprimer un tour.

        Args:
            tour_id: ID du tour a supprimer.
        """
        await client.delete(f"/tours/{tour_id}")
        return {"deleted": str(tour_id)}
