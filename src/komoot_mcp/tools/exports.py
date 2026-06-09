"""MCP tools for exporting Komoot tours as GPX / FIT."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def download_tour_gpx(tour_id: int) -> str:
        """Telecharger un tour au format GPX (trace GPS complete).

        Args:
            tour_id: ID du tour.

        Returns:
            Contenu GPX en texte XML.
        """
        data = await client.get(f"/tours/{tour_id}.gpx", raw=True)
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    @mcp.tool()
    async def download_tour_fit(tour_id: int) -> str:
        """Telecharger un tour au format FIT (format Garmin/ANT+).

        Args:
            tour_id: ID du tour.

        Returns:
            Indication que le fichier FIT a ete recupere (binaire, non affichable).
        """
        data = await client.get(f"/tours/{tour_id}.fit", raw=True)
        if isinstance(data, bytes):
            return f"FIT file downloaded: {len(data)} bytes for tour {tour_id}"
        return str(data)
