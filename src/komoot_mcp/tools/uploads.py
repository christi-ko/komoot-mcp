"""MCP tools for uploading tours to Komoot."""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def upload_tour(
        gpx_content: str,
        sport: str = "hike",
        tour_name: str | None = None,
        data_type: str = "gpx",
    ) -> dict[str, Any]:
        """Uploader un fichier GPX comme activite enregistree sur Komoot.

        Args:
            gpx_content: contenu du fichier GPX (texte XML).
            sport: type de sport ('hike', 'touringbicycle', 'mtb', 'racebicycle',
                   'jogging', 'mountaineering', 'running').
            tour_name: nom du tour (optionnel).
            data_type: 'gpx' ou 'fit'.
        """
        params = {"data_type": data_type}
        headers = {"Content-Type": "application/gpx+xml"}
        result = await client.post(
            f"/tours/",
            data=gpx_content.encode("utf-8"),
            headers=headers,
            params=params,
        )
        # Si un nom est fourni, on met a jour le tour cree
        if tour_name and isinstance(result, dict) and "id" in result:
            await client.patch(f"/tours/{result['id']}", name=tour_name, sport=sport)
        return result
