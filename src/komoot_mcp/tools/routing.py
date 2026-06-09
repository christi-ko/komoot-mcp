"""MCP tools for Komoot route planning and import.

Ces endpoints passent par www.komoot.com/api (pas api.komoot.de).
Inspires du projet export-komoot (Go) de pieterclaerhout.
"""

from __future__ import annotations

from typing import Any

from ..client import KomootClient


def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def import_gpx_route(
        gpx_content: str,
        sport: str = "hike",
    ) -> dict[str, Any]:
        """Importer un fichier GPX et le matcher sur le reseau routier Komoot.

        Etape 1/2 pour creer un tour planifie a partir d'un GPX externe.
        Utiliser ensuite create_planned_tour() avec le resultat.

        Args:
            gpx_content: contenu GPX (texte XML).
            sport: type de sport pour le matching.
        """
        # Etape 1 : upload du fichier
        import_result = await client.post_www(
            "/routing/import/files/",
            data=gpx_content.encode("utf-8"),
            headers={"Content-Type": "application/gpx+xml"},
            params={"data_type": "gpx"},
        )
        if not isinstance(import_result, dict):
            return {"raw": str(import_result)}

        # Etape 2 : route matching
        matched = await client.post_www(
            "/routing/import/tour",
            json_body=import_result,
            params={
                "sport": sport,
                "_embedded": "way_types,surfaces,directions,coordinates",
            },
        )
        return matched

    @mcp.tool()
    async def plan_route(
        coordinates: list[list[float]],
        sport: str = "hike",
    ) -> dict[str, Any]:
        """Planifier un itineraire Komoot a partir de waypoints.

        Args:
            coordinates: liste de [lat, lng] pour les points de passage
                         (ex: [[48.5734, 7.7521], [48.5801, 7.7612]]).
            sport: type de sport ('hike', 'touringbicycle', 'mtb', 'racebicycle', 'jogging').
        """
        # Construire le payload de routing Komoot
        segments = []
        for i, coord in enumerate(coordinates):
            point = {"lat": coord[0], "lng": coord[1], "alt": 0.0}
            if i == 0:
                segments.append({"type": "start", "location": point})
            elif i == len(coordinates) - 1:
                segments.append({"type": "end", "location": point})
            else:
                segments.append({"type": "via", "location": point})

        return await client.post_www(
            "/routing/tour",
            json_body={"segments": segments},
            params={
                "sport": sport,
                "_embedded": "coordinates,way_types,surfaces,directions",
            },
        )

    @mcp.tool()
    async def create_planned_tour(
        route_data: dict[str, Any],
        name: str = "Tour planifie",
        sport: str = "hike",
    ) -> dict[str, Any]:
        """Creer un tour planifie sur Komoot a partir d'un resultat de plan_route ou import_gpx_route.

        Args:
            route_data: resultat JSON de plan_route() ou import_gpx_route().
            name: nom du tour planifie.
            sport: type de sport.
        """
        route_data["name"] = name
        route_data["sport"] = sport
        return await client.post_www(
            "/v007/tours/",
            json_body=route_data,
            params={"reroute": "true"},
        )
