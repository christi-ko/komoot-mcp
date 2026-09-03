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
    async def import_gpx_file(
        file_path: str,
        sport: str = "mtb",
    ) -> dict[str, Any]:
        """Import and analyze a local GPX file using Komoot's matching service.

        Reads a local GPX file, matches it against Komoot's routing network,
        and returns a compact route analysis. This tool does not create a
        planned Komoot tour.

        Args:
            file_path: path to the local GPX file.
            sport: sport type used for Komoot matching (default: mtb).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                gpx_content = f.read()
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": f"Failed to read GPX file: {str(e)}"}

        # First upload the GPX to Komoot's routing import service.
        import_result = await client.post_www(
            "/routing/import/files/",
            data=gpx_content.encode("utf-8"),
            headers={"Content-Type": "application/gpx+xml"},
            params={"data_type": "gpx"},
        )

        if not isinstance(import_result, dict):
            return {"raw": str(import_result)}

        try:
            item = import_result["_embedded"]["items"][0]
        except (KeyError, IndexError, TypeError):
            return {
                "error": "Komoot returned no import item",
                "response": import_result,
            }

        # Match the imported track against Komoot's routing network.
        matched = await client.post_www(
            "/routing/import/tour",
            json_body=item,
            params={
                "sport": sport,
                "_embedded": "way_types,surfaces,directions,coordinates",
            },
        )

        if not isinstance(matched, dict):
            return {"raw": str(matched)}

        # The actual tour data lives under _embedded.matched, not at top level.
        tour = matched.get("_embedded", {})
        if not isinstance(tour, dict):
            tour = {}
        tour = tour.get("matched", {})

        result: dict[str, Any] = {
            "file_path": file_path,
            "sport": tour.get("sport", sport),
            "status": "success",
        }

        # Basic route information.
        if "name" in tour:
            result["name"] = tour["name"]

        if "distance" in tour:
            result["distance_km"] = round(float(tour["distance"]) / 1000.0, 2)

        if "elevation_up" in tour:
            result["elevation_up_m"] = round(float(tour["elevation_up"]), 1)

        if "elevation_down" in tour:
            result["elevation_down_m"] = round(float(tour["elevation_down"]), 1)

        if "duration" in tour:
            duration_seconds = int(float(tour["duration"]))
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60

            if hours:
                result["duration"] = f"{hours}h {minutes}m"
            else:
                result["duration"] = f"{minutes}m"

        difficulty = tour.get("difficulty")
        if isinstance(difficulty, dict) and difficulty.get("grade"):
            result["difficulty"] = difficulty["grade"]

        # Komoot already provides normalized surface and way-type percentages.
        summary = tour.get("summary", {})
        if isinstance(summary, dict):
            surfaces = summary.get("surfaces", [])
            if isinstance(surfaces, list):
                result["surfaces"] = {
                    str(item["type"]).replace("sm#", ""): round(
                        float(item["amount"]) * 100, 1
                    )
                    for item in surfaces
                    if isinstance(item, dict)
                    and "type" in item
                    and "amount" in item
                }

            way_types = summary.get("way_types", [])
            if isinstance(way_types, list):
                result["way_types"] = {
                    str(item["type"]).replace("wt#", ""): round(
                        float(item["amount"]) * 100, 1
                    )
                    for item in way_types
                    if isinstance(item, dict)
                    and "type" in item
                    and "amount" in item
                }

        # Highlight special route information such as off-grid sections
        # or places where Komoot recommends dismounting.
        tour_information = tour.get("tour_information", [])
        if isinstance(tour_information, list):
            special_sections = []

            for info in tour_information:
                if not isinstance(info, dict):
                    continue

                info_type = info.get("type")
                segments = info.get("segments", [])

                if info_type:
                    special_sections.append({
                        "type": str(info_type),
                        "segments": len(segments)
                        if isinstance(segments, list)
                        else 0,
                    })

            if special_sections:
                result["tour_information"] = special_sections

        # Number of matched coordinates, without returning thousands of
        # coordinates to the language model.
        tour_embedded = tour.get("_embedded", {})
        if isinstance(tour_embedded, dict):
            coordinates = tour_embedded.get("coordinates", {})
            if isinstance(coordinates, dict):
                coordinate_items = coordinates.get("items", [])
                if isinstance(coordinate_items, list):
                    result["matched_coordinates"] = len(coordinate_items)

        # Report routed/manual portions of the imported track.
        segments = tour.get("segments", [])
        if isinstance(segments, list):
            routed = sum(
                1 for segment in segments
                if isinstance(segment, dict) and segment.get("type") == "Routed"
            )
            manual = sum(
                1 for segment in segments
                if isinstance(segment, dict) and segment.get("type") == "Manual"
            )

            result["segments"] = {
                "routed": routed,
                "manual": manual,
            }

        return result

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
