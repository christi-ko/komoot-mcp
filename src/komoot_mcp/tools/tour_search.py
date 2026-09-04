"""MCP tool for searching existing Komoot tours by geography.

Builds on the existing list_tours() API endpoint (/users/{user_id}/tours/)
with geographic parameters (center, max_distance, sport_types, proximity
sorting). Adds pure-function filtering for roundtrip detection,
start-to-end matching, and corridor search.

Flow:
  search_tours() -> returns structured tour list
    -> user picks tours -> discover_trail_segments(tour_ids=[...])
    -> derive_routing_waypoints() -> plan_route()

Supports four search modes:
  A) Radius search       — center_lat/lng + radius_km
  B) Roundtrip search    — center_lat/lng + radius_km + route_type=roundtrip
  C) Start-to-end search — start_lat/lng + end_lat/lng
  D) Corridor search     — start_lat/lng + end_lat/lng + corridor_km
"""

from __future__ import annotations

import math
from typing import Any

from ..client import KomootClient, KomootError
from .trail_discovery import _haversine_km

# ── Roundtrip detection ────────────────────────────────────────────────

_ROUNDTRIP_TOLERANCE = 0.05    # 5 % of tour length
_ROUNDTRIP_MIN_KM = 0.5        # absolute minimum tolerance


def _is_roundtrip(
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    tour_distance_km: float,
) -> tuple[bool, float]:
    """Determine if a tour is a roundtrip based on start/end proximity.

    A tour is a roundtrip when the straight-line distance between its
    start point and end point is small relative to its total length.

    Formula:
        deviation_km = haversine(start, end)
        tolerance_km = max(tour_distance_km * 0.05, 0.5)
        is_roundtrip  = deviation_km <= tolerance_km

    This scales with tour length:
        - 5 km tour   -> tolerance 0.50 km
        - 40 km tour  -> tolerance 2.00 km
        - 100 km tour -> tolerance 5.00 km

    Returns:
        (is_roundtrip, deviation_km)
    """
    deviation = _haversine_km(start_lat, start_lng, end_lat, end_lng)
    tolerance = max(tour_distance_km * _ROUNDTRIP_TOLERANCE, _ROUNDTRIP_MIN_KM)
    return deviation <= tolerance, round(deviation, 4)


# ── Corridor geometry helpers ──────────────────────────────────────────

def _cross_track_distance_km(
    p_lat: float, p_lng: float,
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float,
) -> float:
    """Cross-track (perpendicular) distance from point P to the great-circle
    arc from A to B.

    Uses the spherical cross-track formula:
        d_xt = asin(sin(d_AP / R) * sin(theta_AP - theta_AB)) * R

    When A == B, returns haversine distance from P to A.

    Returns positive distance in km, or 0 when points are colinear.
    """
    R = 6371.0

    # Degenerate case: A == B
    if a_lat == b_lat and a_lng == b_lng:
        return _haversine_km(p_lat, p_lng, a_lat, a_lng)

    # Convert to radians
    p_lat_r = math.radians(p_lat)
    p_lng_r = math.radians(p_lng)
    a_lat_r = math.radians(a_lat)
    a_lng_r = math.radians(a_lng)
    b_lat_r = math.radians(b_lat)
    b_lng_r = math.radians(b_lng)

    # Distance from A to P
    d_lat_ap = p_lat_r - a_lat_r
    d_lng_ap = p_lng_r - a_lng_r
    a = (math.sin(d_lat_ap / 2) ** 2
         + math.cos(a_lat_r) * math.cos(p_lat_r)
         * math.sin(d_lng_ap / 2) ** 2)
    d_ap = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Bearing from A to P
    y_ap = math.sin(d_lng_ap) * math.cos(p_lat_r)
    x_ap = (math.cos(a_lat_r) * math.sin(p_lat_r)
            - math.sin(a_lat_r) * math.cos(p_lat_r) * math.cos(d_lng_ap))
    theta_ap = math.atan2(y_ap, x_ap)

    # Bearing from A to B
    d_lng_ab = b_lng_r - a_lng_r
    y_ab = math.sin(d_lng_ab) * math.cos(b_lat_r)
    x_ab = (math.cos(a_lat_r) * math.sin(b_lat_r)
            - math.sin(a_lat_r) * math.cos(b_lat_r) * math.cos(d_lng_ab))
    theta_ab = math.atan2(y_ab, x_ab)

    # Cross-track distance
    d_xt = math.asin(max(-1, min(1,
        math.sin(d_ap / R) * math.sin(theta_ap - theta_ab)
    ))) * R

    return max(0.0, abs(d_xt))


def _point_to_line_segment_distance_km(
    p_lat: float, p_lng: float,
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float,
) -> float:
    """Shortest distance from point P to the great-circle line segment AB.

    Uses the cross-track distance and bearing-based along-track sign to
    determine whether P projects between A and B or beyond an endpoint.
    """
    R = 6371.0

    # Degenerate cases
    if a_lat == b_lat and a_lng == b_lng:
        return _haversine_km(p_lat, p_lng, a_lat, a_lng)

    # Convert to radians
    p_lat_r = math.radians(p_lat)
    p_lng_r = math.radians(p_lng)
    a_lat_r = math.radians(a_lat)
    a_lng_r = math.radians(a_lng)
    b_lat_r = math.radians(b_lat)
    b_lng_r = math.radians(b_lng)

    # Distance from A to P and A to B
    d_ap_km = _haversine_km(a_lat, a_lng, p_lat, p_lng)
    d_ab_km = _haversine_km(a_lat, a_lng, b_lat, b_lng)
    d_bp_km = _haversine_km(b_lat, b_lng, p_lat, p_lng)

    # If P is very close to A or B, return that distance directly
    if d_ap_km < 1e-6 or d_bp_km < 1e-6:
        return min(d_ap_km, d_bp_km)

    # Bearing from A to P
    d_lng_ap = p_lng_r - a_lng_r
    y_ap = math.sin(d_lng_ap) * math.cos(p_lat_r)
    x_ap = (math.cos(a_lat_r) * math.sin(p_lat_r)
            - math.sin(a_lat_r) * math.cos(p_lat_r) * math.cos(d_lng_ap))
    theta_ap = math.atan2(y_ap, x_ap)

    # Bearing from A to B
    d_lng_ab = b_lng_r - a_lng_r
    y_ab = math.sin(d_lng_ab) * math.cos(b_lat_r)
    x_ab = (math.cos(a_lat_r) * math.sin(b_lat_r)
            - math.sin(a_lat_r) * math.cos(b_lat_r) * math.cos(d_lng_ab))
    theta_ab = math.atan2(y_ab, x_ab)

    # Along-track distance sign: positive if P projects towards B, negative if away
    at_sign = math.cos(theta_ap - theta_ab)
    at_dist = d_ap_km * at_sign

    # If projection falls before A (at_dist < 0) or beyond B, return endpoint distance
    if at_dist < 0:
        return d_ap_km
    if at_dist > d_ab_km:
        return d_bp_km

    # Projection falls between A and B: return perpendicular (cross-track) distance
    d_xt = _cross_track_distance_km(p_lat, p_lng, a_lat, a_lng, b_lat, b_lng)
    return d_xt


def _max_corridor_distance_km(
    coords: list[dict[str, Any]],
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float,
) -> float:
    """Maximum distance of any coordinate in a tour from the corridor line AB.

    Args:
        coords: list of {lat, lng} items from a tour response.
        a_lat, a_lng: corridor start.
        b_lat, b_lng: corridor end.

    Returns:
        Maximum perpendicular distance in km (0 if coords empty).
    """
    if not coords:
        return 0.0
    max_d = 0.0
    for c in coords:
        try:
            lat = float(c.get("lat", 0))
            lng = float(c.get("lng", 0))
        except (TypeError, ValueError):
            continue
        d = _point_to_line_segment_distance_km(lat, lng, a_lat, a_lng, b_lat, b_lng)
        if d > max_d:
            max_d = d
    return max_d


def _generate_corridor_centers(
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    corridor_km: float,
) -> list[tuple[float, float]]:
    """Generate search center points along the corridor from A to B.

    Strategy:
      - Compute the total distance between A and B.
      - Space centers so their search circles (radius=corridor_km) overlap
        by at least 50 %, i.e. step = corridor_km * 0.8.
      - Minimum 2 centers (start and end), maximum 8.

    Returns:
        List of (lat, lng) tuples.
    """
    total_km = _haversine_km(start_lat, start_lng, end_lat, end_lng)

    if total_km < 0.1:
        return [(start_lat, start_lng)]

    # Step size: 80 % of corridor radius to ensure overlap
    step_km = max(corridor_km * 0.8, 1.0)
    n_centers = max(2, min(8, int(math.ceil(total_km / step_km)) + 1))

    centers: list[tuple[float, float]] = []
    for i in range(n_centers):
        fraction = i / (n_centers - 1) if n_centers > 1 else 0.0
        lat = start_lat + (end_lat - start_lat) * fraction
        lng = start_lng + (end_lng - start_lng) * fraction
        centers.append((round(lat, 6), round(lng, 6)))

    return centers


# ── Deduplication ──────────────────────────────────────────────────────

def _deduplicate_tours(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate tour results by tour_id, preserving first occurrence."""
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        tid = r.get("id") or r.get("tour_id")
        if tid is not None and tid not in seen:
            seen.add(tid)
            deduped.append(r)
    return deduped


# ── Result building ────────────────────────────────────────────────────

def _extract_tour_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the tour list from a Komoot tours API response.

    The API wraps tours under _embedded.tours (HAL format).
    """
    if not isinstance(raw, dict):
        return []
    embedded = raw.get("_embedded", {})
    if not isinstance(embedded, dict):
        return []
    items = embedded.get("tours", [])
    return items if isinstance(items, list) else []


def _make_compact_result(
    raw: dict[str, Any],
    *,
    distance_from_center_km: float | None = None,
    roundtrip_score: dict | None = None,
    start_distance_km: float | None = None,
    end_distance_km: float | None = None,
    corridor_distance_km: float | None = None,
) -> dict[str, Any]:
    """Build a compact tour result entry from the raw API item.

    Only includes fields relevant to search; keeps the output readable.
    """
    tour_id = raw.get("id")
    result: dict[str, Any] = {
        "tour_id": tour_id,
        "name": raw.get("name", ""),
        "sport": raw.get("sport", ""),
        "type": raw.get("type", ""),
        "status": raw.get("status", ""),
        "distance_km": round(
            (raw.get("distance", 0) or 0) / 1000.0, 2
        ),
    }

    # Optional metrics from the API response
    for key in ("elevation_up", "elevation_down", "duration"):
        val = raw.get(key)
        if val is not None:
            result[key] = val

    # Coordinates — prefer start_point from list response, fall back to _embedded.coordinates
    start_point = raw.get("start_point")
    if isinstance(start_point, dict) and start_point.get("lat") is not None:
        try:
            result["start"] = {
                "lat": float(start_point["lat"]),
                "lng": float(start_point["lng"]),
            }
        except (TypeError, ValueError):
            pass
    else:
        coords = raw.get("_embedded", {}).get("coordinates", {})
        if isinstance(coords, dict) and coords.get("items"):
            items = coords["items"]
            if items:
                try:
                    result["start"] = {
                        "lat": float(items[0].get("lat", 0)),
                        "lng": float(items[0].get("lng", 0)),
                    }
                    result["end"] = {
                        "lat": float(items[-1].get("lat", 0)),
                        "lng": float(items[-1].get("lng", 0)),
                    }
                except (TypeError, ValueError):
                    pass

    # Search-mode-specific distances
    if distance_from_center_km is not None:
        result["distance_from_center_km"] = round(distance_from_center_km, 3)
    if roundtrip_score is not None:
        result["roundtrip_deviation_km"] = roundtrip_score["deviation_km"]
        result["is_roundtrip"] = roundtrip_score["is_roundtrip"]
    if start_distance_km is not None:
        result["start_distance_km"] = round(start_distance_km, 3)
    if end_distance_km is not None:
        result["end_distance_km"] = round(end_distance_km, 3)
    if corridor_distance_km is not None:
        result["corridor_distance_km"] = round(corridor_distance_km, 3)

    return result


# ── MCP tool registration ──────────────────────────────────────────────

def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def search_tours(
        sport: str = "mtb",
        center_lat: float | None = None,
        center_lng: float | None = None,
        radius_km: float | None = None,
        start_lat: float | None = None,
        start_lng: float | None = None,
        end_lat: float | None = None,
        end_lng: float | None = None,
        corridor_km: float | None = None,
        route_type: str = "any",
        tour_type: str | None = None,
        limit: int = 20,
        page: int = 0,
        compact: bool = True,
    ) -> dict[str, Any]:
        """Search for existing Komoot tours by geographic criteria.

        Four search modes, selected automatically based on parameters:

        A) Radius search:
           Provide center_lat, center_lng, radius_km.
           Returns tours within the circle, sorted by proximity.

        B) Roundtrip search:
           Like radius search, plus route_type="roundtrip".
           Filters to tours whose start and end points are close
           (indicating a loop). Tolerance scales with tour length.

        C) Start-to-end search:
           Provide start_lat/lng and end_lat/lng.
           Finds tours whose actual start/end are close to the
           requested points. Uses a single radius search around
           the midpoint of start→end.

        D) Corridor search:
           Like start-to-end, plus corridor_km.
           Performs multiple radius searches along the direct line
           from start to end, then filters tours by their maximum
           perpendicular distance from that line.

        Args:
            sport: Sport type(s), comma-separated ('mtb', 'e_mtb',
                   'touringbicycle', etc.). Default 'mtb'.
            center_lat: Search center latitude (mode A/B).
            center_lng: Search center longitude (mode A/B).
            radius_km: Search radius in km (mode A/B).
            start_lat: Desired start latitude (mode C/D).
            start_lng: Desired start longitude (mode C/D).
            end_lat: Desired end latitude (mode C/D).
            end_lng: Desired end longitude (mode C/D).
            corridor_km: Max corridor width in km (mode D).
            route_type: 'any' (default), 'roundtrip', or 'one_way'.
            tour_type: 'tour_planned', 'tour_recorded', or None (both).
            limit: Max results per page (default 20, max 50).
            page: Page number (0-indexed).
            compact: If True (default), returns clean result list
                     without raw API metadata.

        Returns:
            {
                status: "success",
                search_mode: "radius" | "roundtrip" | "start_to_end" | "corridor",
                total_found: N,
                page: int,
                limit: int,
                results: [ {tour_id, name, sport, type, distance_km, ...} ],
            }
        """
        # ── Determine search mode ──────────────────────────────────────
        is_radius = (
            center_lat is not None
            and center_lng is not None
            and radius_km is not None
        )
        is_corridor = (
            start_lat is not None
            and start_lng is not None
            and end_lat is not None
            and end_lng is not None
            and corridor_km is not None
        )
        is_start_end = (
            start_lat is not None
            and start_lng is not None
            and end_lat is not None
            and end_lng is not None
            and corridor_km is None
        )
        is_roundtrip = is_radius and route_type == "roundtrip"

        if not is_radius and not is_start_end and not is_corridor:
            return {
                "status": "error",
                "message": (
                    "Provide either (center_lat + center_lng + radius_km) "
                    "for radius/roundtrip search, or "
                    "(start_lat + start_lng + end_lat + end_lng) "
                    "for start-to-end / corridor search."
                ),
            }

        capped_limit = min(limit, 50)

        # ── Mode A: Radius search ──────────────────────────────────────
        if is_radius and not is_roundtrip:
            # Guaranteed non-None by is_radius check
            clat: float = center_lat  # type: ignore[assignment]
            clng: float = center_lng  # type: ignore[assignment]
            rkm: float = radius_km   # type: ignore[assignment]

            api_params: dict[str, Any] = {
                "center": f"{clat},{clng}",
                "max_distance": int(rkm * 1000),
                "sport_types": sport,
                "sort_field": "proximity",
                "sort_direction": "asc",
                "page": page,
                "limit": capped_limit,
            }
            if tour_type:
                api_params["type"] = tour_type

            raw = await client.get(f"/users/{client.user_id}/tours/", **api_params)
            raw_results = _extract_tour_items(raw)

            results = []
            for item in raw_results[:capped_limit]:
                sp = item.get("start_point", {})
                if isinstance(sp, dict) and sp.get("lat") is not None:
                    try:
                        item_lat = float(sp["lat"])
                        item_lng = float(sp["lng"])
                    except (TypeError, ValueError):
                        item_lat = item_lng = 0.0
                else:
                    item_lat = item_lng = 0.0
                d_center = _haversine_km(
                    clat, clng, item_lat, item_lng,
                ) if item_lat != 0.0 else None
                results.append(_make_compact_result(
                    item, distance_from_center_km=d_center,
                ))

            if compact:
                return {
                    "status": "success",
                    "search_mode": "radius",
                    "total_found": len(results),
                    "page": page,
                    "limit": capped_limit,
                    "results": results,
                }

            return {
                "status": "success",
                "search_mode": "radius",
                "total_found": len(results),
                "page": page,
                "limit": capped_limit,
                "results": results,
                "raw_response": raw,
            }

        # ── Mode B: Roundtrip search ───────────────────────────────────
        if is_roundtrip:
            clat: float = center_lat  # type: ignore[assignment]
            clng: float = center_lng  # type: ignore[assignment]
            rkm: float = radius_km   # type: ignore[assignment]

            api_params = {
                "center": f"{clat},{clng}",
                "max_distance": int(rkm * 1000),
                "sport_types": sport,
                "sort_field": "proximity",
                "sort_direction": "asc",
                "page": page,
                "limit": capped_limit,
            }
            if tour_type:
                api_params["type"] = tour_type

            raw = await client.get(f"/users/{client.user_id}/tours/", **api_params)
            raw_results = _extract_tour_items(raw)

            # Load full tour data for candidates to get coordinates
            candidates = raw_results[:min(capped_limit, 15)]
            results = []
            for item in candidates:
                tid = item.get("id")
                if tid is None:
                    continue
                try:
                    full = await client.get(
                        f"/tours/{tid}",
                        _embedded="coordinates",
                    )
                except KomootError:
                    continue

                coords = (
                    full.get("_embedded", {})
                    .get("coordinates", {})
                    .get("items", [])
                )
                if not coords:
                    continue

                try:
                    s_lat = float(coords[0].get("lat", 0))
                    s_lng = float(coords[0].get("lng", 0))
                    e_lat = float(coords[-1].get("lat", 0))
                    e_lng = float(coords[-1].get("lng", 0))
                except (TypeError, ValueError):
                    continue

                tour_dist_km = (
                    (full.get("distance", 0) or 0) / 1000.0
                )
                is_rt, deviation = _is_roundtrip(s_lat, s_lng, e_lat, e_lng, tour_dist_km)

                d_center = _haversine_km(
                    center_lat or 0, center_lng or 0,
                    s_lat, s_lng,
                )

                result_entry = _make_compact_result(item, distance_from_center_km=d_center)
                result_entry["start"] = {"lat": s_lat, "lng": s_lng}
                result_entry["end"] = {"lat": e_lat, "lng": e_lng}
                result_entry["roundtrip_deviation_km"] = round(deviation, 4)
                result_entry["is_roundtrip"] = is_rt
                result_entry["tour_distance_km"] = round(tour_dist_km, 2)

                if route_type == "roundtrip" and not is_rt:
                    continue  # filter non-roundtrips

                results.append(result_entry)

            return {
                "status": "success",
                "search_mode": "roundtrip",
                "total_found": len(results),
                "page": page,
                "limit": capped_limit,
                "results": results,
            }

        # ── Mode C: Start-to-end search ────────────────────────────────
        if is_start_end:
            # Search around the midpoint with radius = half the direct
            # distance + a generous buffer
            direct_km = _haversine_km(
                start_lat, start_lng,
                end_lat, end_lng,
            )
            search_radius = max(direct_km * 0.75 + 3.0, 15.0)

            mid_lat = (start_lat + end_lat) / 2
            mid_lng = (start_lng + end_lng) / 2

            api_params = {
                "center": f"{mid_lat},{mid_lng}",
                "max_distance": int(search_radius * 1000),
                "sport_types": sport,
                "sort_field": "proximity",
                "sort_direction": "asc",
                "page": page,
                "limit": capped_limit,
            }
            if tour_type:
                api_params["type"] = tour_type

            raw = await client.get(f"/users/{client.user_id}/tours/", **api_params)
            raw_results = _extract_tour_items(raw)

            # Load full tours for coordinate matching
            candidates = raw_results[:min(capped_limit, 15)]
            results = []
            for item in candidates:
                tid = item.get("id")
                if tid is None:
                    continue
                try:
                    full = await client.get(
                        f"/tours/{tid}",
                        _embedded="coordinates",
                    )
                except KomootError:
                    continue

                coords = (
                    full.get("_embedded", {})
                    .get("coordinates", {})
                    .get("items", [])
                )
                if not coords:
                    continue

                try:
                    s_lat = float(coords[0].get("lat", 0))
                    s_lng = float(coords[0].get("lng", 0))
                    e_lat = float(coords[-1].get("lat", 0))
                    e_lng = float(coords[-1].get("lng", 0))
                except (TypeError, ValueError):
                    continue

                start_d = _haversine_km(start_lat, start_lng, s_lat, s_lng)
                end_d = _haversine_km(end_lat, end_lng, e_lat, e_lng)

                result_entry = _make_compact_result(
                    item,
                    start_distance_km=start_d,
                    end_distance_km=end_d,
                )
                result_entry["start"] = {"lat": s_lat, "lng": s_lng}
                result_entry["end"] = {"lat": e_lat, "lng": e_lng}

                results.append(result_entry)

            # Sort by combined start+end distance
            results.sort(key=lambda r: (
                r.get("start_distance_km", 999),
                r.get("end_distance_km", 999),
            ))

            return {
                "status": "success",
                "search_mode": "start_to_end",
                "total_found": len(results),
                "page": page,
                "limit": capped_limit,
                "results": results,
            }

        # ── Mode D: Corridor search ────────────────────────────────────
        if is_corridor:
            slat: float = start_lat  # type: ignore[assignment]
            slng: float = start_lng  # type: ignore[assignment]
            elat: float = end_lat    # type: ignore[assignment]
            elng: float = end_lng    # type: ignore[assignment]
            ckm: float = corridor_km  # type: ignore[assignment]

            centers = _generate_corridor_centers(slat, slng, elat, elng, ckm)

            all_raw: list[dict[str, Any]] = []
            for c_lat, c_lng in centers:
                api_params = {
                    "center": f"{c_lat},{c_lng}",
                    "max_distance": int(ckm * 1000),
                    "sport_types": sport,
                    "sort_field": "proximity",
                    "sort_direction": "asc",
                    "limit": capped_limit,
                }
                if tour_type:
                    api_params["type"] = tour_type

                try:
                    chunk = await client.get(
                        f"/users/{client.user_id}/tours/",
                        **api_params,
                    )
                except KomootError:
                    continue

                chunk_items = _extract_tour_items(chunk)
                all_raw.extend(chunk_items)

            # Deduplicate and limit
            deduped = _deduplicate_tours(all_raw)
            candidates = deduped[:min(capped_limit, 15)]

            results = []
            for item in candidates:
                tid = item.get("id")
                if tid is None:
                    continue
                try:
                    full = await client.get(
                        f"/tours/{tid}",
                        _embedded="coordinates",
                    )
                except KomootError:
                    continue

                coords = (
                    full.get("_embedded", {})
                    .get("coordinates", {})
                    .get("items", [])
                )
                if not coords:
                    continue

                max_cd = _max_corridor_distance_km(
                    coords,
                    slat, slng,
                    elat, elng,
                )

                result_entry = _make_compact_result(
                    item, corridor_distance_km=max_cd,
                )

                try:
                    result_entry["start"] = {
                        "lat": float(coords[0].get("lat", 0)),
                        "lng": float(coords[0].get("lng", 0)),
                    }
                    result_entry["end"] = {
                        "lat": float(coords[-1].get("lat", 0)),
                        "lng": float(coords[-1].get("lng", 0)),
                    }
                except (TypeError, ValueError):
                    pass

                # Filter: only tours within corridor_km of the line
                if max_cd <= ckm:
                    results.append(result_entry)

            results.sort(key=lambda r: r.get("corridor_distance_km", 999))

            return {
                "status": "success",
                "search_mode": "corridor",
                "total_found": len(results),
                "page": page,
                "limit": capped_limit,
                "corridor_centers": len(centers),
                "results": results,
            }