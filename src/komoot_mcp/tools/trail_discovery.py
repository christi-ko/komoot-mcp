"""MCP tool for discovering trail segments from existing Komoot tours.

Pure functions:
  extract_trail_segments     — extract individual trail segments with coordinates
  cluster_trail_segments     — group spatially close segments (single or multi-tour)
  discover_trail_hotspots    — find areas where multiple tours' trail segments overlap

MCP tool:
  discover_trail_segments    — orchestrate: fetch tour(s) + extract + cluster + hotspots
"""

from __future__ import annotations

import math
from typing import Any

from ..client import KomootClient, KomootError


# ── Module-level helpers (no side effects) ─────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _segment_km(coords: list[dict[str, Any]], from_i: int, to_i: int) -> float:
    total = 0.0
    for i in range(max(0, from_i), min(to_i, len(coords) - 1)):
        c1 = coords[i]
        c2 = coords[i + 1]
        try:
            total += _haversine_km(
                float(c1.get("lat", 0)), float(c1.get("lng", 0)),
                float(c2.get("lat", 0)), float(c2.get("lng", 0)),
            )
        except (TypeError, ValueError):
            pass
    return total


def _extract_way_type_items(tour: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract way_type items from an API tour response."""
    embedded = tour.get("_embedded", {})
    if not isinstance(embedded, dict):
        return []
    container = embedded.get("way_types", {})
    if isinstance(container, dict):
        items = container.get("items", [])
    elif isinstance(container, list):
        items = container
    else:
        items = []
    return items if isinstance(items, list) else []


def _extract_coord_items(tour: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract coordinate items from an API tour response."""
    embedded = tour.get("_embedded", {})
    if not isinstance(embedded, dict):
        return []
    container = embedded.get("coordinates", {})
    if isinstance(container, dict):
        items = container.get("items", [])
    elif isinstance(container, list):
        items = container
    else:
        items = []
    return items if isinstance(items, list) else []


# ── Core feature functions ─────────────────────────────────────────────

def _trail_category(element: str) -> str | None:
    """Classify a way-type element into trail category.

    Returns:
        'classified' for trail_d1..trail_d5,
        'unclassified' for wt#trail without numeric suffix,
        None for non-trail elements.
    """
    if not isinstance(element, str):
        return None
    if element.startswith("wt#trail_d"):
        return "classified"
    if element == "wt#trail":
        return "unclassified"
    return None


def _classified_sort_key(seg: dict[str, Any]) -> tuple:
    """Sort key: classified trails first, then by length descending."""
    cat = seg.get("trail_category", "unclassified")
    cat_order = 0 if cat == "classified" else 1
    length = -seg.get("length_km", 0)
    return (cat_order, length)


def extract_trail_segments(
    wt_items: list[dict[str, Any]],
    coord_items: list[dict[str, Any]],
    tour_id: int,
) -> list[dict[str, Any]]:
    """Extract individual trail segments from way-type items and coordinates.

    Detects both classified singletrail (trail_d1..trail_d5) and unclassified
    trail (wt#trail without numeric suffix). Each segment includes a
    `trail_category` field: 'classified' or 'unclassified'.

    Args:
        wt_items: way_type items from Komoot API _embedded.way_types.items.
        coord_items: coordinate items from _embedded.coordinates.items.
        tour_id: source tour ID for provenance.

    Returns:
        List of segment dicts, each with:
          start, end (lat/lng), length_km, way_type, trail_category,
          from_index, to_index, source_tour_id.
        Empty list when no trail data exists.
    """
    if not wt_items or not coord_items:
        return []

    segments: list[dict[str, Any]] = []

    for item in wt_items:
        if not isinstance(item, dict):
            continue
        element = item.get("element", "")
        category = _trail_category(element)
        if category is None:
            continue

        way_type = element.replace("wt#", "")  # trail_d1..trail_d5 or just "trail"
        if category == "unclassified":
            way_type = "trail_unclassified"  # distinct name for non-d trail

        fi = int(item.get("from", 0) or 0)
        ti = int(item.get("to", 0) or 0)

        if fi < 0 or ti > len(coord_items) or fi >= ti:
            continue

        start = coord_items[fi]
        end = coord_items[ti - 1]  # the segment ends at the coord BEFORE the 'to' index shift

        length_km = round(_segment_km(coord_items, fi, ti), 4)

        segments.append({
            "start": {
                "lat": float(start.get("lat", 0)),
                "lng": float(start.get("lng", 0)),
            },
            "end": {
                "lat": float(end.get("lat", 0)),
                "lng": float(end.get("lng", 0)),
            },
            "length_km": length_km,
            "way_type": way_type,
            "trail_category": category,
            "from_index": fi,
            "to_index": ti,
            "source_tour_id": tour_id,
        })

    return segments


def cluster_trail_segments(
    segments: list[dict[str, Any]],
    max_gap_km: float = 0.2,
) -> list[dict[str, Any]]:
    """Group trail segments that are spatially close.

    Single-tour mode: merges consecutive trail segments where the gap between
    the end of one and start of the next is ≤ max_gap_km, forming longer
    continuous trail areas.

    Multi-tour mode: also groups segments from different tours that are
    spatially close (end-to-start within max_gap_km), regardless of tour.

    Args:
        segments: list of segment dicts from extract_trail_segments().
        max_gap_km: max gap (km) to consider two segments as part of the
                    same cluster.

    Returns:
        List of cluster dicts:
          { segments: [...], total_length_km: float,
            source_tour_ids: list[int], way_types: list[str],
            trail_categories: list[str],
            start: {lat, lng}, end: {lat, lng} }
    """
    if not segments:
        return []

    # Sort by approximate position: use from_index as linear proxy
    sorted_segs = sorted(segments, key=lambda s: (s.get("source_tour_id", 0), s.get("from_index", 0)))

    clusters: list[list[dict[str, Any]]] = [[sorted_segs[0]]]

    for seg in sorted_segs[1:]:
        last = clusters[-1][-1]
        # Distance from last segment's end to this segment's start
        gap = _haversine_km(
            last["end"]["lat"], last["end"]["lng"],
            seg["start"]["lat"], seg["start"]["lng"],
        )
        if gap <= max_gap_km:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    result: list[dict[str, Any]] = []
    for group in clusters:
        total_len = round(sum(s.get("length_km", 0) for s in group), 4)
        source_ids = sorted(set(s["source_tour_id"] for s in group))
        way_types = list(dict.fromkeys(s["way_type"] for s in group))  # ordered unique
        trail_cats = list(dict.fromkeys(
            s["trail_category"] for s in group
            if s.get("trail_category")
        ))  # ordered unique categories present
        result.append({
            "segments": len(group),
            "total_length_km": total_len,
            "source_tour_ids": source_ids,
            "way_types": way_types,
            "trail_categories": trail_cats,
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })

    return result


def discover_trail_hotspots(
    segments: list[dict[str, Any]],
    min_overlap: int = 2,
    radius_km: float = 0.15,
) -> list[dict[str, Any]]:
    """Find areas where trail segments from multiple tours converge.

    A hotspot is a location where at least *min_overlap* trail segments
    (from different source tours) have their start or end points within
    *radius_km* of each other.

    Args:
        segments: list of segment dicts from extract_trail_segments().
        min_overlap: minimum number of *source tours* with segments near
                     a location to qualify as a hotspot.
        radius_km: search radius for considering segments as co-located.

    Returns:
        List of hotspot dicts:
          { center: {lat, lng}, overlapping_segments: int,
            source_tour_ids: list[int], way_types: list[str],
            approx_length_km: float }
    """
    if not segments or len(segments) < min_overlap:
        return []

    # Collect all segment endpoints as candidate centers
    points: list[dict[str, Any]] = []
    for seg in segments:
        points.append({"lat": seg["start"]["lat"], "lng": seg["start"]["lng"],
                        "seg": seg, "type": "start"})
        points.append({"lat": seg["end"]["lat"], "lng": seg["end"]["lng"],
                        "seg": seg, "type": "end"})

    hotspots: list[dict[str, Any]] = []

    for pt in points:
        # Count distinct source tours with any endpoint within radius_km
        nearby_tours: set[int] = set()
        nearby_segs: list[int] = []
        for seg in segments:
            d_start = _haversine_km(pt["lat"], pt["lng"], seg["start"]["lat"], seg["start"]["lng"])
            d_end = _haversine_km(pt["lat"], pt["lng"], seg["end"]["lat"], seg["end"]["lng"])
            if d_start <= radius_km or d_end <= radius_km:
                nearby_tours.add(seg["source_tour_id"])
                nearby_segs.append(seg["source_tour_id"])

        if len(nearby_tours) >= min_overlap:
            way_types = list(dict.fromkeys(
                s["way_type"] for s in segments
                if _haversine_km(pt["lat"], pt["lng"],
                                  s["start"]["lat"], s["start"]["lng"]) <= radius_km
                or _haversine_km(pt["lat"], pt["lng"],
                                  s["end"]["lat"], s["end"]["lng"]) <= radius_km
            ))
            hotspots.append({
                "center": {"lat": round(pt["lat"], 6), "lng": round(pt["lng"], 6)},
                "overlapping_segments": len(nearby_segs),
                "source_tour_ids": sorted(nearby_tours),
                "way_types": way_types,
            })

    # Deduplicate by rounding center to 4 decimal places (~11m resolution)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for h in hotspots:
        key = f"{round(h['center']['lat'], 4)},{round(h['center']['lng'], 4)}"
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    return deduped


# ── Routing waypoint derivation (bridge to plan_route) ────────────────

def derive_routing_waypoints(
    clusters: list[dict[str, Any]],
    hotspots: list[dict[str, Any]] | None = None,
    max_points: int = 8,
) -> list[list[float]]:
    """Derive strategic routing waypoints from trail clusters and hotspots.

    Converts cluster/hotspot data into [lat, lng] points suitable for
    plan_route(coordinates=[...], sport="mtb").

    Strategy:
      1. Represent each trail cluster by its actual entry (start) and
         exit (end) coordinates rather than a synthetic centroid.
      2. Sort clusters by total_length_km descending.
      3. For top-priority clusters allocate two waypoints (entry + exit).
         Lower-priority clusters get a single centroid point when budget
         no longer supports full pairs.
      4. Apply spatial diversity to prevent redundant waypoints.
      5. Add hotspot centers that represent genuinely new trail areas.
      6. Never pad: return fewer than max_points if not enough candidates.

    Args:
        clusters: output from cluster_trail_segments().
        hotspots: output from discover_trail_hotspots() (optional).
        max_points: max routing waypoints to return (default 8).
                    Must be > 0; returns [] when <= 0.

    Returns:
        List of [lat, lng] waypoints, ready for plan_route(coordinates=...).
        Empty list when no clusters exist or max_points <= 0.
    """
    if not clusters or max_points <= 0:
        return []

    # ── Build candidate data for each cluster ────────────────────────
    cluster_data: list[dict[str, Any]] = []
    for cl in clusters:
        start = cl.get("start", {})
        end = cl.get("end", {})
        if not start or not end:
            continue
        entry = [round(float(start["lat"]), 6), round(float(start["lng"]), 6)]
        exit_pt = [round(float(end["lat"]), 6), round(float(end["lng"]), 6)]
        centroid = [
            round((start["lat"] + end["lat"]) / 2, 6),
            round((start["lng"] + end["lng"]) / 2, 6),
        ]
        cluster_data.append({
            "entry": entry,
            "exit": exit_pt,
            "centroid": centroid,
            "length_km": cl.get("total_length_km", 0),
            "trail_category_priority": 0 if "classified" in cl.get("trail_categories", []) else 1,
        })

    if not cluster_data:
        return []

    # ── Sort by trail category first, then length ─────────────────────
    def _cluster_priority(cd: dict[str, Any]) -> tuple:
        """Sort key: classified first, then unclassified, then length."""
        cat = cd.get("trail_category_priority", 1)
        return (cat, -cd["length_km"])

    cluster_data.sort(key=_cluster_priority)

    # ── Compute spatial diversity threshold ──────────────────────────
    # Based on spread of centroids (minimum 100m, max 2km)
    all_centroids = [c["centroid"] for c in cluster_data]
    max_spread = 0.0
    for i in range(len(all_centroids)):
        for j in range(i + 1, len(all_centroids)):
            d = _haversine_km(
                all_centroids[i][0], all_centroids[i][1],
                all_centroids[j][0], all_centroids[j][1],
            )
            if d > max_spread:
                max_spread = d
    min_diversity_km = max(0.1, min(max_spread * 0.2, 2.0)) if max_spread > 0 else 0.1

    waypoints: list[list[float]] = []
    remaining = max_points

    # ── Phase 1: allocate entry+exit pairs to top clusters ──────────
    for cd in cluster_data:
        if remaining <= 0:
            break

        entry, exit_pt = cd["entry"], cd["exit"]

        if remaining >= 2:
            # Check entry diversity against *other* clusters' waypoints
            entry_diverse = not any(
                _haversine_km(w[0], w[1], entry[0], entry[1]) < min_diversity_km
                for w in waypoints
            )

            if entry_diverse:
                # Add entry first, then check exit against waypoints
                # EXCLUDING the just-added entry (intra-pair distance is intentional)
                waypoints.append(entry)
                exit_diverse = not any(
                    _haversine_km(w[0], w[1], exit_pt[0], exit_pt[1]) < min_diversity_km
                    for w in waypoints[:-1]
                )
                if exit_diverse:
                    waypoints.append(exit_pt)
                    remaining -= 2
                    continue
                else:
                    # Exit too close to other clusters' points → remove entry, fallback
                    waypoints.pop()

        # ── Fallback: single centroid point ──────────────────────────
        centroid = cd["centroid"]
        centroid_diverse = not any(
            _haversine_km(w[0], w[1], centroid[0], centroid[1]) < min_diversity_km
            for w in waypoints
        )
        if centroid_diverse:
            waypoints.append(centroid)
            remaining -= 1

    # ── Phase 2: add hotspot centers for genuinely new areas ────────
    if hotspots and remaining > 0:
        for h in hotspots:
            if remaining <= 0:
                break
            center = h.get("center", {})
            if not center:
                continue
            cpt = [center.get("lat"), center.get("lng")]
            if None in cpt:
                continue
            cpt_rounded = [round(cpt[0], 6), round(cpt[1], 6)]
            too_close = any(
                _haversine_km(w[0], w[1], cpt_rounded[0], cpt_rounded[1])
                < min_diversity_km
                for w in waypoints
            )
            if not too_close:
                waypoints.append(cpt_rounded)
                remaining -= 1

    return waypoints


# ── MCP tool registration ──────────────────────────────────────────────

def register(mcp, client: KomootClient) -> None:
    @mcp.tool()
    async def discover_trail_segments(
        tour_ids: list[int],
        max_gap_km: float = 0.2,
        min_hotspot_overlap: int = 2,
        hotspot_radius_km: float = 0.15,
        compact: bool = False,
    ) -> dict[str, Any]:
        """Discover MTB trail segments from existing Komoot tours.

        Analyses one or more existing Komoot tours, extracts all trail_d1..trail_d5
        segments, clusters nearby segments, and identifies trail hotspots where
        segments from multiple tours converge.

        This tool does NOT modify any tours or create new routes. Use the results
        as input for plan_route() when building a new route.

        Args:
            tour_ids: list of Komoot tour IDs to analyse.
            max_gap_km: max gap (km) to merge consecutive trail segments into
                        a single cluster (default: 0.2).
            min_hotspot_overlap: minimum distinct source tours whose segments
                                 must overlap to qualify as a hotspot (default: 2).
            hotspot_radius_km: radius (km) for considering segments as co-located
                               when detecting hotspots (default: 0.15).
            compact: if True, return clusters only (no per-segment details).
                     Default: False (returns full segment list + clusters + hotspots).

        Integration with plan_route():
            After discovering trail segments and clusters, use
            derive_routing_waypoints() to generate strategic [lat, lng] points
            (default: up to 8). Pass these as coordinates to plan_route(sport="mtb").

            Example:
                result = await discover_trail_segments(tour_ids=[...], compact=True)
                waypoints = derive_routing_waypoints(
                    result["clusters"], result.get("hotspots", []), max_points=8
                )
                route = await plan_route(
                    coordinates=waypoints, sport="mtb", compact=True
                )

        Returns:
            { trail_segments: [...], clusters: [...], hotspots: [...],
              source_tours: [source_tour_id, ...], status: "success" }
            On error per tour: { status: "error", errors: [...] }
        """
        all_segments: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        source_tours: list[int] = []

        for tid in tour_ids:
            try:
                tour = await client.get(
                    f"/tours/{tid}",
                    _embedded="way_types,coordinates",
                )
            except KomootError as e:
                errors.append({"tour_id": tid, "error": str(e)})
                continue

            if not isinstance(tour, dict):
                errors.append({"tour_id": tid, "error": "Unexpected API response"})
                continue

            wt_items = _extract_way_type_items(tour)
            coord_items = _extract_coord_items(tour)

            segs = extract_trail_segments(wt_items, coord_items, tid)
            all_segments.extend(segs)
            source_tours.append(tid)

        if not all_segments and not errors:
            return {
                "status": "success",
                "source_tours": source_tours,
                "trail_segments": [],
                "clusters": [],
                "hotspots": [],
                "message": "No trail segments found in the specified tours.",
            }

        clusters = cluster_trail_segments(all_segments, max_gap_km)
        hotspots = discover_trail_hotspots(all_segments, min_hotspot_overlap, hotspot_radius_km)

        result: dict[str, Any] = {
            "status": "success",
            "source_tours": source_tours,
        }

        if compact:
            result["clusters"] = clusters
            result["hotspots"] = hotspots
            result["total_segments"] = len(all_segments)
        else:
            result["trail_segments"] = all_segments
            result["clusters"] = clusters
            result["hotspots"] = hotspots

        if errors:
            result["errors"] = errors
            result["status"] = "partial" if all_segments else "error"

        return result