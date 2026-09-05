"""MCP tools for Komoot route planning and import.

Ces endpoints passent par www.komoot.com/api (pas api.komoot.de).
Inspires du projet export-komoot (Go) de pieterclaerhout.
"""

from __future__ import annotations

import asyncio
from itertools import combinations
import math
import time
from typing import Any, Optional
from uuid import uuid4

from ..client import KomootClient, KomootError

# ── Internal route cache for compact=False → create_planned_tour flow ──
# Maps route_ref -> (timestamp, full_route_data)
_route_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_route_cache_lock = asyncio.Lock()
_CACHE_TTL = 3600  # 1 hour

# Candidate geometry validity thresholds. 96% is redundant; 90-96% is
# retained as a warning so the controller can still compare a viable route.
ROUTE_REDUNDANCY_OVERLAP_PCT = 96.0
ROUTE_SIMILARITY_WARNING_PCT = 90.0
MAX_ADAPTIVE_ROUTING_CALLS = 5


ROUTE_SAMPLE_STEP_KM = 0.10
ROUTE_SIMILARITY_RADIUS_KM = 0.025


def _route_geometry_fingerprint(coord_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compactly represent a route by distance-normalized geometry samples."""
    points: list[tuple[float, float]] = []
    for item in coord_items:
        try:
            lat, lng = item.get("lat"), item.get("lng")
            if lat is not None and lng is not None:
                points.append((float(lat), float(lng)))
        except (TypeError, ValueError):
            continue
    if len(points) < 2:
        return {"hash": "", "samples": []}
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + _haversine_km(a[0], a[1], b[0], b[1]))
    total = lengths[-1]
    targets = [i * ROUTE_SAMPLE_STEP_KM for i in range(int(total / ROUTE_SAMPLE_STEP_KM) + 1)]
    if not targets or targets[-1] < total:
        targets.append(total)
    samples: list[tuple[float, float]] = []
    for target in targets:
        i = next((j for j in range(1, len(lengths)) if lengths[j] >= target), len(lengths) - 1)
        span = lengths[i] - lengths[i - 1]
        ratio = (target - lengths[i - 1]) / span if span else 0.0
        lat = points[i - 1][0] + ratio * (points[i][0] - points[i - 1][0])
        lng = points[i - 1][1] + ratio * (points[i][1] - points[i - 1][1])
        samples.append((round(lat, 6), round(lng, 6)))
    import hashlib
    payload = ";".join(f"{lat:.6f},{lng:.6f}" for lat, lng in samples)
    return {"hash": hashlib.sha256(payload.encode()).hexdigest(), "samples": samples, "length_km": round(total, 4)}


def _geometry_similarity_pct(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Symmetric distance-normalized similarity, independent of direction."""
    ap, bp = a.get("samples", []), b.get("samples", [])
    if not ap or not bp:
        return 0.0
    def coverage(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> float:
        return sum(any(_haversine_km(x, y, u, v) <= ROUTE_SIMILARITY_RADIUS_KM for u, v in target) for x, y in source) / len(source)
    return round((coverage(ap, bp) + coverage(bp, ap)) * 50.0, 1)


# ── Route overlap detection ──────────────────────────────────────────


def _compute_route_overlap(
    coord_items: list[dict[str, Any]],
    proximity_km: float = 0.025,
    min_index_gap: int = 40,
) -> dict[str, Any]:
    """Detect and measure route sections that traverse the same ground twice.

    For each coordinate, checks whether any other coordinate far along the
    route (index gap ≥ min_index_gap) is within *proximity_km* distance.
    When such a pair is found, both coordinates are flagged as overlapping.
    The total overlapping distance is the sum of Haversine steps between
    consecutive flagged coordinates.

    Small crossing overlaps at junctions are automatically filtered out
    because the *min_index_gap* prevents adjacent coordinates from being
    compared. Parallel but distinct paths that are farther apart than
    *proximity_km* do not trigger.

    Args:
        coord_items: coordinate list from _embedded.coordinates.items,
                     each a dict with 'lat' and 'lng'.
        proximity_km: max distance (km) to consider two far-apart
                      coordinates as overlapping (default 0.025 ≈ 25 m).
        min_index_gap: minimum index separation along the route for a
                       coordinate pair to be considered (default 40).
                       Prevents short crossings from registering.

    Returns:
        Dict with:
          overlap_km (float): estimated total overlapping track length.
          overlap_percentage (float): overlap_km relative to total_km.
    """
    n = len(coord_items)
    if n < min_index_gap * 2:
        return {"overlap_km": 0.0, "overlap_percentage": 0.0}

    # Flagged indices
    flagged: set[int] = set()

    # For each coordinate, check later coords at min_index_gap distance
    for i in range(n - min_index_gap):
        if i in flagged:
            continue
        lat_i = float(coord_items[i].get("lat", 0))
        lng_i = float(coord_items[i].get("lng", 0))

        for j in range(i + min_index_gap, n, 2):  # sample every 2nd
            d = _haversine_km(
                lat_i, lng_i,
                float(coord_items[j].get("lat", 0)),
                float(coord_items[j].get("lng", 0)),
            )
            if d < proximity_km:
                flagged.add(i)
                flagged.add(j)

    if not flagged:
        return {"overlap_km": 0.0, "overlap_percentage": 0.0}

    # Walk through all coords; sum distances between consecutive coords
    # where both endpoints are flagged as overlapping.
    overlap_km = 0.0
    for i in range(n - 1):
        if i in flagged and (i + 1) in flagged:
            overlap_km += _haversine_km(
                float(coord_items[i].get("lat", 0)),
                float(coord_items[i].get("lng", 0)),
                float(coord_items[i + 1].get("lat", 0)),
                float(coord_items[i + 1].get("lng", 0)),
            )

    total_km = _segment_km(coord_items, 0, n)
    pct = round(overlap_km / total_km * 100, 1) if total_km > 0 else 0.0
    return {
        "overlap_km": round(overlap_km, 4),
        "overlap_percentage": pct,
    }


# ── Trail coverage (against discovered segments) ─────────────────────


def _compute_trail_coverage(
    route_coord_items: list[dict[str, Any]],
    discovered_segments: list[dict[str, Any]],
    proximity_km: float = 0.050,
) -> dict[str, Any]:
    """Measure what portion of discovered trail segments the route actually traverses.

    For each discovered segment, samples points every ~5 m along its line
    and checks whether the route passes within *proximity_km* of any sample
    point.  covered_km = sum of covered sample intervals, deduplicated
    across overlapping segments via a global position grid.

    A single crossing of a long segment registers only the few samples
    near the crossing point, not the full segment length.

    Args:
        route_coord_items: route coordinates from _embedded.coordinates.items.
        discovered_segments: output from extract_trail_segments().  Each must
                             have 'start', 'end', 'length_km', 'trail_category'.
        proximity_km: max distance (km) to consider a route coordinate near
                      a trail sample point (default 0.050 ≈ 50 m).

    Returns:
        Dict:
          classified: {discovered, covered, total_km, covered_km,
                       coverage_percentage}
          unclassified: {discovered, covered, total_km, covered_km,
                         coverage_percentage}
          total_coverage_percentage: float
    """
    SAMPLE_STEP_KM = 0.005  # ~5 m between sample points

    if not discovered_segments:
        return {
            "classified": {"discovered": 0, "covered": 0, "total_km": 0.0,
                           "covered_km": 0.0, "coverage_percentage": 0.0},
            "unclassified": {"discovered": 0, "covered": 0, "total_km": 0.0,
                             "covered_km": 0.0, "coverage_percentage": 0.0},
            "total_coverage_percentage": 0.0,
        }

    classified_segs: list[dict[str, Any]] = []
    unclassified_segs: list[dict[str, Any]] = []
    for seg in discovered_segments:
        cat = seg.get("trail_category", "")
        if cat == "classified":
            classified_segs.append(seg)
        elif cat == "unclassified":
            unclassified_segs.append(seg)

    # Global dedup grid: (lat_rounded_1e5, lng_rounded_1e5) — ~1.1 m resolution
    global_seen: set[tuple[int, int]] = set()

    def _compute_category(
        segs: list[dict[str, Any]],
    ) -> tuple[int, int, float, float]:
        """Return (discovered, covered_count, total_km, covered_km)."""
        if not segs:
            return 0, 0, 0.0, 0.0

        total_km = 0.0
        covered_count = 0
        cat_covered_km = 0.0

        for seg in segs:
            start = seg.get("start", {})
            end = seg.get("end", {})
            s_lat = float(start.get("lat", 0.0))
            s_lng = float(start.get("lng", 0.0))
            e_lat = float(end.get("lat", 0.0))
            e_lng = float(end.get("lng", 0.0))
            seg_len = max(seg.get("length_km", 0.0), 1e-9)

            total_km += seg_len

            # Determine sample count (at least 2 points = endpoints)
            num_samples = max(2, int(seg_len / SAMPLE_STEP_KM) + 1)
            sample_interval = seg_len / (num_samples - 1)

            seg_covered_count = 0

            for s in range(num_samples):
                t = s / (num_samples - 1) if num_samples > 1 else 0.5
                lat = s_lat + (e_lat - s_lat) * t
                lng = s_lng + (e_lng - s_lng) * t

                # Dedup key: ~1.1 m grid
                key = (round(lat * 100_000), round(lng * 100_000))
                if key in global_seen:
                    continue  # already counted from another segment

                # Check if any route coord is near this sample point
                near = False
                for rc in route_coord_items:
                    d = _haversine_km(
                        lat, lng,
                        float(rc.get("lat", 0)),
                        float(rc.get("lng", 0)),
                    )
                    if d < proximity_km:
                        near = True
                        break

                if near:
                    global_seen.add(key)
                    seg_covered_count += 1

            if seg_covered_count > 0:
                covered_count += 1
            cat_covered_km += seg_covered_count * sample_interval

        return len(segs), covered_count, total_km, cat_covered_km

    def _pct(part: float, total: float) -> float:
        return round(part / total * 100, 1) if total > 0 else 0.0

    c_disc, c_covered, c_total, c_km = _compute_category(classified_segs)
    u_disc, u_covered, u_total, u_km = _compute_category(unclassified_segs)

    # Clamp covered_km to total_km per category (rounding guard)
    c_km = min(c_km, c_total)
    u_km = min(u_km, u_total)

    classified_result = {
        "discovered": c_disc,
        "covered": c_covered,
        "total_km": round(c_total, 4),
        "covered_km": round(c_km, 4),
        "coverage_percentage": _pct(c_km, c_total),
    }
    unclassified_result = {
        "discovered": u_disc,
        "covered": u_covered,
        "total_km": round(u_total, 4),
        "covered_km": round(u_km, 4),
        "coverage_percentage": _pct(u_km, u_total),
    }

    total_covered_km = c_km + u_km
    total_all_km = c_total + u_total
    return {
        "classified": classified_result,
        "unclassified": unclassified_result,
        "total_coverage_percentage": _pct(total_covered_km, total_all_km),
    }


# ── Per-cluster coverage analysis ─────────────────────────────


def _sample_segment_along(
    seg: dict[str, Any],
    route_coord_items: list[dict[str, Any]],
    proximity_km: float = 0.050,
    sample_step_km: float = 0.005,
    global_seen: set[tuple[int, int]] | None = None,
) -> tuple[int, float]:
    """Sample a single segment and return (covered_sample_count, covered_km).

    Uses the same sampling approach as _compute_trail_coverage (5 m step,
    Haversine, dedup grid at ~1.1 m resolution). When global_seen is
    provided, deduplicates against it and updates it.
    """
    start = seg.get("start", {})
    end = seg.get("end", {})
    s_lat = float(start.get("lat", 0.0))
    s_lng = float(start.get("lng", 0.0))
    e_lat = float(end.get("lat", 0.0))
    e_lng = float(end.get("lng", 0.0))
    seg_len = max(seg.get("length_km", 0.0), 1e-9)

    num_samples = max(2, int(seg_len / sample_step_km) + 1)
    sample_interval = seg_len / (num_samples - 1) if num_samples > 1 else seg_len

    covered_count = 0
    for s in range(num_samples):
        t = s / (num_samples - 1) if num_samples > 1 else 0.5
        lat = s_lat + (e_lat - s_lat) * t
        lng = s_lng + (e_lng - s_lng) * t

        key = (round(lat * 100_000), round(lng * 100_000))
        if global_seen is not None:
            if key in global_seen:
                continue
            global_seen.add(key)

        near = False
        for rc in route_coord_items:
            d = _haversine_km(
                lat, lng,
                float(rc.get("lat", 0)), float(rc.get("lng", 0)),
            )
            if d < proximity_km:
                near = True
                break

        if near:
            covered_count += 1

    return covered_count, covered_count * sample_interval


def analyze_cluster_coverage(
    clusters: list[dict[str, Any]],
    route_coord_items: list[dict[str, Any]],
    proximity_km: float = 0.050,
) -> list[dict[str, Any]]:
    """Analyze coverage of each trail cluster against a planned route.

    For each cluster, samples points along the cluster line (start->end)
    at 5 m steps and counts how many sample points are within proximity_km
    of the route. A single crossing registers only a few samples (~3%),
    not the full cluster length.

    Args:
        clusters: output from cluster_trail_segments().
        route_coord_items: route coordinates as list of {'lat': ..., 'lng': ...}.
        proximity_km: max distance (km) for a sample to count as covered.

    Returns:
        List of dicts, one per cluster, each with:
          cluster_index, total_length_km, covered_km, coverage_percentage,
          trail_categories, way_types, covered, partial.
    """
    if not clusters or not route_coord_items:
        return []

    results: list[dict[str, Any]] = []
    global_seen: set[tuple[int, int]] = set()

    for idx, cl in enumerate(clusters):
        seg = {
            "start": cl.get("start", {}),
            "end": cl.get("end", {}),
            "length_km": cl.get("total_length_km", 0),
        }
        covered_count, raw_covered_km = _sample_segment_along(
            seg, route_coord_items, proximity_km, global_seen=global_seen,
        )

        total_len = max(float(seg["length_km"]), 1e-9)
        num_samples = max(2, int(total_len / 0.005) + 1)
        sample_interval = total_len / (num_samples - 1) if num_samples > 1 else total_len
        covered_km = min(covered_count * sample_interval, total_len)

        pct = round(covered_km / total_len * 100, 1) if total_len > 0 else 0.0
        cats = cl.get("trail_categories", [])
        covered = pct >= 70.0
        partial = 10.0 <= pct < 70.0

        results.append({
            "cluster_index": idx,
            "total_length_km": round(total_len, 4),
            "covered_km": round(covered_km, 4),
            "coverage_percentage": pct,
            "trail_categories": cats,
            "way_types": cl.get("way_types", []),
            "covered": covered,
            "partial": partial,
        })

    return results


def find_uncovered_clusters(
    analyzed: list[dict[str, Any]],
    covered_threshold: float = 70.0,
    uncovered_threshold: float = 10.0,
) -> dict[str, Any]:
    """Categorize clusters into covered, partial, and uncovered.

    Args:
        analyzed: output from analyze_cluster_coverage().
        covered_threshold: coverage % above which a cluster is 'covered'.
        uncovered_threshold: coverage % below which a cluster is 'uncovered'.

    Returns:
        { covered: [...], partial: [...], uncovered: [...] }
        Each entry is the enriched cluster dict from analyzed.
    """
    covered: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []

    for cl in analyzed:
        pct = cl.get("coverage_percentage", 0.0)
        if pct >= covered_threshold:
            covered.append(cl)
        elif pct >= uncovered_threshold:
            partial.append(cl)
        else:
            uncovered.append(cl)

    return {
        "covered": covered,
        "partial": partial,
        "uncovered": uncovered,
    }


def suggest_uncovered_waypoints(
    clusters: list[dict[str, Any]],
    analyzed: list[dict[str, Any]],
    current_waypoints: list[list[float]] | None = None,
    max_new: int = 2,
) -> list[list[float]]:
    """Suggest entry/exit waypoints for clusters with low coverage.

    Selects from uncovered (and if budget remains, partial) clusters,
    preferring classified over unclassified, larger over smaller, and
    spatially diverse points.

    Args:
        clusters: original cluster list from cluster_trail_segments().
        analyzed: output from analyze_cluster_coverage().
        current_waypoints: existing waypoints (for diversity check).
        max_new: max suggested points (default 2).

    Returns:
        List of [lat, lng] waypoints for the next routing attempt.
    """
    if not clusters or not analyzed:
        return []

    # Index analyzed by cluster_index
    by_idx: dict[int, dict[str, Any]] = {}
    for c in analyzed:
        by_idx[c["cluster_index"]] = c

    # Gather uncovered + partial clusters, sorted: worst coverage first
    candidates = []
    for c in analyzed:
        pct = c.get("coverage_percentage", 100.0)
        if pct < 70.0:  # uncovered or partial
            candidates.append((c["cluster_index"], pct))

    # Worst coverage first
    candidates.sort(key=lambda x: x[1])

    # Build existing set for diversity
    existing: set[tuple[float, float]] = set()
    if current_waypoints:
        for wp in current_waypoints:
            existing.add((round(wp[0], 4), round(wp[1], 4)))

    suggestions: list[tuple[list[float], int, float, bool]] = []

    for idx, _ in candidates:
        if len(suggestions) >= max_new:
            break
        if idx >= len(clusters):
            continue

        cl = clusters[idx]
        start = cl.get("start", {})
        end = cl.get("end", {})
        is_classified = "classified" in by_idx.get(idx, {}).get("trail_categories", [])
        total_len = cl.get("total_length_km", 0)

        entry = [round(float(start.get("lat", 0)), 6), round(float(start.get("lng", 0)), 6)]
        exit_pt = [round(float(end.get("lat", 0)), 6), round(float(end.get("lng", 0)), 6)]

        entry_key = (round(entry[0], 4), round(entry[1], 4))
        exit_key = (round(exit_pt[0], 4), round(exit_pt[1], 4))

        if entry_key not in existing and len(suggestions) < max_new:
            suggestions.append((entry, 0 if is_classified else 1, -total_len, is_classified))
            existing.add(entry_key)

        if exit_key not in existing and len(suggestions) < max_new:
            suggestions.append((exit_pt, 0 if is_classified else 1, -total_len, is_classified))
            existing.add(exit_key)

    # Sort: classified first, then larger clusters first
    suggestions.sort(key=lambda x: (x[1], x[2]))

    return [s[0] for s in suggestions]


def _suggest_uncovered_waypoints_with_audit(
    clusters: list[dict[str, Any]],
    analyzed: list[dict[str, Any]],
    current_waypoints: list[list[float]] | None = None,
    max_new: int = 2,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Return historical suggestions and an audit without changing selection."""
    suggestions = suggest_uncovered_waypoints(
        clusters, analyzed, current_waypoints=current_waypoints, max_new=max_new
    )
    candidates = sorted(
        (c for c in analyzed if c.get("coverage_percentage", 100) < 70),
        key=lambda c: c.get("coverage_percentage", 100),
    )
    audit: list[dict[str, Any]] = []
    # Consume each selected coordinate exactly once.  Entry and exit can be
    # identical (or shared by adjacent clusters); a set alone would falsely
    # mark both audit records as selected for one actual waypoint.
    selected_counts: dict[tuple[float, float], int] = {}
    for wp in suggestions:
        key = (float(wp[0]), float(wp[1]))
        selected_counts[key] = selected_counts.get(key, 0) + 1
    for priority, candidate in enumerate(candidates, start=1):
        idx = candidate.get("cluster_index")
        cl = clusters[idx] if isinstance(idx, int) and idx < len(clusters) else None
        status = ("uncovered" if candidate.get("coverage_percentage", 0) < 10
                  else "partial")
        entry = (cl or {}).get("start") or {}
        exit_ = (cl or {}).get("end") or {}
        points = [
            ("entry", [entry.get("lat"), entry.get("lng")]),
            ("exit", [exit_.get("lat"), exit_.get("lng")]),
        ]
        for kind, point in points:
            valid = all(isinstance(v, (int, float)) for v in point)
            lat, lng = point
            wp = ([round(float(lat), 6), round(float(lng), 6)]
                  if valid else None)
            key = (float(wp[0]), float(wp[1])) if wp is not None else None
            is_selected = bool(key is not None and selected_counts.get(key, 0) > 0)
            if is_selected:
                selected_counts[key] -= 1
            audit.append({
                "cluster_id": idx,
                "classification": ((candidate.get("trail_categories") or ["other"])[0]),
                "trail_km": candidate.get("total_length_km"),
                "cluster_center": candidate.get("center"),
                "entry": {"latitude": entry.get("lat"), "longitude": entry.get("lng")} if entry else None,
                "exit": {"latitude": exit_.get("lat"), "longitude": exit_.get("lng")} if exit_ else None,
                "status_before": status,
                "priority": priority,
                "selected": is_selected,
                "waypoint": ({"latitude": float(wp[0]), "longitude": float(wp[1])}
                              if is_selected and wp is not None else None),
                "waypoint_kind": kind if is_selected else None,
                "selection_source": "suggest_uncovered_waypoints" if is_selected else None,
                "reason": ("selected_by_existing_order" if is_selected
                           else "not_selected_by_existing_budget_or_order"),
            })
    return suggestions, audit


def _cluster_status_map(trail_clusters: dict[str, Any]) -> dict[Any, str]:
    return {
        c.get("cluster_index"): status
        for status in ("covered", "partial", "uncovered")
        for c in trail_clusters.get(status, [])
    }


def _cluster_transition(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before, after = _cluster_status_map(previous), _cluster_status_map(current)
    return {
        "previous": {s: sorted(i for i, v in before.items() if v == s)
                     for s in ("uncovered", "partial", "covered")},
        "current": {s: sorted(i for i, v in after.items() if v == s)
                    for s in ("uncovered", "partial", "covered")},
        "new_covered": sorted(i for i, v in after.items() if v == "covered" and before.get(i) != "covered"),
        "partial_to_covered": sorted(i for i, v in after.items() if v == "covered" and before.get(i) == "partial"),
    }


def _compact_cluster_info(
    analyzed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build compact per-cluster info without large coordinate lists.

    Args:
        analyzed: output from analyze_cluster_coverage().

    Returns:
        { covered: [...], partial: [...], uncovered: [...] }
        Each entry is a compact dict with cluster_index, total_length_km,
        covered_km, coverage_percentage, trail_categories.
    """
    categorized = find_uncovered_clusters(analyzed)

    def _strip(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "cluster_index": entry["cluster_index"],
            "total_length_km": entry["total_length_km"],
            "covered_km": entry["covered_km"],
            "coverage_percentage": entry["coverage_percentage"],
            "trail_categories": entry["trail_categories"],
        }

    return {
        "covered": [_strip(c) for c in categorized["covered"]],
        "partial": [_strip(c) for c in categorized["partial"]],
        "uncovered": [_strip(c) for c in categorized["uncovered"]],
    }


# ── Feedback loop: iterative route optimization ──────────────────────


def _build_analyzed_from_clusters(
    trail_clusters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct analyzed list from compact trail_clusters output.

    The compact trail_clusters from _compact_cluster_info() stores clusters
    partitioned into covered/partial/uncovered.  This rebuilds the flat
    analyzed list format expected by suggest_uncovered_waypoints().
    """
    analyzed: list[dict[str, Any]] = []
    for entry in trail_clusters.get("covered", []):
        d = dict(entry)
        d["covered"] = True
        d["partial"] = False
        d["way_types"] = []
        analyzed.append(d)
    for entry in trail_clusters.get("partial", []):
        d = dict(entry)
        d["covered"] = d.get("coverage_percentage", 0) >= 70
        d["partial"] = True
        d["way_types"] = []
        analyzed.append(d)
    for entry in trail_clusters.get("uncovered", []):
        d = dict(entry)
        d["covered"] = False
        d["partial"] = False
        d["way_types"] = []
        analyzed.append(d)
    return analyzed


def _is_qualitatively_better(
    a: dict[str, Any],
    b: dict[str, Any],
) -> bool:
    """Return True if route ``a`` wins the qualitative lexicographic order.

    The order is deliberately explicit: hard constraints, classified trail
    coverage, optional difficulty constraints, soft overlap, trail quality,
    then distance/elevation and asphalt.  Optional constraints are read only
    from ``selection_constraints``; absent constraints never create a hidden
    optimization target.
    """

    def _g(container: Any, key: str, default: Any = 0) -> Any:
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def _technical_rank(value: Any) -> int | None:
        if isinstance(value, str) and len(value) == 2 and value[0].upper() == "T":
            try:
                return int(value[1:])
            except ValueError:
                return None
        return None

    def _constraint_status(route: dict[str, Any]) -> tuple[int, float]:
        """Return (violations, excess) for explicitly supplied max limits."""
        constraints = route.get("selection_constraints") or {}
        if not isinstance(constraints, dict):
            return 0, 0.0
        violations = 0
        excess = 0.0

        technical_max = _technical_rank(constraints.get("technical_max"))
        technical = _technical_rank(route.get("technical_difficulty"))
        if technical_max is not None and technical is not None and technical > technical_max:
            violations += 1
            excess += technical - technical_max

        elevation_max = constraints.get("max_elevation_up_m")
        elevation = route.get("elevation_up_m")
        if isinstance(elevation_max, (int, float)) and isinstance(elevation, (int, float)):
            if elevation > elevation_max:
                violations += 1
                excess += elevation - elevation_max

        return violations, excess

    a_hard = _constraint_status(a)
    b_hard = _constraint_status(b)
    if a_hard != b_hard:
        return a_hard < b_hard

    a_cov = a.get("trail_coverage") or {}
    b_cov = b.get("trail_coverage") or {}
    a_cls = a_cov.get("classified") or {}
    b_cls = b_cov.get("classified") or {}

    # B. Actual routed classified trail is the primary quality signal.
    # Discovery coverage is a means of finding promising trail areas; it
    # must not outrank trail that is actually present in the route.
    a_st = _g(a.get("singletrail") or {}, "singletrail_total_km", 0)
    b_st = _g(b.get("singletrail") or {}, "singletrail_total_km", 0)
    a_q = _g(a.get("route_quality_metrics") or {}, "classified_trail_km", a_st)
    b_q = _g(b.get("route_quality_metrics") or {}, "classified_trail_km", b_st)
    if abs(a_q - b_q) > 0.5:
        return a_q > b_q

    # C. Discovery coverage is a tie-break between comparable routes.
    a_cc = _g(a_cls, "covered", 0)
    b_cc = _g(b_cls, "covered", 0)
    if a_cc != b_cc:
        return a_cc > b_cc
    a_cp = _g(a_cls, "coverage_percentage", 0)
    b_cp = _g(b_cls, "coverage_percentage", 0)
    if abs(a_cp - b_cp) > 5:
        return a_cp > b_cp

    # D. Fewer uncovered/partial classified clusters.
    a_tc = a.get("trail_clusters") or {}
    b_tc = b.get("trail_clusters") or {}

    def _classified_remaining(tc: dict[str, Any]) -> int:
        return sum(
            1 for bucket in ("uncovered", "partial")
            for cluster in tc.get(bucket, [])
            if "classified" in (cluster.get("trail_categories") or [])
        )

    a_remaining = _classified_remaining(a_tc)
    b_remaining = _classified_remaining(b_tc)
    if a_remaining != b_remaining:
        return a_remaining < b_remaining

    # E. No hidden T/C preference: an explicit maximum is a hard constraint
    # handled above, not a reason to prefer a harder feasible route.

    # F. Lower overlap is a soft criterion only.
    a_overlap = _g(a.get("route_overlap") or {}, "overlap_percentage", 0)
    b_overlap = _g(b.get("route_overlap") or {}, "overlap_percentage", 0)
    if abs(a_overlap - b_overlap) > 5:
        return a_overlap < b_overlap

    # G. Surface offroad is the next quality criterion. Keep it distinct
    # from classified trail: it is supportive terrain evidence, not trail.
    a_off = _g(a.get("route_quality_metrics") or {}, "surface_offroad_km", 0)
    b_off = _g(b.get("route_quality_metrics") or {}, "surface_offroad_km", 0)
    if abs(a_off - b_off) > 0.5:
        return a_off > b_off

    # H. Distance is soft: after trail quality, prefer an in-range route.
    # Distance does not participate in hard-constraint violations.
    # When explicit distance bounds exist and both routes are in-range,
    # distance is neutral (do not prefer shorter over longer).
    a_constraints = a.get("selection_constraints") or {}
    b_constraints = b.get("selection_constraints") or {}
    has_distance_bounds = bool(
        a_constraints.get("min_distance_km") is not None
        or a_constraints.get("max_distance_km") is not None
        or b_constraints.get("min_distance_km") is not None
        or b_constraints.get("max_distance_km") is not None
    )
    if has_distance_bounds:
        # Check if both are in-range wrt their own constraints
        def _in_range(route: dict[str, Any]) -> bool:
            c = route.get("selection_constraints") or {}
            d = route.get("distance_km")
            if not isinstance(d, (int, float)):
                return True
            min_d = c.get("min_distance_km")
            max_d = c.get("max_distance_km")
            if isinstance(min_d, (int, float)) and d < min_d:
                return False
            if isinstance(max_d, (int, float)) and d > max_d:
                return False
            return True

        a_in_range = _in_range(a)
        b_in_range = _in_range(b)
        if a_in_range and b_in_range:
            # Both in-range: distance is neutral, fall through to I
            pass
        else:
            # Neither route is necessarily feasible. Compare distance to
            # the nearest bound, not raw route length: for two routes below
            # min, the longer route is closer; for two above max, shorter is
            # closer. If only one bound exists, use its violation distance.
            def _distance_to_range(route: dict[str, Any]) -> float:
                c = route.get("selection_constraints") or {}
                d = route.get("distance_km")
                if not isinstance(d, (int, float)):
                    return float("inf")
                min_d = c.get("min_distance_km")
                max_d = c.get("max_distance_km")
                distances = []
                if isinstance(min_d, (int, float)) and d < min_d:
                    distances.append(min_d - d)
                if isinstance(max_d, (int, float)) and d > max_d:
                    distances.append(d - max_d)
                return min(distances, default=0.0)

            a_gap = _distance_to_range(a)
            b_gap = _distance_to_range(b)
            if a_gap != b_gap:
                return a_gap < b_gap
            # Equal violation distance: preserve the historical tie behavior.
            a_d = _g(a, "distance_km", 0)
            b_d = _g(b, "distance_km", 0)
            if abs(a_d - b_d) > 2:
                return a_d < b_d
    else:
        # No explicit distance bounds — old behavior: shorter preferred
        a_d = _g(a, "distance_km", 0)
        b_d = _g(b, "distance_km", 0)
        if abs(a_d - b_d) > 2:
            return a_d < b_d

    # I. Asphalt is last and cannot displace a materially better trail route.
    a_asp = _g(a.get("surfaces") or {}, "asphalt", 0)
    b_asp = _g(b.get("surfaces") or {}, "asphalt", 0)
    if abs(a_asp - b_asp) > 5:
        return a_asp < b_asp

    return a.get("_attempt", 999) < b.get("_attempt", 999)


def _select_best_route_index(
    iterations: list[dict[str, Any]],
) -> int:
    """Select the best route iteration by qualitative comparison.

    Uses _is_qualitatively_better for pairwise comparison.
    Returns the index into iterations of the best route.
    """
    if not iterations:
        return -1
    best = 0
    for i in range(1, len(iterations)):
        if _is_qualitatively_better(iterations[i], iterations[best]):
            best = i
    return best


def _feedback_iteration_summary(
    iteration: dict[str, Any],
    attempt: int,
    waypoints: list[list[float]],
    waypoints_added: int,
) -> dict[str, Any]:
    """Compact summary of one feedback-loop iteration.

    Strips large data while keeping all evaluation-relevant fields.
    """
    summary: dict[str, Any] = {
        "attempt": attempt,
        "distance_km": iteration.get("distance_km"),
        "elevation_up_m": iteration.get("elevation_up_m"),
        "elevation_down_m": iteration.get("elevation_down_m"),
        "duration": iteration.get("duration"),
        "difficulty": iteration.get("difficulty"),
        "technical_difficulty": iteration.get("technical_difficulty"),
        "fitness_difficulty": iteration.get("fitness_difficulty"),
        "waypoints_total": len(waypoints),
        "waypoints_added": waypoints_added,
    }

    if "route_overlap" in iteration:
        summary["route_overlap"] = iteration["route_overlap"]
    if "trail_coverage" in iteration:
        summary["trail_coverage"] = iteration["trail_coverage"]
    if "trail_clusters" in iteration:
        summary["trail_clusters"] = iteration["trail_clusters"]
    if "surfaces" in iteration:
        summary["surfaces"] = iteration["surfaces"]
    if "singletrail" in iteration:
        st = iteration["singletrail"]
        st_compact: dict[str, Any] = {}
        if isinstance(st, dict) and st.get("available", True) is not False:
            for key in ("singletrail_total_km", "singletrail_percentage",
                        "unclassified_trail_total_km", "unclassified_trail_percentage",
                        "trail_d1", "trail_d2", "trail_d3", "trail_d4", "trail_d5",
                        "trail_unclassified"):
                if key in st:
                    st_compact[key] = st[key]
        if st_compact:
            summary["singletrail"] = st_compact
    if "way_types" in iteration:
        summary["way_types"] = iteration["way_types"]
    if "route_quality_metrics" in iteration:
        summary["route_quality_metrics"] = iteration["route_quality_metrics"]
    if "planning_audit" in iteration:
        summary["planning_audit"] = iteration["planning_audit"]
    for key in ("variant_id", "strategy", "selected_cluster_ids", "candidate_status",
                "route_fingerprint", "geometry_similarity", "variant_specific_waypoints"):
        if key in iteration:
            summary[key] = iteration[key]

    return summary


def _build_route_coordinates(initial_coordinates: list[list[float]], variant_specific_waypoints: list[list[float]], route_mode: str = "auto") -> list[list[float]]:
    """Build an independent route while preserving explicit round trips."""
    if not initial_coordinates:
        return [list(point) for point in variant_specific_waypoints]
    start, first, last = list(initial_coordinates[0]), list(initial_coordinates[0]), list(initial_coordinates[-1])
    is_roundtrip = route_mode == "roundtrip" or (route_mode == "auto" and len(initial_coordinates) > 1 and _haversine_km(first[0], first[1], last[0], last[1]) <= 0.5)
    if is_roundtrip:
        return [start, *[list(point) for point in initial_coordinates[1:-1]], *[list(point) for point in variant_specific_waypoints], start]
    return [*[list(point) for point in initial_coordinates[:-1]], *[list(point) for point in variant_specific_waypoints], last]


def _detect_backtracking(coordinates: list[list[float]], tolerance_km: float = 0.05) -> bool:
    """Detect a route that returns to an earlier point away from the closure."""
    route_points = coordinates[:-1] if len(coordinates) > 3 and coordinates[0] == coordinates[-1] else coordinates
    for index, point in enumerate(route_points):
        for prior in route_points[:max(0, index - 1)]:
            if _haversine_km(float(point[0]), float(point[1]), float(prior[0]), float(prior[1])) <= tolerance_km:
                return True
    return False


def _planning_audit(
    route: dict[str, Any],
    coordinates: list[list[float]],
    trail_areas: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    selected_cluster_ids: list[int],
    connection_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a conservative planning audit from route and discovery data."""
    closed = bool(coordinates and coordinates[0] == coordinates[-1])
    q = route.get("route_quality_metrics") or {}
    cov = (route.get("trail_coverage") or {}).get("classified") or {}
    direction = [_classify_trail_direction(
        (clusters[i].get("start") or {}).get("alt"),
        (clusters[i].get("end") or {}).get("alt"),
    ) for i in selected_cluster_ids if 0 <= i < len(clusters)]
    way_audit = _connection_waytype_audit(
        route.get("way_types"),
        comparison_available=bool(route.get("comparison_route_available", False)),
    )
    return {
        "route_mode": "roundtrip" if closed else "point_to_point",
        "roundtrip_closed": closed,
        "start_coordinate": coordinates[0] if coordinates else None,
        "end_coordinate": coordinates[-1] if coordinates else None,
        "trail_area_sequence": [next((a.get("area_id") for a in trail_areas if i in a.get("cluster_indices", [])), None) for i in selected_cluster_ids],
        "cluster_sequence": list(selected_cluster_ids),
        "area_sequence": [next((a.get("area_id") for a in trail_areas if i in a.get("cluster_indices", [])), None) for i in selected_cluster_ids],
        "connection_plan": connection_segments,
        "entry_exit_plan": [{"cluster_id": i, "entry_point": clusters[i].get("start"), "exit_point": clusters[i].get("end"), "preferred_direction": direction[n] if n < len(direction) else "direction_unknown", "direction_status": direction[n] if n < len(direction) else "direction_unknown", "steepness_status": clusters[i].get("steepness_status", "unknown"), "difficulty_status": clusters[i].get("difficulty_status", "unknown"), "role": "trail_area"} for n, i in enumerate(selected_cluster_ids) if 0 <= i < len(clusters)],
        "direction_status": direction if direction else ["direction_unknown"],
        "steepness_status": [clusters[i].get("steepness_status", "unknown") for i in selected_cluster_ids if 0 <= i < len(clusters)],
        "difficulty_status": [clusters[i].get("difficulty_status", "unknown") for i in selected_cluster_ids if 0 <= i < len(clusters)],
        "connection_segments": connection_segments,
        "connection_way_types": way_audit["way_types"],
        "cycleway_present": way_audit["cycleway_present"],
        "local_road_present": way_audit["local_road_present"],
        "major_road_present": way_audit["major_road_present"],
        "major_road_avoidable": way_audit["major_road_avoidable"],
        "backtracking_detected": _detect_backtracking(coordinates),
        "route_overlap": route.get("route_overlap", "unknown"),
        "classified_trail_km": q.get("classified_trail_km", "unknown"),
        "surface_offroad_km": q.get("surface_offroad_km", "unknown"),
        "discovery_trail_coverage": cov.get("coverage_percentage", "unknown"),
        "distance": route.get("distance_km", "unknown"),
        "elevation": route.get("elevation_up_m", "unknown"),
        "T/C": {"technical": route.get("technical_difficulty", "unknown"), "fitness": route.get("fitness_difficulty", "unknown")},
        "asphalt": (route.get("surfaces") or {}).get("asphalt", "unknown"),
        "geometry_similarity": route.get("geometry_similarity", "unknown"),
    }


def _connection_waytype_audit(
    way_types: dict[str, Any] | list[Any] | None,
    comparison_available: bool = True,
) -> dict[str, Any]:
    """Summarize objective route way types for inter-area connections."""
    if isinstance(way_types, dict):
        names = {str(name).lower() for name in way_types}
    else:
        names = {str(item.get("type", item) if isinstance(item, dict) else item).lower().replace("wt#", "") for item in (way_types or [])}
    cycleway = any("cycleway" in name or "bicycle" in name for name in names)
    local = any(name in names for name in {"minor_road", "residential", "living_street", "track", "path", "street"})
    major = any(name in names for name in {"highway", "primary", "secondary", "trunk", "motorway"})
    return {
        "cycleway_present": cycleway,
        "local_road_present": local,
        "major_road_present": major,
        "major_road_avoidable": (not major) if comparison_available else "unknown",
        "way_types": sorted(names),
    }


def _classify_trail_direction(start_elevation_m: float | None, end_elevation_m: float | None, flat_threshold_m: float = 5.0) -> str:
    """Classify direction without inventing one when elevations are absent."""
    if not isinstance(start_elevation_m, (int, float)) or not isinstance(end_elevation_m, (int, float)):
        return "direction_unknown"
    delta = float(start_elevation_m) - float(end_elevation_m)
    if delta >= flat_threshold_m:
        return "downhill_start_to_end"
    if delta <= -flat_threshold_m:
        return "downhill_end_to_start"
    return "approach_or_bidirectional"


def _plan_trail_areas(clusters: list[dict[str, Any]], proximity_km: float = 1.5) -> list[dict[str, Any]]:
    """Group clusters into stable connected components using a proximity graph."""
    if not clusters:
        return []
    centers = []
    for cluster in clusters:
        start, end = cluster.get("start", {}), cluster.get("end", {})
        centers.append(cluster.get("center") or {"lat": (float(start.get("lat", 0)) + float(end.get("lat", 0))) / 2, "lng": (float(start.get("lng", 0)) + float(end.get("lng", 0))) / 2})
    parent = list(range(len(clusters)))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            if _haversine_km(centers[left]["lat"], centers[left]["lng"], centers[right]["lat"], centers[right]["lng"]) <= proximity_km:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(clusters)):
        groups.setdefault(find(index), []).append(index)
    areas = []
    for area_id, indices in enumerate(sorted(groups.values(), key=lambda group: min(group))):
        lat = sum(centers[index]["lat"] for index in indices) / len(indices)
        lng = sum(centers[index]["lng"] for index in indices) / len(indices)
        areas.append({"area_id": area_id, "cluster_indices": indices, "center": {"lat": lat, "lng": lng}, "classified_trail_km": sum(float(clusters[index].get("total_length_km", 0) or 0) for index in indices)})
    return areas


def _cluster_entry_exit(cluster: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Choose conservative trail entry/exit points from available elevation."""
    start = cluster.get("start") or {}
    end = cluster.get("end") or {}
    direction = _classify_trail_direction(start.get("alt"), end.get("alt"))
    if direction == "downhill_end_to_start":
        return end, start, direction
    return start, end, direction


def _build_area_chain(
    areas: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    cluster_sequence: list[int],
) -> dict[str, Any]:
    """Build a directed area/cluster plan before routing."""
    area_sequence: list[int] = []
    for cluster_id in cluster_sequence:
        area_id = next((a.get("area_id") for a in areas if cluster_id in a.get("cluster_indices", [])), None)
        if area_id is not None and (not area_sequence or area_sequence[-1] != area_id):
            area_sequence.append(area_id)
    entry_exit_plan = []
    for cluster_id in cluster_sequence:
        cluster = clusters[cluster_id]
        entry, exit_, direction = _cluster_entry_exit(cluster)
        entry_exit_plan.append({
            "cluster_id": cluster_id,
            "entry_point": entry,
            "exit_point": exit_,
            "preferred_direction": direction,
            "direction_status": direction,
            "steepness_status": cluster.get("steepness_status", "unknown"),
            "difficulty_status": cluster.get("difficulty_status", "unknown"),
            "role": "trail_area",
        })
    connection_plan = [
        {"from_area": area_sequence[i], "to_area": area_sequence[i + 1], "role": "area_to_area_connection"}
        for i in range(len(area_sequence) - 1)
    ]
    return {
        "area_sequence": area_sequence,
        "cluster_sequence": list(cluster_sequence),
        "entry_exit_plan": entry_exit_plan,
        "connection_plan": connection_plan,
    }


def _ordered_area_waypoints(area: dict[str, Any], clusters: list[dict[str, Any]]) -> list[list[float]]:
    """Return ordered entry points for a planned forest/trail area."""
    points: list[list[float]] = []
    for index in area.get("cluster_indices", []):
        cluster = clusters[index]
        point = cluster.get("start") or cluster.get("end") or {}
        if point.get("lat") is not None and point.get("lng") is not None:
            points.append([round(float(point["lat"]), 6), round(float(point["lng"]), 6)])
    return points


def _candidate_waypoint_signature(cluster_ids: list[int], waypoints: list[list[float]]) -> tuple:
    return (
        tuple(sorted(cluster_ids)),
        tuple(sorted((round(float(p[0]), 6), round(float(p[1]), 6)) for p in waypoints)),
    )


def _route_metrics(route: dict[str, Any]) -> dict[str, float]:
    q = route.get("route_quality_metrics") or {}
    cov = (route.get("trail_coverage") or {}).get("classified") or {}
    return {
        "covered_clusters": float(cov.get("covered", 0)),
        "discovery_coverage": float(cov.get("coverage_percentage", 0)),
        "classified_trail_km": float(q.get(
            "classified_trail_km",
            (route.get("singletrail") or {}).get("singletrail_total_km", 0),
        )),
        "surface_offroad_km": float(q.get("surface_offroad_km", 0)),
        "asphalt_percentage": float(q.get("asphalt_percentage", 0)),
        "major_road_present": bool(_connection_waytype_audit(route.get("way_types"))["major_road_present"]),
        "cycleway_present": bool(_connection_waytype_audit(route.get("way_types"))["cycleway_present"]),
        "distance_km": float(route.get("distance_km", 0) or 0),
    }


def _adaptive_hard_violations(route: dict[str, Any], constraints: dict[str, Any]) -> int:
    violations = 0
    elevation = route.get("elevation_up_m")
    if isinstance(elevation, (int, float)) and isinstance(constraints.get("max_elevation_up_m"), (int, float)):
        violations += int(elevation > constraints["max_elevation_up_m"])
    tech = route.get("technical_difficulty")
    limit = constraints.get("technical_max")
    if isinstance(tech, str) and isinstance(limit, str) and tech.startswith("T") and limit.startswith("T"):
        try:
            violations += int(int(tech[1:]) > int(limit[1:]))
        except ValueError:
            pass
    return violations


def _adaptive_route_better(a: dict[str, Any], b: dict[str, Any], constraints: dict[str, Any]) -> bool:
    ah, bh = _adaptive_hard_violations(a, constraints), _adaptive_hard_violations(b, constraints)
    if ah != bh:
        return ah < bh
    am, bm = _route_metrics(a), _route_metrics(b)
    # Routed classified trail is the quality objective. Discovery coverage
    # only describes how much of the known candidate pool was reached.
    for key in ("classified_trail_km", "surface_offroad_km", "discovery_coverage", "covered_clusters"):
        if am[key] != bm[key]:
            return bool(am[key] > bm[key])
    if am["major_road_present"] != bm["major_road_present"]:
        return not bool(am["major_road_present"])
    if am["cycleway_present"] != bm["cycleway_present"]:
        return bool(am["cycleway_present"])

    amin, amax = constraints.get("min_distance_km"), constraints.get("max_distance_km")
    if amin is None and amax is None:
        if am["asphalt_percentage"] != bm["asphalt_percentage"]:
            return am["asphalt_percentage"] < bm["asphalt_percentage"]
        return int(a.get("_attempt", 999)) < int(b.get("_attempt", 999))
    def distance_key(d: float) -> tuple[int, float]:
        min_value = float(amin) if isinstance(amin, (int, float)) else None
        max_value = float(amax) if isinstance(amax, (int, float)) else None
        if min_value is not None and d < min_value:
            return (1, min_value - d)
        if max_value is not None and d > max_value:
            return (1, d - max_value)
        return (0, 0.0)
    ad, bd = distance_key(am["distance_km"]), distance_key(bm["distance_km"])
    if ad != bd:
        return ad < bd
    if am["asphalt_percentage"] != bm["asphalt_percentage"]:
        return am["asphalt_percentage"] < bm["asphalt_percentage"]
    return int(a.get("_attempt", 999)) < int(b.get("_attempt", 999))


def _adaptive_strategy_order(route: dict[str, Any], tested: set[str]) -> list[str]:
    base = ["classified_trail_coverage", "spatial_trail_diversity", "distance_aware", "balanced"]
    if _distance_strategy_needed(route) and "distance_aware" in base:
        base.remove("distance_aware")
        base.insert(0, "distance_aware")
    return [strategy for strategy in base if strategy not in tested]


def _build_feedback_transition(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "from_variant_id": previous.get("variant_id"),
        "to_variant_id": current.get("variant_id"),
        "distance_delta_km": round(float(current.get("distance_km", 0)) - float(previous.get("distance_km", 0)), 3),
        "classified_trail_delta_km": round(
            float((current.get("route_quality_metrics") or {}).get("classified_trail_km", 0))
            - float((previous.get("route_quality_metrics") or {}).get("classified_trail_km", 0)), 3,
        ),
    }


def _adaptive_has_meaningful_next_strategy(
    route: dict[str, Any],
    tested_strategies: set[str],
    constraints: dict[str, Any],
) -> bool:
    """Decide whether another attempt has a concrete, untested purpose."""
    if _distance_strategy_needed({**route, "selection_constraints": constraints}) and "distance_aware" not in tested_strategies:
        return True
    clusters = route.get("trail_clusters") or {}
    remaining = [
        c for bucket in ("uncovered", "partial")
        for c in clusters.get(bucket, [])
        if "classified" in (c.get("trail_categories") or [])
        and c.get("coverage_percentage", 0) < 70
    ]
    if not remaining:
        # A route can cover every discovered cluster while still being
        # materially outside the requested distance window. Keep searching
        # with independent strategies until the distance target is feasible.
        if _distance_strategy_needed({**route, "selection_constraints": constraints}):
            return any(strategy not in tested_strategies for strategy in (
                "classified_trail_coverage", "spatial_trail_diversity",
                "distance_aware", "balanced",
            ))
        return False
    return any(strategy not in tested_strategies for strategy in (
        "classified_trail_coverage", "spatial_trail_diversity", "distance_aware", "balanced"
    ))


def _build_adaptive_variant(
    strategy: str,
    clusters: list[dict[str, Any]],
    analyzed: list[dict[str, Any]],
    used_cluster_ids: set[int],
    used_signatures: set[tuple],
    current_waypoints: list[list[float]],
    trail_areas: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    # Reuse is allowed; only the exact cluster/waypoint signature is forbidden.
    candidates = [c for c in analyzed if c.get("coverage_percentage", 0) < 70]
    if strategy == "distance_aware" and not candidates:
        # Distance recovery may need to revisit already covered clusters;
        # coverage completion alone must not suppress longer variants.
        candidates = list(analyzed)
    candidates = [c for c in candidates if "classified" in (c.get("trail_categories") or [])]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.get("coverage_percentage", 0), -c.get("total_length_km", 0)))
    if strategy in {"distance_aware", "balanced"}:
        candidates.sort(key=lambda c: c.get("total_length_km", 0), reverse=True)
    if strategy == "spatial_trail_diversity" and used_cluster_ids:
        prior = [clusters[i].get("center", {}) for i in used_cluster_ids if i < len(clusters)]
        def dist(c: dict[str, Any]) -> float:
            center = c.get("center", {})
            return min((_haversine_km(center.get("lat", 0), center.get("lng", 0), p.get("lat", 0), p.get("lng", 0)) for p in prior), default=0)
        candidates.sort(key=lambda c: (dist(c), -c.get("coverage_percentage", 0), c.get("total_length_km", 0)), reverse=True)
    # Prefer candidates from one planned area, then adjacent areas, instead
    # of selecting unrelated clusters solely by trail length.
    if trail_areas:
        area_by_cluster = {
            idx: area.get("area_id")
            for area in trail_areas
            for idx in area.get("cluster_indices", [])
        }
        start = current_waypoints[0] if current_waypoints else None
        area_distance = {
            area.get("area_id"): (
                _haversine_km(float(start[0]), float(start[1]), area["center"]["lat"], area["center"]["lng"])
                if start else float(area.get("area_id", 999))
            )
            for area in trail_areas
        }
        candidates.sort(key=lambda c: (
            area_distance.get(area_by_cluster.get(c.get("cluster_index")), float("inf")),
            c.get("coverage_percentage", 0),
        ))
    # Try alternate pairs when the preferred pair duplicates a prior variant.
    for selected in combinations(candidates, min(2, len(candidates))):
        waypoints: list[list[float]] = []
        selected_ids: list[int] = []
        for c in selected:
            idx = c.get("cluster_index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(clusters):
                continue
            cl = clusters[idx]
            selected_ids.append(idx)
            entry, exit_, _ = _cluster_entry_exit(cl)
            for point in (entry, exit_):
                if point and point.get("lat") is not None and point.get("lng") is not None:
                    waypoints.append([round(float(point["lat"]), 6), round(float(point["lng"]), 6)])
        sig = _candidate_waypoint_signature(selected_ids, waypoints)
        if waypoints and sig not in used_signatures:
            chain = _build_area_chain(trail_areas or [], clusters, selected_ids)
            chain_waypoints: list[list[float]] = []
            for item in chain["entry_exit_plan"]:
                for key in ("entry_point", "exit_point"):
                    point = item.get(key) or {}
                    if point.get("lat") is not None and point.get("lng") is not None:
                        chain_waypoints.append([round(float(point["lat"]), 6), round(float(point["lng"]), 6)])
            return {"strategy": strategy, "selected_cluster_ids": selected_ids, "excluded_cluster_ids": sorted(used_cluster_ids), "waypoints": chain_waypoints or waypoints, "signature": _candidate_waypoint_signature(selected_ids, chain_waypoints or waypoints), **chain}
    return None


def _distance_strategy_needed(route: dict[str, Any]) -> bool:
    """Whether a soft distance target still warrants another strategy."""
    constraints = route.get("selection_constraints") or {}
    distance = route.get("distance_km")
    if not isinstance(distance, (int, float)):
        return False
    minimum = constraints.get("min_distance_km")
    maximum = constraints.get("max_distance_km")
    return ((isinstance(minimum, (int, float)) and distance < minimum)
            or (isinstance(maximum, (int, float)) and distance > maximum))


async def _run_adaptive_variant_controller(
    initial_coordinates: list[list[float]], sport: str, discovered_segments: list[dict[str, Any]],
    discovered_clusters: list[dict[str, Any]], plan_route_fn: Any,
    selection_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    constraints = selection_constraints or {}
    results: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    used_sigs: set[tuple] = set()
    tested_strategies: set[str] = set()
    trail_areas = _plan_trail_areas(discovered_clusters)
    cluster_area = {
        cluster_index: area["area_id"]
        for area in trail_areas
        for cluster_index in area["cluster_indices"]
    }
    strategies = ["baseline", "classified_trail_coverage", "spatial_trail_diversity", "distance_aware", "balanced"]
    feedback_transitions: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ADAPTIVE_ROUTING_CALLS + 1):
        if attempt == 1:
            plan = {"strategy": "baseline", "selected_cluster_ids": [], "excluded_cluster_ids": [], "waypoints": [], "signature": ((), ())}
        else:
            prior = results[-1]
            analyzed = _build_analyzed_from_clusters(prior.get("trail_clusters") or {})
            strategy_order = _adaptive_strategy_order(prior, tested_strategies)
            if not strategy_order:
                break
            strategy = strategy_order[0]
            plan = _build_adaptive_variant(strategy, discovered_clusters, analyzed, used_ids, used_sigs, initial_coordinates, trail_areas=trail_areas)
            if plan is None:
                break
        coords = _build_route_coordinates(initial_coordinates, plan["waypoints"], "auto")
        route = await plan_route_fn(coords, sport=sport, compact=True, discovered_segments=discovered_segments, discovered_clusters=discovered_clusters, **constraints)
        route["route_mode"] = "roundtrip" if coords and coords[0] == coords[-1] else "point_to_point"
        route["roundtrip_closed"] = bool(coords and coords[0] == coords[-1])
        route["comparison_route_available"] = attempt > 1
        route["trail_area_sequence"] = [cluster_area.get(cluster_id) for cluster_id in plan["selected_cluster_ids"]]
        route["planning_audit"] = _planning_audit(
            route, coords, trail_areas, discovered_clusters,
            plan["selected_cluster_ids"], [],
        )
        route["planning_audit"]["connection_policy"] = ["cycleway", "forest_or_economic_road", "quiet_local_road", "major_road_last_resort"]
        route["planning_audit"]["connection_waytype_audit"] = _connection_waytype_audit(route.get("way_types"), comparison_available=attempt > 1)
        route["_attempt"] = attempt
        route["variant_id"] = f"variant_{attempt}"
        route["strategy"] = plan["strategy"]
        tested_strategies.add(plan["strategy"])
        route["selection_constraints"] = constraints
        route["_waypoints"] = coords
        route["variant_specific_waypoints"] = [list(w) for w in plan["waypoints"]]
        route["selected_cluster_ids"] = plan["selected_cluster_ids"]
        route["route_fingerprint"] = (route.get("_route_geometry_fingerprint") or {}).get("hash")
        similarities = [_geometry_similarity_pct(route.get("_route_geometry_fingerprint") or {}, old.get("_route_geometry_fingerprint") or {}) for old in results]
        route["geometry_similarity"] = {str(i + 1): p for i, p in enumerate(similarities)}
        max_similarity = max(similarities, default=0.0)
        if max_similarity >= ROUTE_REDUNDANCY_OVERLAP_PCT:
            status = "rejected_duplicate"
        elif _adaptive_hard_violations(route, constraints):
            status = "rejected_constraint"
        else:
            status = "accepted"
        route["candidate_status"] = status
        route["attempt_reason"] = f"adaptive strategy: {plan['strategy']}"
        excluded_ids = plan["excluded_cluster_ids"]
        route["excluded_cluster_reason"] = (
            {str(cluster_id): "previously selected in another variant; excluded to diversify cluster combinations"
             for cluster_id in excluded_ids}
            if excluded_ids else {}
        )
        audit.append({
            "variant_id": route["variant_id"], "strategy": plan["strategy"],
            "selected_cluster_ids": plan["selected_cluster_ids"],
            "excluded_cluster_ids": plan["excluded_cluster_ids"],
            "excluded_cluster_reason": route["excluded_cluster_reason"],
            "candidate_status": status, "attempt_reason": route["attempt_reason"],
            "variant_waypoints": route["variant_specific_waypoints"],
            "route_fingerprint": route.get("route_fingerprint"),
            "similarity_to_existing": route["geometry_similarity"],
            "selected_waypoints": route["variant_specific_waypoints"],
        })
        results.append(route)
        if len(results) > 1:
            feedback_transitions.append(_build_feedback_transition(results[-2], results[-1]))
        used_ids.update(plan["selected_cluster_ids"])
        used_sigs.add(plan["signature"])
        if status == "accepted":
            if not any(r.get("candidate_status") == "accepted_best" for r in results):
                route["candidate_status"] = "accepted_best"
            else:
                best = next(r for r in results if r.get("candidate_status") == "accepted_best")
                if _adaptive_route_better(route, best, constraints):
                    best["candidate_status"] = "accepted"
                    route["candidate_status"] = "accepted_best"
        if attempt < MAX_ADAPTIVE_ROUTING_CALLS and not _adaptive_has_meaningful_next_strategy(route, tested_strategies, constraints):
            break
    valid = [r for r in results if r.get("candidate_status") in {"accepted", "accepted_best"}]
    if not valid and results:
        valid = [r for r in results if r.get("candidate_status") != "rejected_duplicate"] or results
    selected = min(valid, key=lambda r: r.get("_attempt", 999)) if not valid else valid[0]
    for r in valid[1:]:
        if _adaptive_route_better(r, selected, constraints):
            selected = r
    for entry in audit:
        entry["candidate_status"] = next(
            (r.get("candidate_status") for r in results if r.get("variant_id") == entry.get("variant_id")),
            entry.get("candidate_status"),
        )
        entry["selected"] = entry.get("variant_id") == selected.get("variant_id")
    best = selected
    return {"status": "feedback_complete", "best_route": best, "iterations": results, "total_attempts": len(results), "feedback_decisions": audit, "feedback_transitions": feedback_transitions}


async def _run_feedback_loop(
    initial_coordinates: list[list[float]],
    sport: str,
    discovered_segments: list[dict[str, Any]],
    discovered_clusters: list[dict[str, Any]],
    plan_route_fn: Any,
    max_iterations: int = 3,
    selection_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run up to *max_iterations* routing attempts with coverage-guided tuning.

    Each iteration:
      1. Calls *plan_route_fn* with current waypoints + discovered data.
      2. Analyses trail coverage → identifies uncovered/partial clusters.
      3. Adds up to 2 targeted waypoints for the worst-covered clusters.

    Stops early when:
      - All classified clusters are fully covered (>70 %).
      - No uncovered clusters remain.
      - suggest_uncovered_waypoints() returns no viable waypoints.
      - *max_iterations* reached.

    On completion, compares all iterations qualitatively and returns
    the best result with a full iteration log.

    Discovery is run ONCE before this function — it never re-queries
    discovered_segments/discovered_clusters.

    Args:
        initial_coordinates: Starting waypoints as [lat, lng] pairs.
        sport: Sport type for routing.
        discovered_segments: Trail segments from extract_trail_segments().
        discovered_clusters: Trail clusters from cluster_trail_segments().
        plan_route_fn: Async callable matching plan_route(..., compact=True).
        max_iterations: Maximum routing attempts (default 3, min 1).
        selection_constraints: Optional dict of hard constraint limits
            (min_distance_km, max_distance_km, max_elevation_up_m,
             technical_max) used in qualitative comparison.

    Returns:
        Dict with:
          status: "feedback_complete" or "error".
          best_route: compact route summary of the best iteration.
          iterations: list of compact iteration summaries.
          best_index: index into iterations of the chosen route.
          total_attempts: number of actual routing attempts.
          improvement_log: human-readable notes per iteration.
          clusters_covered: classified / unclassified coverage counts.
          clusters_uncovered_indices: list of uncovered cluster indices.
    """
    if max_iterations < 1:
        return {"status": "error", "error": "max_iterations must be at least 1"}
    if not discovered_segments:
        return {"status": "error", "error": "discovered_segments required for feedback loop"}

    current_coords = [list(c) for c in initial_coordinates]
    raw_iterations: list[dict[str, Any]] = []
    improvement_log: list[str] = []
    feedback_decisions: list[dict[str, Any]] = []
    feedback_transitions: list[dict[str, Any]] = []

    for attempt in range(1, max_iterations + 1):
        # ── Route ────────────────────────────────────────────────
        kwargs: dict[str, Any] = dict(
            sport=sport,
            compact=True,
            discovered_segments=discovered_segments,
            discovered_clusters=discovered_clusters,
        )
        if selection_constraints:
            # Forward the public plan_route parameters, not the internal
            # comparison dictionary. This keeps recursive real calls
            # signature-compatible while preserving the internal metadata.
            for key in (
                "min_distance_km",
                "max_distance_km",
                "max_elevation_up_m",
                "technical_max",
            ):
                if key in selection_constraints:
                    kwargs[key] = selection_constraints[key]
        route = await plan_route_fn(current_coords, **kwargs)
        route["_attempt"] = attempt
        if selection_constraints is not None:
            route["selection_constraints"] = dict(selection_constraints)
        route["_waypoints"] = [list(c) for c in current_coords]
        if raw_iterations:
            previous_route = raw_iterations[-1]
            feedback_transitions.append({
                "from_attempt": previous_route.get("_attempt"),
                "to_attempt": attempt,
                "cluster_diff": _cluster_transition(
                    previous_route.get("trail_clusters") or {},
                    route.get("trail_clusters") or {},
                ),
                "metrics": {
                    "distance_km": {"before": previous_route.get("distance_km"), "after": route.get("distance_km")},
                    "elevation_up_m": {"before": previous_route.get("elevation_up_m"), "after": route.get("elevation_up_m")},
                    "asphalt": {"before": (previous_route.get("surfaces") or {}).get("asphalt"), "after": (route.get("surfaces") or {}).get("asphalt")},
                    "trail_coverage": {"before": (previous_route.get("trail_coverage") or {}).get("classified"), "after": (route.get("trail_coverage") or {}).get("classified")},
                },
            })
        raw_iterations.append(route)

        # ── Decide whether to continue ───────────────────────────
        if attempt >= max_iterations:
            improvement_log.append(
                f"Attempt {attempt}: final iteration (max={max_iterations})."
            )
            break

        trail_clusters = route.get("trail_clusters", None) or {}
        uncovered = trail_clusters.get("uncovered", [])
        partial = trail_clusters.get("partial", [])

        # Early stop: all classified clusters covered?
        all_classified_done = all(
            "classified" not in (c.get("trail_categories") or [])
            for c in uncovered
        )
        if all_classified_done and not partial:
            improvement_log.append(
                f"Attempt {attempt}: all classified clusters covered, stopping."
            )
            break

        # ── Generate targeted waypoints ──────────────────────────
        analyzed = _build_analyzed_from_clusters(trail_clusters)
        suggestions, candidate_audit = _suggest_uncovered_waypoints_with_audit(
            discovered_clusters,
            analyzed,
            current_waypoints=current_coords,
            max_new=2,
        )
        feedback_decisions.append({
            "from_attempt": attempt,
            "to_attempt": attempt + 1,
            "candidates": candidate_audit,
            "selected_waypoints": [item for item in candidate_audit if item.get("selected")],
        })

        if not suggestions:
            improvement_log.append(
                f"Attempt {attempt}: no viable waypoint suggestions, stopping."
            )
            break

        n_uncovered = len(uncovered)
        n_partial = len(partial)
        improvement_log.append(
            f"Attempt {attempt}: added {len(suggestions)} waypoint(s) "
            f"({n_uncovered} uncovered + {n_partial} partial clusters)."
        )

        current_coords.extend(suggestions)

    # ── Select best route ──────────────────────────────────────────
    best_index = _select_best_route_index(raw_iterations)

    # ── Build compact iteration summaries ─────────────────────────
    summaries: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_iterations):
        wp = raw.get("_waypoints", initial_coordinates)
        if i == 0:
            wp_added_this = 0
        else:
            prev_wp = raw_iterations[i - 1].get("_waypoints", initial_coordinates)
            wp_added_this = len(wp) - len(prev_wp)
        summaries.append(_feedback_iteration_summary(raw, i + 1, wp, wp_added_this))

    # ── Cluster summary ────────────────────────────────────────────
    best_route = raw_iterations[best_index]
    best_tc = best_route.get("trail_clusters", None) or {}
    clusters_covered = {
        "classified": sum(
            1 for c in best_tc.get("covered", [])
            if "classified" in (c.get("trail_categories") or [])
        ),
        "unclassified": sum(
            1 for c in best_tc.get("covered", [])
            if "unclassified" in (c.get("trail_categories") or [])
        ),
    }
    uncovered_indices = [
        c["cluster_index"] for c in best_tc.get("uncovered", [])
    ]

    return {
        "status": "feedback_complete",
        "best_route": _feedback_iteration_summary(
            best_route,
            best_route.get("_attempt", best_index + 1),
            best_route.get("_waypoints", initial_coordinates),
            0,
        ),
        "iterations": summaries,
        "best_index": best_index,
        "total_attempts": len(raw_iterations),
        "improvement_log": improvement_log,
        "clusters_covered": clusters_covered,
        "clusters_uncovered_indices": uncovered_indices,
        "feedback_decisions": feedback_decisions,
        "feedback_transitions": feedback_transitions,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    a = max(0.0, min(1.0, a))
    return R * 2 * math.asin(math.sqrt(a))


def _segment_km(coords: list[dict[str, Any]], from_i: int, to_i: int) -> float:
    """Sum of Haversine distances between consecutive coords from from_i to to_i."""
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


def _compute_singletrail(
    wt_items: list[dict[str, Any]],
    coord_items: list[dict[str, Any]],
    total_km: float,
) -> dict[str, Any]:
    """Compute trail breakdown from way-type items and coordinates.

    Extracts both classified singletrail (trail_d1..trail_d5) and unclassified
    trail (wt#trail without numeric suffix), merges overlapping intervals,
    calculates Haversine distance, and returns totals with percentages.

    Returns {'available': False} when no trail data is present.
    """
    if not wt_items or not coord_items:
        return {"available": False}

    # Separate classified (trail_d1..d5) and unclassified (wt#trail) items
    classified: dict[str, list[dict[str, Any]]] = {}
    unclassified: list[dict[str, Any]] = []

    for item in wt_items:
        if not isinstance(item, dict):
            continue
        element = item.get("element", "")
        if not isinstance(element, str):
            continue

        if element.startswith("wt#trail_d"):
            key = element.replace("wt#", "")  # trail_d1 .. trail_d5
            classified.setdefault(key, []).append({
                "from": item.get("from"),
                "to": item.get("to"),
            })
        elif element == "wt#trail":
            unclassified.append({
                "from": item.get("from"),
                "to": item.get("to"),
            })

    if not classified and not unclassified:
        return {"available": False}

    st_out: dict[str, Any] = {}
    classified_total_km = 0.0
    unclassified_total_km = 0.0

    def _process_intervals(intervals, coord_items):
        """Merge overlapping intervals and compute Haversine distance."""
        if not intervals:
            return None, None
        intervals.sort(key=lambda x: int(x.get("from", 0) or 0))
        merged: list[dict[str, Any]] = []
        for iv in intervals:
            if not merged:
                merged.append(dict(iv))
            else:
                last = merged[-1]
                fv = int(iv.get("from") or 0)
                lv = int(iv.get("to") or 0)
                lf = int(last.get("from") or 0)
                lt = int(last.get("to") or 0)
                if fv <= lt:
                    last["to"] = max(lt, lv)
                else:
                    merged.append(dict(iv))
        d_km: float | None = None
        if coord_items and merged:
            try:
                seg_d = 0.0
                for m in merged:
                    fi = int(m.get("from") or 0)
                    ti = int(m.get("to") or 0)
                    if 0 <= fi < ti <= len(coord_items):
                        seg_d += _segment_km(coord_items, fi, ti)
                if seg_d > 0:
                    d_km = round(seg_d, 2)
            except (TypeError, ValueError, IndexError):
                d_km = None
        return merged, d_km

    # Process classified trail_d1..d5
    for trail_key in sorted(classified.keys()):
        intervals = classified[trail_key]
        merged, d_km = _process_intervals(intervals, coord_items)
        entry: dict[str, Any] = {
            "segments": len(intervals),
            "from_to": [[int(iv["from"]), int(iv["to"])] for iv in intervals],
        }
        if d_km is not None:
            entry["distance_km"] = d_km
            classified_total_km += d_km
        st_out[trail_key] = entry

    # Process unclassified wt#trail
    if unclassified:
        merged, d_km = _process_intervals(unclassified, coord_items)
        entry = {
            "segments": len(unclassified),
            "from_to": [[int(iv["from"]), int(iv["to"])] for iv in unclassified],
        }
        if d_km is not None:
            entry["distance_km"] = d_km
            unclassified_total_km = d_km
        st_out["trail_unclassified"] = entry

    # Ordered: d1..d5 first, then unclassified
    ordered: dict[str, Any] = {}
    for d in range(1, 6):
        key = f"trail_d{d}"
        if key in st_out:
            ordered[key] = st_out[key]
    if "trail_unclassified" in st_out:
        ordered["trail_unclassified"] = st_out["trail_unclassified"]

    if classified_total_km > 0:
        ordered["singletrail_total_km"] = round(classified_total_km, 2)
        if total_km and total_km > 0:
            ordered["singletrail_percentage"] = round(
                classified_total_km / total_km * 100, 1
            )
    if unclassified_total_km > 0:
        ordered["unclassified_trail_total_km"] = round(unclassified_total_km, 2)
        if total_km and total_km > 0:
            ordered["unclassified_trail_percentage"] = round(
                unclassified_total_km / total_km * 100, 1
            )

    return ordered


def _compact_route_summary(
    raw: dict[str, Any],
    discovered_segments: list[dict[str, Any]] | None = None,
    discovered_clusters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert raw plan_route API response into a compact route summary.

    Uses the same extraction pattern as import_gpx_file, enriched with
    singletrail data from _embedded way-type items + coordinates.
    No additional API calls are made.

    When discovered_segments are provided, also computes trail_coverage
    and (if clusters provided) per-cluster coverage breakdown.
    """
    result: dict[str, Any] = {}

    # Distance
    if "distance" in raw:
        d = float(raw["distance"])
        result["distance_km"] = round(d / 1000.0, 2)
        result["distance_m"] = round(d, 1)

    # Elevation
    if "elevation_up" in raw:
        result["elevation_up_m"] = round(float(raw["elevation_up"]), 1)
    if "elevation_down" in raw:
        result["elevation_down_m"] = round(float(raw["elevation_down"]), 1)

    # Duration
    if "duration" in raw:
        dur_s = int(float(raw["duration"]))
        result["duration_seconds"] = dur_s
        h = dur_s // 3600
        m = (dur_s % 3600) // 60
        result["duration"] = f"{h}h {m}m" if h else f"{m}m"

    # Difficulty (grade + T/C)
    diff = raw.get("difficulty")
    if isinstance(diff, dict):
        if diff.get("grade"):
            result["difficulty"] = diff["grade"]
        tech_raw = diff.get("explanation_technical", "")
        if tech_raw and isinstance(tech_raw, str) and "#" in tech_raw:
            result["technical_difficulty"] = tech_raw.split("#", 1)[1].upper()
        fit_raw = diff.get("explanation_fitness", "")
        if fit_raw and isinstance(fit_raw, str) and "#" in fit_raw:
            result["fitness_difficulty"] = fit_raw.split("#", 1)[1].upper()

    total_km = result.get("distance_km", 0)

    # Way types (percentage + approximated km from total distance)
    summary = raw.get("summary", {})
    if isinstance(summary, dict):
        wts = summary.get("way_types", [])
        if isinstance(wts, list):
            result["way_types"] = {
                str(item["type"]).replace("wt#", ""): {
                    "percentage": round(float(item["amount"]) * 100, 1),
                    "distance_km": round(float(item["amount"]) * total_km, 2)
                    if total_km > 0 else 0.0,
                }
                for item in wts
                if isinstance(item, dict) and "type" in item and "amount" in item
            }

        # Surfaces (percentage + approximated km from total distance)
        surfaces = summary.get("surfaces", [])
        if isinstance(surfaces, list):
            result["surfaces"] = {
                str(item["type"]).replace("sm#", ""): {
                    "percentage": round(float(item["amount"]) * 100, 1),
                    "distance_km": round(float(item["amount"]) * total_km, 2)
                    if total_km > 0 else 0.0,
                }
                for item in surfaces
                if isinstance(item, dict) and "type" in item and "amount" in item
            }

    # Singletrail (from _embedded.way_type items + coordinates)
    embedded = raw.get("_embedded", {})
    if isinstance(embedded, dict):
        wt_container = embedded.get("way_types", {})
        wt_items: list[dict[str, Any]] = []
        if isinstance(wt_container, dict):
            wt_items = wt_container.get("items", [])
        elif isinstance(wt_container, list):
            wt_items = wt_container
        if not isinstance(wt_items, list):
            wt_items = []

        coord_container = embedded.get("coordinates", {})
        coord_items: list[dict[str, Any]] = []
        if isinstance(coord_container, dict):
            coord_items = coord_container.get("items", [])
        elif isinstance(coord_container, list):
            coord_items = coord_container
        if not isinstance(coord_items, list):
            coord_items = []

        result["singletrail"] = _compute_singletrail(
            wt_items, coord_items, total_km
        )

        # Coordinate count — NOT the full list
        if coord_items:
            result["matched_coordinates"] = len(coord_items)

        # Route overlap — detect duplicate track sections
        result["route_overlap"] = _compute_route_overlap(coord_items)
        result["_route_geometry_fingerprint"] = _route_geometry_fingerprint(coord_items)

    # Normalized metrics. Discovery coverage is added separately below and is
    # intentionally not part of classified-trail metrics.
    surface_data = result.get("surfaces") or {}
    singletrail_data = result.get("singletrail") or {}

    def _surface_value(name: str, field: str) -> float:
        value = surface_data.get(name, {})
        return float(value.get(field, 0.0)) if isinstance(value, dict) else 0.0

    result["route_quality_metrics"] = {
        "classified_trail_km": round(float(singletrail_data.get("singletrail_total_km", 0.0)), 2),
        "classified_trail_percentage": round(float(singletrail_data.get("singletrail_percentage", 0.0)), 1),
        "surface_offroad_km": round(sum(_surface_value(k, "distance_km") for k in ("unpaved", "gravel", "nature")), 2),
        "surface_offroad_percentage": round(sum(_surface_value(k, "percentage") for k in ("unpaved", "gravel", "nature")), 1),
        "asphalt_km": round(_surface_value("asphalt", "distance_km"), 2),
        "asphalt_percentage": round(_surface_value("asphalt", "percentage"), 1),
    }

    # Segments
    segs = raw.get("segments", [])
    if isinstance(segs, list):
        routed = sum(
            1 for s in segs if isinstance(s, dict) and s.get("type") == "Routed"
        )
        manual = sum(
            1 for s in segs if isinstance(s, dict) and s.get("type") == "Manual"
        )
        result["segments"] = {"routed": routed, "manual": manual}

    # Tour information
    ti = raw.get("tour_information", [])
    if isinstance(ti, list) and ti:
        info_list: list[dict[str, Any]] = []
        for item in ti:
            if isinstance(item, dict) and item.get("type"):
                segs = item.get("segments", [])
                info_list.append({
                    "type": str(item["type"]),
                    "segments": len(segs) if isinstance(segs, list) else 0,
                })
        result["tour_information"] = info_list

    # Trail coverage from discovered segments (if provided)
    if discovered_segments:
        embedded = raw.get("_embedded", {})
        if isinstance(embedded, dict):
            coord_container = embedded.get("coordinates", {})
            coord_items: list[dict[str, Any]] = []
            if isinstance(coord_container, dict):
                coord_items = coord_container.get("items", [])
            elif isinstance(coord_container, list):
                coord_items = coord_container
            if not isinstance(coord_items, list):
                coord_items = []

            if coord_items:
                result["trail_coverage"] = _compute_trail_coverage(
                    coord_items, discovered_segments
                )
                # Explicit name prevents confusion with routed classified trail.
                result["discovery_trail_coverage"] = result["trail_coverage"]

                if discovered_clusters:
                    analyzed = analyze_cluster_coverage(
                        discovered_clusters, coord_items
                    )
                    result["trail_clusters"] = _compact_cluster_info(analyzed)
                    result["trail_coverage_suggestions"] = suggest_uncovered_waypoints(
                        discovered_clusters, analyzed, max_new=2,
                    )

    return result


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

        # Etape 2 : extraire l'item importe (pas le wrapper complet)
        try:
            item = import_result["_embedded"]["items"][0]
        except (KeyError, IndexError, TypeError):
            return {
                "error": "Komoot returned no import item",
                "response": import_result,
            }

        # Etape 3 : route matching
        matched = await client.post_www(
            "/routing/import/tour",
            json_body=item,
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

        # Extract Komoot's technical and fitness difficulty ratings.
        if isinstance(difficulty, dict):
            tech_raw = difficulty.get("explanation_technical", "")
            if tech_raw and isinstance(tech_raw, str) and "#" in tech_raw:
                result["technical_difficulty"] = tech_raw.split("#", 1)[1].upper()
            fit_raw = difficulty.get("explanation_fitness", "")
            if fit_raw and isinstance(fit_raw, str) and "#" in fit_raw:
                result["fitness_difficulty"] = fit_raw.split("#", 1)[1].upper()

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

        # parent_tour_id: the imported file's tour ID, if assigned by Komoot.
        # This is present when the matched route is linked to an existing
        # Komoot tour (e.g., a previously created planned tour).
        if "id" in tour:
            result["parent_tour_id"] = tour["id"]

        return result

    @mcp.tool()
    async def plan_route(
        coordinates: list[list[float]],
        sport: str = "hike",
        compact: bool = False,
        discovered_segments: list[dict[str, Any]] | None = None,
        discovered_clusters: list[dict[str, Any]] | None = None,
        feedback_loop: bool = False,
        min_distance_km: float | None = None,
        max_distance_km: float | None = None,
        max_elevation_up_m: float | None = None,
        technical_max: str | None = None,
    ) -> dict[str, Any]:
        """Planifier un itineraire Komoot a partir de waypoints.

        Args:
            coordinates: liste de [lat, lng] pour les points de passage
                         (ex: [[48.5734, 7.7521], [48.5801, 7.7612]]).
            sport: type de sport ('hike', 'touringbicycle', 'mtb', 'racebicycle', 'jogging').
            compact: si True, renvoie un resume compact de la route
                     (distance, elevation, duree, difficulte, way_types,
                     surfaces, singletrail, segments) au lieu de la
                     reponse complete de l'API Komoot. Defaut: False
                     (comportement historique, retrocompatible).
            discovered_segments: liste optionnelle de segments de trail decouverts
                                 (depuis extract_trail_segments()). Quand fournis,
                                 le resume contient trail_coverage.
            discovered_clusters: liste optionnelle de clusters de trail (depuis
                                 cluster_trail_segments()). Quand fournis avec
                                 discovered_segments, le resume contient aussi
                                 trail_clusters (covered/partial/uncovered).
            feedback_loop: si True, lance le controleur adaptatif avec jusqu'a
                           5 candidats de routage independants guides par les
                           contraintes, les trails decouverts et la distance.
                           Necessite discovered_segments. Defaut: False.
            min_distance_km: Distance minimale souhaitee (hard constraint
                             de selection, pas de routage Komoot).
            max_distance_km: Distance maximale souhaitee (hard constraint
                             de selection, pas de routage Komoot).
            max_elevation_up_m: Denivele max (hard constraint de selection).
            technical_max: Difficulté technique max ex. 'T2' (hard constraint
                           de selection).
        """
        # ── Build selection_constraints dict ─────────────────────
        selection_constraints: dict[str, Any] = {}
        if min_distance_km is not None:
            selection_constraints["min_distance_km"] = min_distance_km
        if max_distance_km is not None:
            selection_constraints["max_distance_km"] = max_distance_km
        if max_elevation_up_m is not None:
            selection_constraints["max_elevation_up_m"] = max_elevation_up_m
        if technical_max is not None:
            selection_constraints["technical_max"] = technical_max

        # ── Feedback loop mode ────────────────────────────────────
        if feedback_loop:
            return await _run_adaptive_variant_controller(
                coordinates, sport,
                discovered_segments or [],
                discovered_clusters or [],
                plan_route,
                selection_constraints=selection_constraints if selection_constraints else None,
            )

        # Build path (indexed waypoints) and segments (Routed geometry between consecutive points)
        path = []
        segments = []
        for i, coord in enumerate(coordinates):
            point = {"lat": coord[0], "lng": coord[1], "alt": 0.0}
            path.append({"location": point, "index": i})
            if i > 0:
                prev_coord = coordinates[i - 1]
                prev_point = {"lat": prev_coord[0], "lng": prev_coord[1], "alt": 0.0}
                segments.append({"type": "Routed", "geometry": [prev_point, point]})

        result = await client.post_www(
            "/routing/tour",
            json_body={
                "path": path,
                "segments": segments,
                "sport": sport,
            },
            params={
                "_embedded": "coordinates,way_types,surfaces,directions",
            },
        )
        if compact:
            return _compact_route_summary(
                result, discovered_segments, discovered_clusters,
            )

        # compact=False: cache full result internally, return compact summary + route_ref
        route_ref = f"route_{uuid4().hex}"
        now = time.time()
        async with _route_cache_lock:
            # Opportunistic cleanup of expired entries
            for ref in list(_route_cache.keys()):
                if now - _route_cache[ref][0] > _CACHE_TTL:
                    del _route_cache[ref]
            _route_cache[route_ref] = (now, result)

        summary = _compact_route_summary(
            result, discovered_segments, discovered_clusters,
        )
        # Geometry fingerprints are controller-internal and must not inflate
        # the public compact=False response; the full route remains cached.
        summary.pop("_route_geometry_fingerprint", None)
        summary["route_ref"] = route_ref
        return summary

    def _prepare_planned_tour_payload(
        route_data: dict[str, Any],
        name: str = "Tour planifie",
        sport: str = "hike",
    ) -> dict[str, Any]:
        """Normalise route_data en payload pour POST /v007/tours/?reroute=true.

        Si route_data est le resultat brut d'import_gpx_route (contenant
        _embedded.matched), extrait automatiquement le tour matche.
        Agit sur une copie, ne mute jamais l'original.

        Args:
            route_data: resultat de import_gpx_route() ou plan_route().
            name: nom du tour planifie.
            sport: type de sport.
        """
        payload = dict(route_data)
        # Si c'est le resultat brut de import_gpx_route
        embedded = payload.get("_embedded")
        if isinstance(embedded, dict) and "matched" in embedded:
            payload = dict(embedded["matched"])
        payload["name"] = name
        payload["sport"] = sport
        # Remove fields that would break the POST
        for key in list(payload.keys()):
            if key in ("_links", "uri"):
                del payload[key]
        return payload

    def _validate_route_data_for_save(
        route_data: dict[str, Any],
    ) -> str | None:
        """Validate route_data is suitable for create_planned_tour.

        Returns None if valid, or an error string explaining what's wrong.

        Checks:
        - Fields use the full plan_route() format (compact=False), not the
          compact summary (compact=True).
        - Required fields exist and have correct types:
          distance (numeric), duration (numeric), elevation_up (numeric),
          elevation_down (numeric), path (list), segments (list),
          _embedded (dict).
        """
        # ── Detect compact=True format ──────────────────────────────────
        if "distance_km" in route_data:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). The compact summary cannot be used for saving. "
                "Detected 'distance_km' field (compact format); expected 'distance'."
            )
        if "elevation_up_m" in route_data:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). Detected compact-style field "
                "'elevation_up_m'; expected 'elevation_up'."
            )
        if "elevation_down_m" in route_data:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). Detected compact-style field "
                "'elevation_down_m'; expected 'elevation_down'."
            )

        # ── Check required top-level fields ─────────────────────────────
        required_numeric = ["distance", "duration", "elevation_up", "elevation_down"]
        for field in required_numeric:
            value = route_data.get(field)
            if value is None:
                return (
                    f"create_planned_tour requires the full plan_route() result "
                    f"(compact=False). Missing required numeric field '{field}'."
                )
            if isinstance(value, str):
                return (
                    f"create_planned_tour requires the full plan_route() result "
                    f"(compact=False). Expected numeric value for '{field}', "
                    f"got string '{value}'."
                )
            if not isinstance(value, (int, float)):
                return (
                    f"create_planned_tour requires the full plan_route() result "
                    f"(compact=False). Expected numeric value for '{field}', "
                    f"got {type(value).__name__}."
                )

        # ── Check path ──────────────────────────────────────────────────
        path = route_data.get("path")
        if path is None:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). Missing required field 'path'."
            )
        if not isinstance(path, list):
            return (
                f"create_planned_tour requires the full plan_route() result "
                f"(compact=False). Expected 'path' to be a list, "
                f"got {type(path).__name__}."
            )
        if len(path) == 0:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). 'path' is empty."
            )

        # ── Check segments ──────────────────────────────────────────────
        segments = route_data.get("segments")
        if segments is None:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). Missing required field 'segments'."
            )
        if not isinstance(segments, list):
            return (
                f"create_planned_tour requires the full plan_route() result "
                f"(compact=False). Expected 'segments' to be a list, "
                f"got {type(segments).__name__}."
            )
        if len(segments) == 0:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). 'segments' is empty."
            )

        # ── Check _embedded ─────────────────────────────────────────────
        embedded = route_data.get("_embedded")
        if embedded is None:
            return (
                "create_planned_tour requires the full plan_route() result "
                "(compact=False). Missing required field '_embedded'."
            )
        if not isinstance(embedded, dict):
            return (
                f"create_planned_tour requires the full plan_route() result "
                f"(compact=False). Expected '_embedded' to be a dict, "
                f"got {type(embedded).__name__}."
            )

        return None

    @mcp.tool()
    async def create_planned_tour(
        route_data: Optional[dict[str, Any]] = None,
        route_ref: str = "",
        name: str = "Tour planifie",
        sport: str = "hike",
    ) -> dict[str, Any]:
        """Creer un tour planifie sur Komoot a partir d'un resultat de plan_route ou import_gpx_route.

        Args:
            route_data: resultat JSON de plan_route() ou import_gpx_route().
            route_ref: reference de route depuis plan_route(compact=False).
                       Si fournie, route_data est ignore et la route est chargee depuis le cache interne.
            name: nom du tour planifie.
            sport: type de sport.
        """
        # ── Load route_data from cache if route_ref is provided ─────────
        if route_ref:
            async with _route_cache_lock:
                entry = _route_cache.get(route_ref)
                if entry is None:
                    return {
                        "status": "error",
                        "error": f"route_ref '{route_ref}' is invalid or expired",
                    }
                ts, cached_data = entry
                if time.time() - ts > _CACHE_TTL:
                    _route_cache.pop(route_ref, None)
                    return {
                        "status": "error",
                        "error": f"route_ref '{route_ref}' has expired",
                    }
                route_data = cached_data
                # Keep in cache until validation passes (so a retry is possible)
        elif route_data is None:
            return {
                "status": "error",
                "error": "Either route_data or route_ref is required",
            }

        # ── Validate route_data before sending to Komoot ───────────────
        error = _validate_route_data_for_save(route_data)
        if error is not None:
            return {"status": "error", "error": error}

        # ── Remove from cache now that validation passed ───────────────
        if route_ref:
            async with _route_cache_lock:
                _route_cache.pop(route_ref, None)

        payload = _prepare_planned_tour_payload(route_data, name, sport)
        return await client.request(
            "POST",
            f"https://api.komoot.de/v007/tours/",
            json_body=payload,
            params={"reroute": "true"},
        )

    @mcp.tool()
    async def analyze_komoot_tour(tour_id: int) -> dict[str, Any]:
        """Analyse detaillee d'une tour existante (singletrail, difficulte, surface, etc.).

        Args:
            tour_id: ID du tour Komoot existant.
        """
        try:
            tour = await client.get(
                f"/tours/{tour_id}",
                _embedded="way_types,surfaces,coordinates",
            )
        except KomootError as e:
            return {"tour_id": tour_id, "status": "error", "error": str(e)}

        if not isinstance(tour, dict):
            return {"tour_id": tour_id, "status": "error", "error": "Unexpected API response"}

        result: dict[str, Any] = {
            "tour_id": tour_id,
            "status": "success",
        }

        # ── Basic tour info ──────────────────────────────────────────────
        if tour.get("name"):
            result["name"] = tour["name"]
        if tour.get("sport"):
            result["sport"] = tour["sport"]
        if tour.get("distance"):
            result["distance_km"] = round(float(tour["distance"]) / 1000.0, 2)
        if "elevation_up" in tour:
            result["elevation_up_m"] = round(float(tour["elevation_up"]), 1)
        if "elevation_down" in tour:
            result["elevation_down_m"] = round(float(tour["elevation_down"]), 1)
        if tour.get("duration"):
            dur_s = int(float(tour["duration"]))
            h = dur_s // 3600
            m = (dur_s % 3600) // 60
            result["duration"] = f"{h}h {m}m" if h else f"{m}m"
        if "constitution" in tour:
            result["constitution"] = tour["constitution"]

        # ── Difficulty ────────────────────────────────────────────────────
        diff = tour.get("difficulty")
        if isinstance(diff, dict):
            if diff.get("grade"):
                result["difficulty"] = diff["grade"].upper()
            tech_raw = diff.get("explanation_technical", "")
            if tech_raw and isinstance(tech_raw, str) and "#" in tech_raw:
                result["technical_difficulty"] = tech_raw.split("#", 1)[1].upper()
            fit_raw = diff.get("explanation_fitness", "")
            if fit_raw and isinstance(fit_raw, str) and "#" in fit_raw:
                result["fitness_difficulty"] = fit_raw.split("#", 1)[1].upper()

        # ── Summary (surfaces, way_types) ─────────────────────────────────
        summary = tour.get("summary")
        if isinstance(summary, dict):
            surfaces = summary.get("surfaces", [])
            if isinstance(surfaces, list):
                result["surfaces"] = {
                    s["type"].replace("sm#", ""): round(float(s["amount"]) * 100, 1)
                    for s in surfaces
                    if isinstance(s, dict) and "type" in s and "amount" in s
                }
            wts = summary.get("way_types", [])
            if isinstance(wts, list):
                result["way_types"] = {
                    w["type"].replace("wt#", ""): round(float(w["amount"]) * 100, 1)
                    for w in wts
                    if isinstance(w, dict) and "type" in w and "amount" in w
                }

        # ── Tour information ──────────────────────────────────────────────
        ti = tour.get("tour_information", [])
        if isinstance(ti, list) and ti:
            info_list: list[dict[str, Any]] = []
            for item in ti:
                if isinstance(item, dict) and item.get("type"):
                    segs = item.get("segments", [])
                    info_list.append({
                        "type": str(item["type"]),
                        "segments": len(segs) if isinstance(segs, list) else 0,
                    })
            result["tour_information"] = info_list

        # ── Segments ──────────────────────────────────────────────────────
        segs = tour.get("segments", [])
        if isinstance(segs, list):
            routed = sum(1 for s in segs if isinstance(s, dict) and s.get("type") == "Routed")
            manual = sum(1 for s in segs if isinstance(s, dict) and s.get("type") == "Manual")
            result["segments"] = {"routed": routed, "manual": manual}

        # ── Singletrail analysis (trail_d1..trail_d5) ─────────────────────
        embedded = tour.get("_embedded", {})
        if not isinstance(embedded, dict):
            embedded = {}

        # Way-type items
        wt_container = embedded.get("way_types", {})
        wt_items: list[dict[str, Any]] = []
        if isinstance(wt_container, dict):
            wt_items = wt_container.get("items", [])
        elif isinstance(wt_container, list):
            wt_items = wt_container
        if not isinstance(wt_items, list):
            wt_items = []

        # Coordinates
        coord_container = embedded.get("coordinates", {})
        coord_items: list[dict[str, Any]] = []
        if isinstance(coord_container, dict):
            coord_items = coord_container.get("items", [])
        elif isinstance(coord_container, list):
            coord_items = coord_container
        if not isinstance(coord_items, list):
            coord_items = []

        total_km = result.get("distance_km", 0)
        result["singletrail"] = _compute_singletrail(wt_items, coord_items, total_km)

        # Coordinate count (not the actual list)
        if coord_items:
            result["matched_coordinates"] = len(coord_items)

        return result

    @mcp.tool()
    async def analyze_route_trail_coverage(
        route_coordinates: list[list[float]],
        discovered_segments: list[dict[str, Any]],
        discovered_clusters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze which discovered trail segments/clusters a route actually covers.

        Pure analysis: takes a route path + discovered segments/clusters and
        returns per-cluster coverage breakdown (covered/partial/uncovered)
        plus overall trail coverage stats.

        Use this to determine which trail clusters need improved coverage in
        the next routing attempt.

        Args:
            route_coordinates: [lat, lng] pairs of the planned route path.
            discovered_segments: trail segments from extract_trail_segments().
            discovered_clusters: trail clusters from cluster_trail_segments().

        Returns:
            trail_coverage: overall coverage stats (classified/unclassified).
            trail_clusters: per-cluster breakdown (covered/partial/uncovered).
            suggestions: suggested entry/exit waypoints for uncovered clusters.
        """
        coord_items: list[dict[str, Any]] = [
            {"lat": float(c[0]), "lng": float(c[1])} for c in route_coordinates
        ]

        coverage = _compute_trail_coverage(coord_items, discovered_segments)
        analyzed = analyze_cluster_coverage(discovered_clusters, coord_items)
        cluster_info = _compact_cluster_info(analyzed)
        suggestions = suggest_uncovered_waypoints(
            discovered_clusters, analyzed, max_new=2,
        )

        return {
            "trail_coverage": coverage,
            "trail_clusters": cluster_info,
            "suggestions": suggestions,
        }

    @mcp.tool()
    async def create_planned_tour_from_gpx(
        file_path: str,
        sport: str = "mtb",
        name: str = "GPX import",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Import a local GPX file and prepare a planned Komoot tour.

        Workflow:
          1. Upload GPX  →  POST /routing/import/files/
          2. Match route →  POST /routing/import/tour
          3. Prepare payload for POST /v007/tours/?reroute=true

        The write API call is ONLY executed when dry_run=False.
        Default: dry_run=True (prepares and inspects, never writes).

        Args:
            file_path: path to the local GPX file.
            sport: sport type (e.g. mtb, hike, roadbike, touringbicycle).
            name: name for the planned tour.
            dry_run: if True (default), prepare payload but do NOT create the tour.
        """
        # ── Step 1: Read GPX file ────────────────────────────────────────
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                gpx_content = f.read()
        except FileNotFoundError:
            return {"status": "error", "error": f"File not found: {file_path}"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to read GPX file: {str(e)}"}

        # ── Step 2: Upload GPX ───────────────────────────────────────────
        import_result = await client.post_www(
            "/routing/import/files/",
            data=gpx_content.encode("utf-8"),
            headers={"Content-Type": "application/gpx+xml"},
            params={"data_type": "gpx"},
        )
        if not isinstance(import_result, dict):
            return {"status": "error", "raw": str(import_result)}

        try:
            item = import_result["_embedded"]["items"][0]
        except (KeyError, IndexError, TypeError):
            return {
                "status": "error",
                "error": "Komoot returned no import item",
                "response": import_result,
            }

        # ── Step 3: Match against routing network ────────────────────────
        matched = await client.post_www(
            "/routing/import/tour",
            json_body=item,
            params={
                "sport": sport,
                "_embedded": "way_types,surfaces,directions,coordinates",
            },
        )
        if not isinstance(matched, dict):
            return {"status": "error", "raw": str(matched)}

        # ── Step 4: Extract matched tour data ────────────────────────────
        matched_tour = matched.get("_embedded", {}).get("matched", {})
        if not matched_tour:
            return {"status": "error", "error": "No matched tour in response", "response": matched}

        # ── Step 5: Prepare payload ──────────────────────────────────────
        payload = _prepare_planned_tour_payload(matched, name, sport)

        # ── Step 6: Build result ─────────────────────────────────────────
        result: dict[str, Any] = {
            "status": "prepared" if dry_run else "pending",
            "file_path": file_path,
            "name": name,
            "sport": sport,
        }

        # Extract basic info for display
        if matched_tour.get("distance"):
            result["distance_km"] = round(float(matched_tour["distance"]) / 1000.0, 2)
        if "elevation_up" in matched_tour:
            result["elevation_up_m"] = round(float(matched_tour["elevation_up"]), 1)
        if "elevation_down" in matched_tour:
            result["elevation_down_m"] = round(float(matched_tour["elevation_down"]), 1)
        if matched_tour.get("duration"):
            dur_s = int(float(matched_tour["duration"]))
            h = dur_s // 3600
            m = (dur_s % 3600) // 60
            result["duration"] = f"{h}h {m}m" if h else f"{m}m"

        diff = matched_tour.get("difficulty")
        if isinstance(diff, dict):
            if diff.get("grade"):
                result["difficulty"] = diff["grade"]
            tech_raw = diff.get("explanation_technical", "")
            if tech_raw and isinstance(tech_raw, str) and "#" in tech_raw:
                result["technical_difficulty"] = tech_raw.split("#", 1)[1].upper()
            fit_raw = diff.get("explanation_fitness", "")
            if fit_raw and isinstance(fit_raw, str) and "#" in fit_raw:
                result["fitness_difficulty"] = fit_raw.split("#", 1)[1].upper()

        if dry_run:
            result["dry_run"] = True
            result["message"] = "Payload prepared. Set dry_run=False to create the tour."
            result["payload_structure"] = {
                "endpoint": "POST https://api.komoot.de/v007/tours/?reroute=true",
                "keys": list(payload.keys()),
                "name": payload.get("name"),
                "sport": payload.get("sport"),
                "segments_count": len(payload.get("segments", [])),
            }
        else:
            # ── Step 7: Create the tour (writes to Komoot!) ──────────────
            creation_result = await client.request(
                "POST",
                "https://api.komoot.de/v007/tours/",
                json_body=payload,
                params={"reroute": "true"},
            )
            if isinstance(creation_result, dict):
                tour_id = creation_result.get("id") or creation_result.get("tour_id")
                if tour_id:
                    result["status"] = "success"
                    result["tour_id"] = tour_id
                    result["komoot_tour_url"] = f"https://www.komoot.com/tour/{tour_id}"
                else:
                    result["status"] = "created"
                    result["response"] = str(creation_result)[:500]
            else:
                result["status"] = "error"
                result["response"] = str(creation_result)[:500]

        return result
