"""MCP tools for Komoot route planning and import.

Ces endpoints passent par www.komoot.com/api (pas api.komoot.de).
Inspires du projet export-komoot (Go) de pieterclaerhout.
"""

from __future__ import annotations

import asyncio
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

        min_distance = constraints.get("min_distance_km")
        max_distance = constraints.get("max_distance_km")
        distance = route.get("distance_km")
        if isinstance(distance, (int, float)):
            if isinstance(min_distance, (int, float)) and distance < min_distance:
                violations += 1
                excess += min_distance - distance
            if isinstance(max_distance, (int, float)) and distance > max_distance:
                violations += 1
                excess += distance - max_distance

        return violations, excess

    a_hard = _constraint_status(a)
    b_hard = _constraint_status(b)
    if a_hard != b_hard:
        return a_hard < b_hard

    a_cov = a.get("trail_coverage") or {}
    b_cov = b.get("trail_coverage") or {}
    a_cls = a_cov.get("classified") or {}
    b_cls = b_cov.get("classified") or {}

    # B. More desired classified clusters fully covered.
    a_cc = _g(a_cls, "covered", 0)
    b_cc = _g(b_cls, "covered", 0)
    if a_cc != b_cc:
        return a_cc > b_cc

    # C. Higher classified trail coverage.
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

    # G. Classified singletrail / trail quality.
    a_st = _g(a.get("singletrail") or {}, "singletrail_total_km", 0)
    b_st = _g(b.get("singletrail") or {}, "singletrail_total_km", 0)
    if abs(a_st - b_st) > 0.5:
        return a_st > b_st

    # H. Distance is the final route-size criterion. Elevation is not an
    # optimization target without an explicit constraint.
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

    return summary


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
        suggestions = suggest_uncovered_waypoints(
            discovered_clusters,
            analyzed,
            current_waypoints=current_coords,
            max_new=2,
        )

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

    # Singletrail (from _embedded.way_types.items + coordinates)
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
            feedback_loop: si True, lance une optimisation iterative avec jusqu'a
                           3 tentatives de routage guidees par trail_coverage.
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
            return await _run_feedback_loop(
                coordinates, sport,
                discovered_segments or [],
                discovered_clusters or [],
                plan_route,
                max_iterations=3,
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
