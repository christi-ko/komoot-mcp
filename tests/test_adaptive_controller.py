from __future__ import annotations

import asyncio

from komoot_mcp.tools.routing import (
    ROUTE_REDUNDANCY_OVERLAP_PCT,
    ROUTE_SIMILARITY_WARNING_PCT,
    _geometry_similarity_pct,
    _route_geometry_fingerprint,
    _run_adaptive_variant_controller,
)


def _line(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"lat": lat, "lng": lng} for lat, lng in points]


def test_geometry_identical_and_reversed_are_identical() -> None:
    a = _route_geometry_fingerprint(_line([(47.0, 9.0), (47.01, 9.0), (47.02, 9.0)]))
    b = _route_geometry_fingerprint(_line([(47.02, 9.0), (47.01, 9.0), (47.0, 9.0)]))
    assert _geometry_similarity_pct(a, b) == 100.0


def test_common_approach_with_different_branch_is_not_redundant() -> None:
    a = _route_geometry_fingerprint(_line([(47.0, 9.0), (47.01, 9.0), (47.02, 9.0), (47.03, 9.0)]))
    b = _route_geometry_fingerprint(_line([(47.0, 9.0), (47.01, 9.0), (47.02, 9.0), (47.02, 9.02)]))
    similarity = _geometry_similarity_pct(a, b)
    assert similarity < ROUTE_SIMILARITY_WARNING_PCT
    assert similarity < ROUTE_REDUNDANCY_OVERLAP_PCT


def test_controller_never_exceeds_five_calls_and_keeps_waypoints_independent() -> None:
    clusters = [
        {"start": {"lat": 47.1 + i * .01, "lng": 9.1}, "end": {"lat": 47.101 + i * .01, "lng": 9.101}, "total_length_km": 1.0, "trail_categories": ["classified"]}
        for i in range(8)
    ]
    segments = [{"start": c["start"], "end": c["end"], "length_km": 1.0, "trail_category": "classified"} for c in clusters]
    calls: list[list[list[float]]] = []
    async def fake(coords, **kwargs):
        calls.append([list(c) for c in coords])
        return {
            "distance_km": 36.0, "elevation_up_m": 500.0, "technical_difficulty": "T2",
            "route_quality_metrics": {"classified_trail_km": 1.0, "surface_offroad_km": 10.0, "asphalt_percentage": 20.0},
            "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": 0}},
            "trail_clusters": {"covered": [], "partial": [], "uncovered": [{"cluster_index": i, "coverage_percentage": 0, "trail_categories": ["classified"], "total_length_km": 1.0} for i in range(8)]},
            "_route_geometry_fingerprint": _route_geometry_fingerprint(_line([(47.0, 9.0), (47.01, 9.0), (47.02, 9.0 + len(calls) * .001)])),
        }
    initial = [[47.0, 9.0], [47.0, 9.1]]
    result = asyncio.run(_run_adaptive_variant_controller(initial, "mtb", segments, clusters, fake, {"max_elevation_up_m": 600, "technical_max": "T2", "min_distance_km": 35, "max_distance_km": 40}))
    assert len(calls) <= 5
    assert all(call[:2] == initial for call in calls)
    assert result["total_attempts"] == len(calls)
    assert all("variant_id" in a and "candidate_status" in a for a in result["feedback_decisions"])
