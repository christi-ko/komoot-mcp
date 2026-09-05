from __future__ import annotations

import asyncio

from komoot_mcp.tools.routing import (
    ROUTE_REDUNDANCY_OVERLAP_PCT,
    _build_route_coordinates,
    _plan_trail_areas,
    _ordered_area_waypoints,
    _classify_trail_direction,
    ROUTE_SIMILARITY_WARNING_PCT,
    _geometry_similarity_pct,
    _route_geometry_fingerprint,
    _run_adaptive_variant_controller,
    _adaptive_has_meaningful_next_strategy,
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
    assert all(call[0] == initial[0] and call[-1] == initial[-1] for call in calls)
    assert all(call[1:-1] != calls[0][1:-1] or call == calls[0] for call in calls)
    assert result["total_attempts"] == len(calls)
    assert all("variant_id" in a and "candidate_status" in a for a in result["feedback_decisions"])


def test_roundtrip_coordinates_put_waypoints_before_return_to_start() -> None:
    start = [[47.0, 9.0], [47.0, 9.0]]
    assert _build_route_coordinates(start, [[47.1, 9.1]], "roundtrip") == [
        [47.0, 9.0], [47.1, 9.1], [47.0, 9.0]
    ]


def test_trail_areas_group_nearby_clusters_and_preserve_cluster_order() -> None:
    clusters = [
        {"start": {"lat": 47.000, "lng": 9.000}, "end": {"lat": 47.001, "lng": 9.001}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.002, "lng": 9.001}, "end": {"lat": 47.003, "lng": 9.002}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.100, "lng": 9.100}, "end": {"lat": 47.101, "lng": 9.101}, "total_length_km": 1.0, "trail_categories": ["classified"]},
    ]
    areas = _plan_trail_areas(clusters, proximity_km=0.5)
    assert len(areas) == 2
    assert areas[0]["cluster_indices"] == [0, 1]
    assert _ordered_area_waypoints(areas[0], clusters) == [[47.0, 9.0], [47.002, 9.001]]


def test_steep_trail_prefers_downhill_direction_and_flat_is_approach() -> None:
    assert _classify_trail_direction(100.0, 80.0) == "downhill_start_to_end"
    assert _classify_trail_direction(100.0, 99.0) == "approach_or_bidirectional"


def test_waytype_connection_audit_prefers_cycleway_and_flags_major_road() -> None:
    from komoot_mcp.tools.routing import _connection_waytype_audit
    audit = _connection_waytype_audit({"cycleway": 10.0, "minor_road": 20.0, "highway": 5.0})
    assert audit["cycleway_present"] is True
    assert audit["local_road_present"] is True
    assert audit["major_road_present"] is True
    assert audit["major_road_avoidable"] is False


def test_planning_audit_records_closed_route_and_connection_policy() -> None:
    from komoot_mcp.tools.routing import _connection_waytype_audit
    assert _connection_waytype_audit({"cycleway": 10.0, "minor_road": 20.0})["major_road_avoidable"] is True
    clusters = [{"start": {"lat": 47.1, "lng": 9.1}, "end": {"lat": 47.101, "lng": 9.101}, "total_length_km": 1.0, "trail_categories": ["classified"]}]
    async def fake(coords, **kwargs):
        return {"distance_km": 36.0, "elevation_up_m": 500.0, "technical_difficulty": "T2", "route_quality_metrics": {"classified_trail_km": 1.0, "surface_offroad_km": 10.0}, "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": 0}}, "trail_clusters": {"covered": [], "partial": [], "uncovered": [{"cluster_index": 0, "coverage_percentage": 0, "trail_categories": ["classified"], "total_length_km": 1.0}]}, "_route_geometry_fingerprint": _route_geometry_fingerprint(_line([(47.0, 9.0), (47.1, 9.1)]))}
    result = asyncio.run(_run_adaptive_variant_controller([[47.0, 9.0], [47.0, 9.0]], "mtb", [], clusters, fake, {}))
    audit = result["iterations"][0]["planning_audit"]
    assert audit["roundtrip_closed"] is True
    assert audit["connection_policy"][0] == "cycleway"


def test_area_chain_drives_waypoints_and_entry_exit_plan() -> None:
    clusters = [
        {"start": {"lat": 47.01, "lng": 9.01, "alt": 300}, "end": {"lat": 47.02, "lng": 9.02, "alt": 200}, "total_length_km": 1.0, "trail_categories": ["classified"], "steepness_status": "steep"},
        {"start": {"lat": 47.011, "lng": 9.011, "alt": 290}, "end": {"lat": 47.021, "lng": 9.021, "alt": 190}, "total_length_km": 1.0, "trail_categories": ["classified"], "steepness_status": "steep"},
        {"start": {"lat": 47.10, "lng": 9.10, "alt": 100}, "end": {"lat": 47.11, "lng": 9.11, "alt": 95}, "total_length_km": 1.0, "trail_categories": ["classified"]},
    ]
    areas = _plan_trail_areas(clusters, proximity_km=0.5)
    plan = __import__("komoot_mcp.tools.routing", fromlist=["_build_adaptive_variant"])._build_adaptive_variant(
        "classified_trail_coverage",
        clusters,
        [{"cluster_index": i, "coverage_percentage": 0, "total_length_km": 1.0, "trail_categories": ["classified"]} for i in range(3)],
        set(), set(), [], trail_areas=areas,
    )
    assert plan["selected_cluster_ids"][:2] == [0, 1]
    assert plan["waypoints"] == [[47.01, 9.01], [47.02, 9.02], [47.011, 9.011], [47.021, 9.021]]
    assert plan["area_sequence"] == [0]
    assert plan["entry_exit_plan"][0]["role"] == "trail_area"


def test_missing_elevation_is_unknown_and_audit_has_explicit_fields() -> None:
    assert _classify_trail_direction(None, None) == "direction_unknown"
    required = {"route_mode", "roundtrip_closed", "trail_area_sequence", "cluster_sequence", "entry_exit_plan", "direction_status", "steepness_status", "difficulty_status", "connection_segments", "connection_way_types", "backtracking_detected", "route_overlap", "classified_trail_km", "surface_offroad_km", "discovery_trail_coverage", "distance", "elevation", "T/C", "asphalt", "geometry_similarity"}
    from komoot_mcp.tools.routing import _planning_audit
    assert required <= set(_planning_audit({}, [[47.0, 9.0], [47.0, 9.0]], [], [], [], []).keys())


def test_a_to_b_coordinates_keep_final_destination_and_no_closure() -> None:
    assert _build_route_coordinates([[47.0, 9.0], [47.5, 9.5]], [[47.1, 9.1]], "point_to_point") == [[47.0, 9.0], [47.1, 9.1], [47.5, 9.5]]


def test_distance_out_of_range_with_full_coverage_keeps_searching() -> None:
    route = {"distance_km": 16.0, "trail_clusters": {"covered": [], "partial": [], "uncovered": []}}
    assert _adaptive_has_meaningful_next_strategy(route, set(), {"min_distance_km": 35, "max_distance_km": 40}) is True
    assert _adaptive_has_meaningful_next_strategy(route, {"classified_trail_coverage", "spatial_trail_diversity", "distance_aware", "balanced"}, {"min_distance_km": 35, "max_distance_km": 40}) is False


def test_distance_strategy_can_build_variant_when_previous_route_covered_all_clusters() -> None:
    from komoot_mcp.tools.routing import _build_adaptive_variant

    clusters = [
        {"start": {"lat": 47.01, "lng": 9.01}, "end": {"lat": 47.02, "lng": 9.02}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.10, "lng": 9.10}, "end": {"lat": 47.11, "lng": 9.11}, "total_length_km": 2.0, "trail_categories": ["classified"]},
    ]
    analyzed = [
        {"cluster_index": 0, "coverage_percentage": 100, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"cluster_index": 1, "coverage_percentage": 100, "total_length_km": 2.0, "trail_categories": ["classified"]},
    ]
    plan = _build_adaptive_variant(
        "distance_aware", clusters, analyzed, set(), set(),
        [[47.0, 9.0], [47.0, 9.0]], trail_areas=_plan_trail_areas(clusters),
    )
    assert plan is not None
    assert plan["selected_cluster_ids"] == [0, 1]




def test_backtracking_is_detected_from_repeated_route_points() -> None:
    from komoot_mcp.tools.routing import _detect_backtracking
    assert _detect_backtracking([[47.0, 9.0], [47.1, 9.1], [47.0, 9.0]]) is True


def test_area_grouping_uses_transitive_cluster_graph_components() -> None:
    clusters = [
        {"start": {"lat": 47.000, "lng": 9.000}, "end": {"lat": 47.000, "lng": 9.000}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.010, "lng": 9.000}, "end": {"lat": 47.010, "lng": 9.000}, "total_length_km": 2.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.020, "lng": 9.000}, "end": {"lat": 47.020, "lng": 9.000}, "total_length_km": 3.0, "trail_categories": ["classified"]},
    ]
    areas = _plan_trail_areas(clusters, proximity_km=1.2)
    assert len(areas) == 1
    assert areas[0]["cluster_indices"] == [0, 1, 2]
    assert areas[0]["classified_trail_km"] == 6.0


def test_major_road_avoidability_is_unknown_without_comparison_route() -> None:
    from komoot_mcp.tools.routing import _planning_audit
    audit = _planning_audit({"way_types": {"primary": {"percentage": 5}}}, [[47.0, 9.0], [47.0, 9.0]], [], [], [], [])
    assert audit["major_road_present"] is True
    assert audit["major_road_avoidable"] == "unknown"


def test_major_road_avoidability_is_unknown_before_route_comparison() -> None:
    from komoot_mcp.tools.routing import _connection_waytype_audit
    audit = _connection_waytype_audit({"primary": {"percentage": 5}}, comparison_available=False)
    assert audit["major_road_avoidable"] == "unknown"


def test_major_road_avoidability_becomes_comparable_after_alternative() -> None:
    from komoot_mcp.tools.routing import _planning_audit
    audit = _planning_audit({"way_types": {"primary": {"percentage": 5}}, "comparison_route_available": True}, [[47.0, 9.0], [47.0, 9.0]], [], [], [], [])
    assert audit["major_road_present"] is True
    assert audit["major_road_avoidable"] is False




def test_area_chain_contains_entry_exit_and_connection_plan() -> None:
    clusters = [
        {"start": {"lat": 47.01, "lng": 9.01}, "end": {"lat": 47.02, "lng": 9.02}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.10, "lng": 9.10}, "end": {"lat": 47.11, "lng": 9.11}, "total_length_km": 1.0, "trail_categories": ["classified"]},
    ]
    areas = _plan_trail_areas(clusters, proximity_km=0.5)
    from komoot_mcp.tools.routing import _build_area_chain
    chain = _build_area_chain(areas, clusters, [0, 1])
    assert chain["area_sequence"] == [0, 1]
    assert chain["entry_exit_plan"][0]["entry_point"] == clusters[0]["start"]
    assert chain["entry_exit_plan"][0]["exit_point"] == clusters[0]["end"]
    assert chain["connection_plan"][0]["from_area"] == 0
    assert chain["connection_plan"][0]["to_area"] == 1


def test_direction_aware_waypoints_enter_downhill_trail_at_lower_end() -> None:
    from komoot_mcp.tools.routing import _build_adaptive_variant

    clusters = [{
        "start": {"lat": 47.01, "lng": 9.01, "alt": 300},
        "end": {"lat": 47.02, "lng": 9.02, "alt": 200},
        "total_length_km": 1.0,
        "trail_categories": ["classified"],
    }]
    plan = _build_adaptive_variant(
        "classified_trail_coverage", clusters,
        [{"cluster_index": 0, "coverage_percentage": 0,
          "total_length_km": 1.0, "trail_categories": ["classified"]}],
        set(), set(), [], trail_areas=_plan_trail_areas(clusters),
    )
    assert plan is not None
    assert plan["waypoints"][:2] == [[47.01, 9.01], [47.02, 9.02]]


def test_reverse_downhill_direction_uses_end_as_entry() -> None:
    from komoot_mcp.tools.routing import _build_adaptive_variant

    clusters = [{
        "start": {"lat": 47.01, "lng": 9.01, "alt": 200},
        "end": {"lat": 47.02, "lng": 9.02, "alt": 300},
        "total_length_km": 1.0,
        "trail_categories": ["classified"],
    }]
    plan = _build_adaptive_variant(
        "classified_trail_coverage", clusters,
        [{"cluster_index": 0, "coverage_percentage": 0,
          "total_length_km": 1.0, "trail_categories": ["classified"]}],
        set(), set(), [], trail_areas=_plan_trail_areas(clusters),
    )
    assert plan is not None
    assert plan["waypoints"][:2] == [[47.02, 9.02], [47.01, 9.01]]


def test_directed_area_chain_starts_with_area_nearest_to_route_start() -> None:
    from komoot_mcp.tools.routing import _build_adaptive_variant

    clusters = [
        {"start": {"lat": 47.20, "lng": 9.20}, "end": {"lat": 47.21, "lng": 9.21}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.01, "lng": 9.01}, "end": {"lat": 47.02, "lng": 9.02}, "total_length_km": 1.0, "trail_categories": ["classified"]},
        {"start": {"lat": 47.10, "lng": 9.10}, "end": {"lat": 47.11, "lng": 9.11}, "total_length_km": 1.0, "trail_categories": ["classified"]},
    ]
    analyzed = [{"cluster_index": i, "coverage_percentage": 0,
                 "total_length_km": 1.0, "trail_categories": ["classified"]}
                for i in range(3)]
    plan = _build_adaptive_variant(
        "classified_trail_coverage", clusters, analyzed, set(), set(),
        [[47.0, 9.0], [47.0, 9.0]], trail_areas=_plan_trail_areas(clusters),
    )
    assert plan is not None
    assert plan["selected_cluster_ids"][0] == 1
    assert plan["area_sequence"][0] == 1
