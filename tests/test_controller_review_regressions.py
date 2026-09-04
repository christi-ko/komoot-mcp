from komoot_mcp.tools.routing import _adaptive_route_better, _is_qualitatively_better


def _route(distance, trail, constraints=None, asphalt=50.0):
    return {
        "distance_km": distance,
        "singletrail": {"singletrail_total_km": trail},
        "route_quality_metrics": {"classified_trail_km": trail, "surface_offroad_km": 1.0, "asphalt_percentage": asphalt},
        "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": 0}},
        "trail_clusters": {"uncovered": [], "partial": []},
        "surfaces": {"asphalt": asphalt},
        "selection_constraints": constraints or {},
    }


def test_unbounded_distance_does_not_beat_lower_asphalt_after_trail_tie():
    shorter = _route(20, 2.0, asphalt=60.0)
    longer = _route(35, 2.0, asphalt=40.0)
    assert _adaptive_route_better(longer, shorter, {})


def test_single_distance_bound_recognizes_feasible_route():
    below = _route(5, 2.0, {"min_distance_km": 10})
    feasible = _route(15, 2.0, {"min_distance_km": 10})
    assert _adaptive_route_better(feasible, below, {"min_distance_km": 10})
    above = _route(50, 2.0, {"max_distance_km": 40})
    feasible_max = _route(35, 2.0, {"max_distance_km": 40})
    assert _adaptive_route_better(feasible_max, above, {"max_distance_km": 40})


def test_controller_transition_audit_is_not_always_empty():
    # The public audit contract requires a transition record between candidates.
    from komoot_mcp.tools.routing import _build_feedback_transition

    transition = _build_feedback_transition(
        {"variant_id": "variant_1", "distance_km": 20},
        {"variant_id": "variant_2", "distance_km": 35},
    )
    assert transition["from_variant_id"] == "variant_1"
    assert transition["to_variant_id"] == "variant_2"
    assert transition["distance_delta_km"] == 15


def test_routed_classified_trail_outranks_discovery_coverage():
    def route(trail, coverage):
        return {
            "singletrail": {"singletrail_total_km": trail},
            "route_quality_metrics": {"classified_trail_km": trail, "surface_offroad_km": 10.0},
            "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": coverage}},
            "trail_clusters": {"uncovered": [], "partial": []},
            "selection_constraints": {"min_distance_km": 35, "max_distance_km": 40},
            "distance_km": 38.0,
        }
    assert _is_qualitatively_better(route(4.82, 38.8), route(3.84, 41.4))


def test_offroad_breaks_a_classified_trail_tie():
    def route(offroad):
        return {
            "singletrail": {"singletrail_total_km": 3.0},
            "route_quality_metrics": {"classified_trail_km": 3.0, "surface_offroad_km": offroad},
            "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": 0}},
            "trail_clusters": {"uncovered": [], "partial": []},
            "selection_constraints": {"min_distance_km": 35, "max_distance_km": 40},
            "distance_km": 38.0,
        }
    assert _is_qualitatively_better(route(15.0), route(10.0))


def test_distance_strategy_is_inserted_once_without_skipping_coverage_strategy():
    # The controller's strategy queue must retain each strategy exactly once.
    from komoot_mcp.tools.routing import _adaptive_strategy_order

    order = _adaptive_strategy_order(_route(25, 2.0, {"min_distance_km": 35, "max_distance_km": 40}), set())
    assert order[:2] == ["distance_aware", "classified_trail_coverage"]
    assert len(order) == len(set(order))


def test_controller_transition_audit_is_not_always_empty_duplicate():
    # The public audit contract requires a transition record between candidates.
    from komoot_mcp.tools.routing import _build_feedback_transition

    transition = _build_feedback_transition(
        {"variant_id": "variant_1", "distance_km": 20},
        {"variant_id": "variant_2", "distance_km": 35},
    )
    assert transition["from_variant_id"] == "variant_1"
    assert transition["to_variant_id"] == "variant_2"
    assert transition["distance_delta_km"] == 15
