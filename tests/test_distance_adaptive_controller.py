from __future__ import annotations

from komoot_mcp.tools.routing import _is_qualitatively_better


def _route(distance: float, trail: float, coverage: float = 0.0) -> dict:
    return {
        "distance_km": distance,
        "singletrail": {"singletrail_total_km": trail},
        "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": coverage}},
        "trail_clusters": {"uncovered": [], "partial": []},
        "surfaces": {"asphalt": 50.0},
        "selection_constraints": {"min_distance_km": 35, "max_distance_km": 40},
    }


def test_in_range_wins_when_trail_quality_is_comparable() -> None:
    assert _is_qualitatively_better(_route(37, 2.0), _route(25, 2.0))


def test_materially_better_trail_wins_outside_distance_range() -> None:
    assert _is_qualitatively_better(_route(25, 4.0), _route(37, 2.0))


def test_distance_strategy_is_not_stopped_by_trail_stagnation() -> None:
    # Regression contract for the controller: a best route below min_distance
    # must leave the distance-aware strategy eligible before the five-call cap.
    from komoot_mcp.tools.routing import _distance_strategy_needed

    assert _distance_strategy_needed(_route(25, 2.0))
    assert not _distance_strategy_needed(_route(37, 2.0))


def test_excluded_clusters_have_a_reason() -> None:
    reason = {"8": "previously selected in another variant; excluded to diversify cluster combinations"}
    audit = {"excluded_cluster_ids": [8], "excluded_cluster_reason": reason}
    assert audit["excluded_cluster_ids"]
    assert audit["excluded_cluster_reason"]
    assert set(map(str, audit["excluded_cluster_ids"])) == set(audit["excluded_cluster_reason"])
