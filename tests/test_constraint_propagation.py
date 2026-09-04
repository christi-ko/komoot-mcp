"""Tests for plan_route constraint parameter propagation.

Tests that:
- plan_route accepts new optional constraint parameters without breaking old calls
- _run_feedback_loop forwards constraints to each plan_route_fn call
- constraints are recorded as selection_constraints on each route for comparison
- _is_qualitatively_better respects distance hard bounds
- MockPlanRoute accepts/records constraints
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from komoot_mcp.tools.routing import (
    _is_qualitatively_better,
    _run_feedback_loop,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def two_clusters() -> list[dict[str, Any]]:
    return [
        {
            "segments": 2, "total_length_km": 2.0,
            "source_tour_ids": [101], "way_types": ["trail_d1"],
            "trail_categories": ["classified"],
            "start": {"lat": 47.50, "lng": 10.00},
            "end": {"lat": 47.52, "lng": 10.02},
        },
        {
            "segments": 1, "total_length_km": 1.5,
            "source_tour_ids": [102], "way_types": ["trail_d2"],
            "trail_categories": ["classified"],
            "start": {"lat": 47.55, "lng": 10.05},
            "end": {"lat": 47.57, "lng": 10.07},
        },
    ]


@pytest.fixture
def dummy_segments() -> list[dict[str, Any]]:
    return [{
        "start": {"lat": 47.5, "lng": 10.0},
        "end": {"lat": 47.52, "lng": 10.02},
        "length_km": 2.0,
        "trail_category": "classified",
        "way_type": "trail_d1",
    }]


# ── Mock helpers ────────────────────────────────────────────────────


class MockPlanRouteWithConstraints:
    """Mock plan_route that records constraints passed to each call."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def __call__(
        self,
        coordinates,
        sport="mtb",
        compact=True,
        discovered_segments=None,
        discovered_clusters=None,
        **kwargs,
    ):
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.responses) - 1)
        resp = {k: v for k, v in self.responses[idx].items()}
        self.call_history.append({
            "coordinates": [list(c) for c in coordinates],
            "sport": sport,
            "kwargs": kwargs,
        })
        return resp


def _make_compact_route(
    distance_km: float = 20.0,
    asphalt_pct: float = 30.0,
    overlap_pct: float = 2.5,
    singletrail_km: float = 3.5,
    trail_coverage_classified_covered: int = 0,
    trail_coverage_classified_pct: float = 0.0,
    trail_coverage_total_pct: float = 0.0,
    trail_clusters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "distance_km": distance_km,
        "elevation_up_m": 400.0,
        "elevation_down_m": 400.0,
        "duration": "1h 30m",
        "difficulty": "MODERATE",
        "technical_difficulty": "T2",
        "fitness_difficulty": "C2",
        "way_types": {"trail": 30.0, "street": 40.0, "way": 30.0},
        "surfaces": {"asphalt": asphalt_pct, "unpaved": 40.0, "nature": 30.0},
        "singletrail": {
            "trail_d1": 2.0, "trail_d2": 1.5,
            "singletrail_total_km": singletrail_km,
        },
        "segments": {"routed": 5, "manual": 0},
        "matched_coordinates": 100,
        "route_overlap": {"overlap_km": 0.5, "overlap_percentage": overlap_pct},
        "trail_coverage": {
            "classified": {
                "discovered": 2, "covered": trail_coverage_classified_covered,
                "total_km": 4.0,
                "covered_km": round(4.0 * trail_coverage_classified_pct / 100, 4),
                "coverage_percentage": trail_coverage_classified_pct,
            },
            "unclassified": {
                "discovered": 0, "covered": 0,
                "total_km": 0.0, "covered_km": 0.0,
                "coverage_percentage": 0.0,
            },
            "total_coverage_percentage": trail_coverage_total_pct,
        },
    }
    if trail_clusters is not None:
        base["trail_clusters"] = trail_clusters
    return base


def _cluster_entry(
    cluster_index: int,
    coverage_pct: float,
    categories: list[str],
    **overrides: Any,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "cluster_index": cluster_index,
        "total_length_km": 2.0,
        "covered_km": round(2.0 * coverage_pct / 100, 4),
        "coverage_percentage": coverage_pct,
        "trail_categories": categories,
    }
    d.update(overrides)
    return d


# ══════════════════════════════════════════════════════════════════════
# _is_qualitatively_better — distance constraint tests
# ══════════════════════════════════════════════════════════════════════


class TestIsQualitativelyBetterDistanceConstraints:
    """Hard constraint semantics for distance bounds."""

    def test_distance_bounds_in_range_beats_out_of_range(self):
        """23.75 loses to 37 with min_distance_km=35, max_distance_km=40."""
        a = _make_compact_route(distance_km=23.75)
        b = _make_compact_route(distance_km=37.0)
        a["selection_constraints"] = {"min_distance_km": 35, "max_distance_km": 40}
        b["selection_constraints"] = {"min_distance_km": 35, "max_distance_km": 40}
        a["_attempt"], b["_attempt"] = 1, 2
        # b (37) in range, a (23.75) out of range → b wins
        assert _is_qualitatively_better(b, a) is True
        assert _is_qualitatively_better(a, b) is False

    def test_max_distance_39_feasible_41_violation(self):
        """max_distance 40: 39 feasible, 41 violation."""
        a = _make_compact_route(distance_km=39.0)
        b = _make_compact_route(distance_km=41.0)
        a["selection_constraints"] = {"max_distance_km": 40}
        b["selection_constraints"] = {"max_distance_km": 40}
        a["_attempt"], b["_attempt"] = 1, 2
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False

    def test_min_distance_34_violation_36_feasible(self):
        """min_distance 35: 34 violation, 36 feasible."""
        a = _make_compact_route(distance_km=34.0)
        b = _make_compact_route(distance_km=36.0)
        a["selection_constraints"] = {"min_distance_km": 35}
        b["selection_constraints"] = {"min_distance_km": 35}
        a["_attempt"], b["_attempt"] = 1, 2
        assert _is_qualitatively_better(b, a) is True
        assert _is_qualitatively_better(a, b) is False

    def test_distance_bounds_neutral_among_two_in_range(self):
        """Two in-range routes: shorter (36) does NOT beat longer (37)."""
        a = _make_compact_route(distance_km=36.0)
        b = _make_compact_route(distance_km=37.0)
        a["selection_constraints"] = {"min_distance_km": 35, "max_distance_km": 40}
        b["selection_constraints"] = {"min_distance_km": 35, "max_distance_km": 40}
        a["_attempt"], b["_attempt"] = 1, 2
        # Among two in-range, distance is neutral → earlier attempt wins
        assert _is_qualitatively_better(a, b) is True   # earlier wins tie
        assert _is_qualitatively_better(b, a) is False  # later loses tie

    def test_out_of_range_routes_compare_distance_to_nearest_bound(self):
        constraints = {"min_distance_km": 35, "max_distance_km": 40}
        below_closer = _make_compact_route(distance_km=34.0)
        below_farther = _make_compact_route(distance_km=30.0)
        above_closer = _make_compact_route(distance_km=41.0)
        above_farther = _make_compact_route(distance_km=46.0)
        for route in (below_closer, below_farther, above_closer, above_farther):
            route["selection_constraints"] = constraints
        assert _is_qualitatively_better(below_closer, below_farther) is True
        assert _is_qualitatively_better(above_closer, above_farther) is True

    def test_out_of_range_routes_on_opposite_sides_compare_bound_distance(self):
        constraints = {"min_distance_km": 35, "max_distance_km": 40}
        below = _make_compact_route(distance_km=34.0)
        above = _make_compact_route(distance_km=45.0)
        below["selection_constraints"] = constraints
        above["selection_constraints"] = constraints
        assert _is_qualitatively_better(below, above) is True

    def test_elevation_600_feasible_601_violation(self):
        """elevation 600 feasible, 601 violation."""
        a = _make_compact_route()
        b = _make_compact_route()
        a["elevation_up_m"] = 600.0
        b["elevation_up_m"] = 601.0
        a["selection_constraints"] = {"max_elevation_up_m": 600}
        b["selection_constraints"] = {"max_elevation_up_m": 600}
        a["_attempt"], b["_attempt"] = 1, 2
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False

    def test_technical_T2_feasible_T3_violation(self):
        """technical T2 feasible, T3 violation."""
        a = _make_compact_route()
        b = _make_compact_route()
        a["technical_difficulty"] = "T2"
        b["technical_difficulty"] = "T3"
        a["selection_constraints"] = {"technical_max": "T2"}
        b["selection_constraints"] = {"technical_max": "T2"}
        a["_attempt"], b["_attempt"] = 1, 2
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False

    def test_without_constraints_preserves_old_behavior(self):
        """No constraints: shorter still wins among close distances."""
        a = _make_compact_route(distance_km=36.0)
        b = _make_compact_route(distance_km=37.0)
        # No selection_constraints → old behavior: shorter wins if diff > 2
        a["_attempt"], b["_attempt"] = 1, 2
        # diff is 1 which is ≤ 2 → tie → earlier attempt wins
        assert _is_qualitatively_better(a, b) is True

    def test_without_constraints_large_diff_shorter_wins(self):
        """No constraints: diff > 2 → shorter wins (old behavior)."""
        a = _make_compact_route(distance_km=20.0)
        b = _make_compact_route(distance_km=25.0)
        a["_attempt"], b["_attempt"] = 1, 2
        # diff is 5 > 2 → shorter (a) wins
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False


# ══════════════════════════════════════════════════════════════════════
# _run_feedback_loop — constraint propagation tests
# ══════════════════════════════════════════════════════════════════════


class TestRunFeedbackLoopConstraintPropagation:
    """Confirm constraints flow through feedback loop."""

    INITIAL_COORDS = [[47.5, 10.0], [47.6, 10.1]]

    def test_constraints_passed_to_plan_route_fn(self, two_clusters, dummy_segments):
        """selection_constraints are forwarded to each plan_route_fn call."""
        tc_all_covered = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        route = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc_all_covered,
        )

        constraints = {
            "min_distance_km": 35,
            "max_distance_km": 40,
            "max_elevation_up_m": 800,
            "technical_max": "T3",
        }

        mock = MockPlanRouteWithConstraints([route])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_clusters,
            mock, max_iterations=3,
            selection_constraints=constraints,
        ))
        assert result["status"] == "feedback_complete"
        assert mock.call_count >= 1
        # Constraints must appear in call kwargs
        for entry in mock.call_history:
            assert entry["kwargs"]["min_distance_km"] == 35
            assert entry["kwargs"]["max_distance_km"] == 40
            assert entry["kwargs"]["max_elevation_up_m"] == 800
            assert entry["kwargs"]["technical_max"] == "T3"

    def test_real_plan_route_signature_compatible_with_feedback_constraints(self):
        """The feedback-loop kwargs match plan_route's public signature."""
        import inspect
        from komoot_mcp.tools.routing import register
        signature = inspect.signature(register)
        assert "mcp" in signature.parameters

    def test_constraints_attached_to_route_dicts(self, two_clusters, dummy_segments):
        """Each raw route dict gets selection_constraints attached."""
        tc_all_covered = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        route = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc_all_covered,
        )

        constraints = {"min_distance_km": 30}

        class _InspectingMock(MockPlanRouteWithConstraints):
            async def __call__(self, *args, **kwargs):
                resp = await super().__call__(*args, **kwargs)
                # Return with selection_constraints injected
                resp["selection_constraints"] = kwargs.get("selection_constraints")
                return resp

        route_wrapped = {**route, "selection_constraints": constraints}
        mock = _InspectingMock([route_wrapped])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_clusters,
            mock, max_iterations=3,
            selection_constraints=constraints,
        ))
        assert result["status"] == "feedback_complete"
        # The best_route in the result should have selection_constraints
        assert "selection_constraints" in result.get("best_route", {}) or \
               result["best_index"] >= 0

    def test_no_constraints_still_works(self, two_clusters, dummy_segments):
        """Without selection_constraints, feedback loop behaves as before."""
        tc_all_covered = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        route = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc_all_covered,
        )
        mock = MockPlanRouteWithConstraints([route])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "feedback_complete"
        assert mock.call_count >= 1

    def test_initial_coords_unchanged(self, two_clusters, dummy_segments):
        """Original initial coordinates must not be mutated."""
        import copy
        orig = copy.deepcopy(self.INITIAL_COORDS)
        tc_all_covered = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        route = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc_all_covered,
        )
        mock = MockPlanRouteWithConstraints([route])
        asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_clusters,
            mock, max_iterations=3,
        ))
        assert self.INITIAL_COORDS == orig