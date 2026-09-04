"""Tests for the feedback-loop route optimization (_run_feedback_loop).

Tests the full orchestration: route -> analyze -> suggest -> re-route,
up to max 3 iterations, with early-stop logic and qualitative comparison.

All plan_route calls are mocked — no real Komoot API calls.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from komoot_mcp.tools.routing import (
    _build_analyzed_from_clusters,
    _feedback_iteration_summary,
    _is_qualitatively_better,
    _run_feedback_loop,
    _select_best_route_index,
)


# ── Fixtures: shared test data ────────────────────────────────────────


@pytest.fixture
def two_classified_clusters() -> list[dict[str, Any]]:
    """Two classified trail clusters (discovery format with start/end)."""
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
def three_mixed_clusters() -> list[dict[str, Any]]:
    """Two classified + one unclassified cluster."""
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
        {
            "segments": 1, "total_length_km": 1.0,
            "source_tour_ids": [103], "way_types": ["trail_unclassified"],
            "trail_categories": ["unclassified"],
            "start": {"lat": 47.60, "lng": 10.10},
            "end": {"lat": 47.62, "lng": 10.12},
        },
    ]


@pytest.fixture
def dummy_segments() -> list[dict[str, Any]]:
    """Minimal discovered_segments to satisfy the function pre-check."""
    return [{
        "start": {"lat": 47.5, "lng": 10.0},
        "end": {"lat": 47.52, "lng": 10.02},
        "length_km": 2.0,
        "trail_category": "classified",
        "way_type": "trail_d1",
    }]


_EMPTY_TC = {"covered": [], "partial": [], "uncovered": []}


# ── Mock helpers ──────────────────────────────────────────────────────


class MockPlanRoute:
    """Mocks plan_route(compact=True) with caller-controlled responses.

    Each response is a compact route dict.  Responses repeat if fewer
    than calls made.  Records call history for assertions.
    """

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
        record = {
            "coordinates": [list(c) for c in coordinates],
            "sport": sport,
            "waypoint_count": len(coordinates),
            "kwargs": dict(kwargs),
        }
        self.call_history.append(record)
        return resp


def _cluster_entry(
    cluster_index: int,
    coverage_pct: float,
    categories: list[str],
    **overrides: Any,
) -> dict[str, Any]:
    """Build a single analyzed cluster entry (format returned by
    _compact_cluster_info inside trail_clusters.covered/partial/uncovered).
    """
    d: dict[str, Any] = {
        "cluster_index": cluster_index,
        "total_length_km": 2.0,
        "covered_km": round(2.0 * coverage_pct / 100, 4),
        "coverage_percentage": coverage_pct,
        "trail_categories": categories,
    }
    d.update(overrides)
    return d


def _make_compact_route(
    distance_km: float = 20.0,
    asphalt_pct: float = 30.0,
    overlap_pct: float = 2.5,
    singletrail_km: float = 3.5,
    trail_coverage_classified_covered: int = 0,
    trail_coverage_classified_pct: float = 0.0,
    trail_coverage_total_pct: float = 0.0,
    trail_coverage_uniclassified_covered: int = 0,
    trail_clusters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a simulated plan_route compact output."""
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
                "discovered": 0, "covered": trail_coverage_uniclassified_covered,
                "total_km": 0.0, "covered_km": 0.0,
                "coverage_percentage": 0.0,
            },
            "total_coverage_percentage": trail_coverage_total_pct,
        },
    }
    if trail_clusters is not None:
        base["trail_clusters"] = trail_clusters
    return base


# ══════════════════════════════════════════════════════════════════════
# Unit tests for helper functions
# ══════════════════════════════════════════════════════════════════════


class TestBuildAnalyzedFromClusters:
    """_build_analyzed_from_clusters"""

    def test_empty(self):
        assert _build_analyzed_from_clusters(_EMPTY_TC) == []

    def test_covered_only(self):
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        result = _build_analyzed_from_clusters(tc)
        assert len(result) == 1
        assert result[0]["covered"] is True
        assert result[0]["cluster_index"] == 0

    def test_partial_only(self):
        tc = {
            "covered": [],
            "partial": [_cluster_entry(0, 45.0, ["classified"])],
            "uncovered": [],
        }
        result = _build_analyzed_from_clusters(tc)
        assert len(result) == 1
        assert result[0]["partial"] is True
        assert result[0]["covered"] is False

    def test_uncovered_only(self):
        tc = {
            "covered": [],
            "partial": [],
            "uncovered": [_cluster_entry(0, 3.0, ["classified"])],
        }
        result = _build_analyzed_from_clusters(tc)
        assert len(result) == 1
        assert result[0]["covered"] is False
        assert result[0]["partial"] is False

    def test_all_categories(self):
        tc = {
            "covered": [_cluster_entry(0, 90.0, ["classified"])],
            "partial": [_cluster_entry(1, 45.0, ["classified"])],
            "uncovered": [_cluster_entry(2, 2.0, ["unclassified"])],
        }
        result = _build_analyzed_from_clusters(tc)
        assert len(result) == 3
        assert result[0]["covered"] is True
        assert result[1]["partial"] is True
        assert result[2]["covered"] is False


class TestSelectBestRouteIndex:
    """_select_best_route_index"""

    def test_empty_returns_minus_one(self):
        assert _select_best_route_index([]) == -1

    def test_single_iteration(self):
        its = [{"_attempt": 1}]
        assert _select_best_route_index(its) == 0

    def test_better_formance_selected(self):
        its = [
            {"_attempt": 1,
             "trail_coverage": {"classified": {"covered": 0, "coverage_percentage": 10.0},
                                "total_coverage_percentage": 10.0}},
            {"_attempt": 2,
             "trail_coverage": {"classified": {"covered": 1, "coverage_percentage": 50.0},
                                "total_coverage_percentage": 50.0}},
        ]
        # Attempt 2 has more covered clusters -> better
        assert _select_best_route_index(its) == 1


class TestIsQualitativelyBetter:
    """_is_qualitatively_better"""

    def test_more_covered_clusters_wins(self):
        a = _make_compact_route(
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=30.0,
            trail_coverage_total_pct=20.0,
            trail_clusters={"covered": [_cluster_entry(0, 30.0, ["classified"])],
                            "partial": [],
                            "uncovered": [_cluster_entry(1, 5.0, ["classified"])]},
        )
        b = _make_compact_route(
            trail_coverage_classified_covered=0, trail_coverage_classified_pct=10.0,
            trail_coverage_total_pct=5.0,
            trail_clusters={"covered": [], "partial": [],
                            "uncovered": [_cluster_entry(0, 5.0, ["classified"]),
                                          _cluster_entry(1, 3.0, ["classified"])]},
        )
        a["_attempt"] = 2
        b["_attempt"] = 1
        assert _is_qualitatively_better(a, b) is True

    def test_higher_coverage_pct_wins(self):
        a = _make_compact_route(
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=60.0,
            trail_coverage_total_pct=40.0,
            trail_clusters={"covered": [_cluster_entry(0, 60.0, ["classified"])],
                            "partial": [], "uncovered": []},
        )
        b = _make_compact_route(
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=20.0,
            trail_coverage_total_pct=15.0,
            trail_clusters={"covered": [_cluster_entry(0, 20.0, ["classified"])],
                            "partial": [], "uncovered": []},
        )
        a["_attempt"] = 2
        b["_attempt"] = 1
        assert _is_qualitatively_better(a, b) is True

    def test_less_asphalt_wins(self):
        a = _make_compact_route(asphalt_pct=15.0)
        b = _make_compact_route(asphalt_pct=40.0)
        assert _is_qualitatively_better(a, b) is True

    def test_more_trail_coverage_beats_shorter_route(self):
        """Trail coverage outranks distance."""
        a = _make_compact_route(
            distance_km=30.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=80.0,
            trail_coverage_total_pct=80.0,
        )
        b = _make_compact_route(
            distance_km=15.0,
            trail_coverage_classified_covered=0,
            trail_coverage_classified_pct=10.0,
            trail_coverage_total_pct=10.0,
        )
        assert _is_qualitatively_better(a, b) is True

    def test_more_trail_beats_more_asphalt(self):
        """Asphalt is only a late tie-breaker after trail criteria."""
        a = _make_compact_route(
            asphalt_pct=60.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=80.0,
            trail_coverage_total_pct=80.0,
        )
        b = _make_compact_route(
            asphalt_pct=10.0,
            trail_coverage_classified_covered=0,
            trail_coverage_classified_pct=20.0,
            trail_coverage_total_pct=20.0,
        )
        assert _is_qualitatively_better(a, b) is True

    def test_technical_limit_only_matters_when_explicit(self):
        """T difficulty is considered only with an explicit max constraint."""
        a = _make_compact_route()
        b = _make_compact_route()
        a["technical_difficulty"] = "T3"
        b["technical_difficulty"] = "T2"
        a["_attempt"], b["_attempt"] = 2, 1
        assert _is_qualitatively_better(a, b) is False

        a["selection_constraints"] = {"technical_max": "T2"}
        b["selection_constraints"] = {"technical_max": "T2"}
        assert _is_qualitatively_better(b, a) is True

    def test_elevation_limit_does_not_prefer_lower_feasible_route(self):
        """Under a max-HM limit, two feasible routes remain comparable."""
        a = _make_compact_route()
        b = _make_compact_route()
        a["elevation_up_m"], b["elevation_up_m"] = 500.0, 590.0
        a["selection_constraints"] = {"max_elevation_up_m": 600}
        b["selection_constraints"] = {"max_elevation_up_m": 600}
        a["_attempt"], b["_attempt"] = 2, 1
        assert _is_qualitatively_better(b, a) is True

    def test_elevation_limit_prefers_feasible_route(self):
        a = _make_compact_route()
        b = _make_compact_route()
        a["elevation_up_m"], b["elevation_up_m"] = 500.0, 700.0
        a["selection_constraints"] = {"max_elevation_up_m": 600}
        b["selection_constraints"] = {"max_elevation_up_m": 600}
        assert _is_qualitatively_better(a, b) is True

    def test_small_overlap_gap_not_decisive(self):
        """Small overlap diff (4.5 vs 3.0 = 1.5% gap) must NOT alone decide.
        Tie goes to earlier attempt."""
        a = _make_compact_route(overlap_pct=4.5)
        b = _make_compact_route(overlap_pct=3.0)
        a["_attempt"] = 1
        b["_attempt"] = 2
        # All other metrics identical. Overlap diff < 5% → tie.
        # Earlier attempt wins: a (attempt 1) beats b (attempt 2)
        assert _is_qualitatively_better(a, b) is True   # earlier wins tie
        assert _is_qualitatively_better(b, a) is False  # later loses tie

    def test_large_overlap_gap_penalises(self):
        """10% overlap gap IS meaningful -> lower overlap wins."""
        a = _make_compact_route(overlap_pct=12.0)
        b = _make_compact_route(overlap_pct=2.0)
        assert _is_qualitatively_better(b, a) is True
        assert _is_qualitatively_better(a, b) is False

    def test_earlier_attempt_wins_tie(self):
        a = _make_compact_route()
        b = _make_compact_route()
        a["_attempt"] = 1
        b["_attempt"] = 2
        assert _is_qualitatively_better(a, b) is True   # earlier wins
        assert _is_qualitatively_better(b, a) is False  # later loses

    # ── Targeted tests per parent requirements ───────────────────────

    def test_classified_beats_longer_unclassified_route(self):
        """Classification beats longer but unclassified route."""
        a = _make_compact_route(
            distance_km=25.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=50.0,
            trail_coverage_total_pct=50.0,
        )
        b = _make_compact_route(
            distance_km=10.0,
            trail_coverage_classified_covered=0,
            trail_coverage_classified_pct=0.0,
            trail_coverage_total_pct=30.0,
        )
        # a has classified coverage → beats b despite being longer
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False


    def test_covered_clusters_beat_shorter_distance(self):
        """More covered classified clusters beat a shorter route with fewer."""
        a = _make_compact_route(
            distance_km=30.0,
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=60.0,
            trail_coverage_total_pct=60.0,
            trail_clusters={
                "covered": [_cluster_entry(0, 85.0, ["classified"]),
                            _cluster_entry(1, 75.0, ["classified"])],
                "partial": [],
                "uncovered": [],
            },
        )
        b = _make_compact_route(
            distance_km=15.0,
            trail_coverage_classified_covered=0,
            trail_coverage_classified_pct=10.0,
            trail_coverage_total_pct=10.0,
            trail_clusters={
                "covered": [],
                "partial": [_cluster_entry(0, 30.0, ["classified"])],
                "uncovered": [_cluster_entry(1, 5.0, ["classified"])],
            },
        )
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False

    def test_technical_difficulty_not_compared_without_explicit_max(self):
        """T diff is irrelevant when no explicit max constraint exists."""
        a = _make_compact_route()
        b = _make_compact_route()
        a["technical_difficulty"] = "T3"
        b["technical_difficulty"] = "T1"
        a["_attempt"], b["_attempt"] = 1, 2
        # T3 vs T1, but no max constraint → earlier attempt wins
        assert _is_qualitatively_better(a, b) is True  # earlier
        assert _is_qualitatively_better(b, a) is False

    def test_approach_return_overlap_allowed(self):
        """Approach/return overlap (≤25%) does not disqualify a route."""
        # Both have same classified coverage, same distance
        # a has 0% overlap, b has 20% approach/return overlap
        tc_covered = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        a = _make_compact_route(
            distance_km=20.0,
            overlap_pct=0.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=85.0,
            trail_coverage_total_pct=85.0,
            trail_clusters=tc_covered,
        )
        b = _make_compact_route(
            distance_km=20.0,
            overlap_pct=20.0,  # approach/return overlap
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=85.0,
            trail_coverage_total_pct=85.0,
            trail_clusters=tc_covered,
        )
        a["_attempt"], b["_attempt"] = 1, 2
        # Same trail coverage, distance — overlap gap 20% > 5% → a wins
        assert _is_qualitatively_better(a, b) is True
        # But b is not categorically excluded — if b has better trail
        # COVERAGE (higher priority criterion), it wins despite overlap
        b2 = _make_compact_route(
            distance_km=22.0,
            overlap_pct=20.0,
            singletrail_km=3.5,
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=90.0,
            trail_coverage_total_pct=90.0,
            trail_clusters={
                "covered": [_cluster_entry(0, 95.0, ["classified"]),
                            _cluster_entry(1, 85.0, ["classified"])],
                "partial": [], "uncovered": [],
            },
        )
        b2["_attempt"] = 2
        # b2 has more covered classified clusters (2 vs 1) → beats a despite overlap
        assert _is_qualitatively_better(b2, a) is True

    def test_asphalt_does_not_beat_better_trail_route(self):
        """Asphalt is last criterion and cannot override trail coverage."""
        a = _make_compact_route(
            asphalt_pct=80.0,
            singletrail_km=8.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=80.0,
            trail_coverage_total_pct=80.0,
            trail_clusters={
                "covered": [_cluster_entry(0, 85.0, ["classified"])],
                "partial": [], "uncovered": [],
            },
        )
        b = _make_compact_route(
            asphalt_pct=5.0,
            singletrail_km=1.0,
            trail_coverage_classified_covered=0,
            trail_coverage_classified_pct=10.0,
            trail_coverage_total_pct=10.0,
            trail_clusters={},
        )
        # a has 80% asphalt but much better trail → a wins
        assert _is_qualitatively_better(a, b) is True
        assert _is_qualitatively_better(b, a) is False

    def test_lower_elevation_not_always_better_if_trail_lost(self):
        """Weniger Höhenmeter sind nicht besser, wenn dafür Trail verloren geht."""
        # Same cluster structure, but a loses a classified covered cluster vs b
        tc_a = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [], "uncovered": [_cluster_entry(1, 5.0, ["classified"])],
        }
        tc_b = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 88.0, ["classified"])],
            "partial": [], "uncovered": [],
        }
        a = _make_compact_route(
            distance_km=18.0,
            trail_coverage_classified_covered=1,
            trail_coverage_classified_pct=45.0,
            trail_coverage_total_pct=45.0,
            trail_clusters=tc_a,
        )
        a["elevation_up_m"] = 300.0
        b = _make_compact_route(
            distance_km=25.0,
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=86.0,
            trail_coverage_total_pct=86.0,
            trail_clusters=tc_b,
        )
        b["elevation_up_m"] = 800.0
        a["_attempt"], b["_attempt"] = 1, 2
        # b has more covered clusters (2 vs 1) → b wins despite more HM
        assert _is_qualitatively_better(b, a) is True
        assert _is_qualitatively_better(a, b) is False


class TestFeedbackIterationSummary:
    """_feedback_iteration_summary"""

    def test_compact_format(self):
        route = _make_compact_route(
            distance_km=22.0,
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=50.0,
            trail_coverage_total_pct=50.0,
        )
        route["trail_clusters"] = {"covered": [_cluster_entry(0, 85.0, ["classified"])],
                                   "partial": [], "uncovered": []}
        summary = _feedback_iteration_summary(route, 1, [[47.5, 10.0]], 0)
        assert summary["attempt"] == 1
        assert summary["distance_km"] == 22.0
        assert "trail_clusters" in summary
        assert "route_overlap" in summary
        assert "surfaces" in summary
        assert "waypoints_total" in summary
        assert "path" not in summary

    def test_no_trail_coverage_omitted(self):
        route = {"distance_km": 15.0}
        s = _feedback_iteration_summary(route, 1, [[47.5, 10.0]], 0)
        assert "trail_coverage" not in s
        assert "route_overlap" not in s


# ══════════════════════════════════════════════════════════════════════
# Integration tests for _run_feedback_loop
# ══════════════════════════════════════════════════════════════════════


class TestRunFeedbackLoop:
    """_run_feedback_loop integration tests."""

    INITIAL_COORDS = [[47.5, 10.0], [47.6, 10.1]]

    def test_early_stop_all_covered_1_call(
        self, two_classified_clusters, dummy_segments,
    ):
        """If first attempt covers all clusters -> only 1 routing call."""
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        route1 = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc,
        )
        mock = MockPlanRoute([route1])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "feedback_complete"
        assert mock.call_count == 1
        assert result["total_attempts"] == 1

    def test_progressive_improvement_3_calls(
        self, three_mixed_clusters, dummy_segments,
    ):
        """With uncovered classified clusters, up to 3 iterations used.
        Each iteration shows progress toward more coverage.
        """
        tc1 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(1, 5.0, ["classified"]),
                          _cluster_entry(2, 2.0, ["unclassified"])],
        }
        r1 = _make_compact_route(
            distance_km=20.0,
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=42.0,
            trail_coverage_total_pct=30.0,
            trail_clusters=tc1,
        )
        tc2 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [_cluster_entry(1, 45.0, ["classified"])],
            "uncovered": [_cluster_entry(2, 2.0, ["unclassified"])],
        }
        r2 = _make_compact_route(
            distance_km=22.0,
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=65.0,
            trail_coverage_total_pct=50.0,
            trail_clusters=tc2,
        )
        tc3 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 72.0, ["classified"]),
                        _cluster_entry(2, 82.0, ["unclassified"])],
            "partial": [],
            "uncovered": [],
        }
        r3 = _make_compact_route(
            distance_km=24.0,
            trail_coverage_classified_covered=2, trail_coverage_classified_pct=78.0,
            trail_coverage_total_pct=80.0,
            trail_clusters=tc3,
        )

        mock = MockPlanRoute([r1, r2, r3])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, three_mixed_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "feedback_complete"
        assert mock.call_count == 3, f"Expected 3 calls, got {mock.call_count}"
        assert result["total_attempts"] == 3
        assert result["best_index"] == 2  # attempt 3 has best coverage
        assert len(result["iterations"]) == 3
        for i, it in enumerate(result["iterations"]):
            assert it["attempt"] == i + 1
            assert it["distance_km"] > 0

    def test_never_exceeds_max_3(
        self, two_classified_clusters, dummy_segments,
    ):
        """Even with always-uncovered clusters, never more than 3 calls."""
        tc_uncovered = {
            "covered": [],
            "partial": [],
            "uncovered": [_cluster_entry(0, 2.0, ["classified"]),
                          _cluster_entry(1, 1.0, ["classified"])],
        }
        route = _make_compact_route(
            trail_coverage_classified_covered=0, trail_coverage_classified_pct=2.0,
            trail_coverage_total_pct=2.0,
            trail_clusters=tc_uncovered,
        )
        mock = MockPlanRoute([route, route, route, route])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["total_attempts"] <= 3
        assert mock.call_count <= 3

    def test_no_auto_save(self, two_classified_clusters, dummy_segments):
        """Feedback loop must NOT create route_ref / cache entries."""
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        route1 = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc,
        )
        mock = MockPlanRoute([route1])
        from komoot_mcp.tools import routing as routing_mod
        routing_mod._route_cache.clear()
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "feedback_complete"
        assert len(routing_mod._route_cache) == 0

    def test_discovery_not_rerun(self, three_mixed_clusters, dummy_segments):
        """Loop never re-fetches discovered data (it uses the passed values)."""
        tc = {
            "covered": [],
            "partial": [],
            "uncovered": [_cluster_entry(0, 3.0, ["classified"]),
                          _cluster_entry(1, 2.0, ["classified"]),
                          _cluster_entry(2, 1.0, ["unclassified"])],
        }
        r1 = _make_compact_route(trail_clusters=tc)
        r2 = _make_compact_route(trail_clusters=tc)
        r3 = _make_compact_route(trail_clusters=tc)
        mock = MockPlanRoute([r1, r2, r3])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, three_mixed_clusters,
            mock, max_iterations=3,
        ))
        assert result["total_attempts"] <= 3

    def test_best_route_selected_correctly(
        self, two_classified_clusters, dummy_segments,
    ):
        """Route with more covered clusters is selected as best."""
        tc1 = {
            "covered": [],
            "partial": [_cluster_entry(0, 45.0, ["classified"])],
            "uncovered": [_cluster_entry(1, 3.0, ["classified"])],
        }
        tc2 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 72.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        r1 = _make_compact_route(
            distance_km=20.0,
            trail_coverage_classified_covered=0, trail_coverage_classified_pct=45.0,
            trail_coverage_total_pct=30.0,
            trail_clusters=tc1,
        )
        r2 = _make_compact_route(
            distance_km=24.0,
            trail_coverage_classified_covered=2, trail_coverage_classified_pct=78.0,
            trail_coverage_total_pct=78.0,
            trail_clusters=tc2,
        )
        mock = MockPlanRoute([r1, r2])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=2,
        ))
        assert result["best_index"] == 1

    def test_classified_clusters_prioritised(
        self, three_mixed_clusters, dummy_segments,
    ):
        """Classified clusters are selected before unclassified for waypoints."""
        tc1 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(1, 5.0, ["classified"]),
                          _cluster_entry(2, 3.0, ["unclassified"])],
        }
        r1 = _make_compact_route(
            trail_coverage_classified_covered=1, trail_coverage_classified_pct=42.0,
            trail_coverage_total_pct=30.0,
            trail_clusters=tc1,
        )
        tc2 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 72.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(2, 3.0, ["unclassified"])],
        }
        r2 = _make_compact_route(
            trail_coverage_classified_covered=2, trail_coverage_classified_pct=78.0,
            trail_coverage_total_pct=60.0,
            trail_clusters=tc2,
        )
        mock = MockPlanRoute([r1, r2])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, three_mixed_clusters,
            mock, max_iterations=3,
        ))
        assert result["total_attempts"] >= 2
        assert result["best_index"] >= 0

    def test_overlap_alone_does_not_trigger_extra(
        self, two_classified_clusters, dummy_segments,
    ):
        """High overlap alone, without uncovered clusters, does NOT add iterations."""
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        r1 = _make_compact_route(
            overlap_pct=25.0,
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc,
        )
        mock = MockPlanRoute([r1])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert mock.call_count == 1
        assert result["total_attempts"] == 1

    def test_approach_overlap_not_rejected(
        self, two_classified_clusters, dummy_segments,
    ):
        """Route with more coverage but moderate overlap wins over less overlap + less coverage."""
        tc1 = {
            "covered": [],
            "partial": [_cluster_entry(0, 45.0, ["classified"])],
            "uncovered": [_cluster_entry(1, 3.0, ["classified"])],
        }
        r1 = _make_compact_route(
            overlap_pct=8.0,
            trail_coverage_classified_covered=0, trail_coverage_classified_pct=24.0,
            trail_coverage_total_pct=24.0,
            trail_clusters=tc1,
        )
        tc2 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 72.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        r2 = _make_compact_route(
            overlap_pct=12.0,
            trail_coverage_classified_covered=2, trail_coverage_classified_pct=78.0,
            trail_coverage_total_pct=78.0,
            trail_clusters=tc2,
        )
        mock = MockPlanRoute([r1, r2])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["best_index"] == 1  # more coverage wins over less overlap

    def test_max_waypoints_2_per_iteration(
        self, two_classified_clusters, dummy_segments,
    ):
        """Each iteration adds at most 2 waypoints."""
        tc1 = {
            "covered": [],
            "partial": [],
            "uncovered": [_cluster_entry(0, 2.0, ["classified"]),
                          _cluster_entry(1, 1.0, ["classified"])],
        }
        route1 = _make_compact_route(trail_clusters=tc1)
        tc2 = {
            "covered": [_cluster_entry(0, 75.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(1, 3.0, ["classified"])],
        }
        route2 = _make_compact_route(trail_clusters=tc2)
        tc3 = {
            "covered": [_cluster_entry(0, 75.0, ["classified"]),
                        _cluster_entry(1, 70.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        route3 = _make_compact_route(trail_clusters=tc3)
        mock = MockPlanRoute([route1, route2, route3])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        for i, it in enumerate(result["iterations"]):
            assert it["waypoints_added"] <= 2

    def test_clusters_covered_summary(
        self, two_classified_clusters, dummy_segments,
    ):
        """Final result includes clusters_covered breakdown."""
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(1, 3.0, ["classified"])],
        }
        r1 = _make_compact_route(trail_clusters=tc)
        mock = MockPlanRoute([r1])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=1,
        ))
        assert "clusters_covered" in result
        assert "classified" in result["clusters_covered"]
        assert "clusters_uncovered_indices" in result

    @pytest.mark.parametrize("bad_max", [0, -1])
    def test_invalid_max_iterations(self, two_classified_clusters, bad_max, dummy_segments):
        """max_iterations < 1 returns error immediately."""
        mock = MockPlanRoute([])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=bad_max,
        ))
        assert result["status"] == "error"
        assert mock.call_count == 0

    def test_no_discovered_segments_returns_error(self, two_classified_clusters):
        """Missing discovered_segments returns error, no routing calls."""
        mock = MockPlanRoute([])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb",
            [],  # empty list — still "no segments" from the function's perspective
            two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "error"
        assert mock.call_count == 0

    def test_waypoints_come_from_cluster_entry_exit(
        self, two_classified_clusters, dummy_segments,
    ):
        """Added waypoints must be cluster start/end coordinates, not random."""
        tc1 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"])],
            "partial": [],
            "uncovered": [_cluster_entry(1, 3.0, ["classified"])],
        }
        route1 = _make_compact_route(trail_clusters=tc1)
        tc2 = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 75.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        route2 = _make_compact_route(trail_clusters=tc2)
        mock = MockPlanRoute([route1, route2])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=2,
        ))
        assert result["total_attempts"] == 2
        # The second call should have received more waypoints than the first
        assert len(mock.call_history[0]["coordinates"]) < len(mock.call_history[1]["coordinates"])
        # Extra waypoints should match cluster start or end coords
        extra = mock.call_history[1]["coordinates"][2:]  # after initial 2
        for wp in extra:
            # Must be close to cluster start or end
            near_cluster = any(
                abs(wp[0] - c["start"]["lat"]) < 0.001 and abs(wp[1] - c["start"]["lng"]) < 0.001
                or abs(wp[0] - c["end"]["lat"]) < 0.001 and abs(wp[1] - c["end"]["lng"]) < 0.001
                for c in two_classified_clusters
            )
            assert near_cluster, f"Waypoint {wp} is not from any cluster entry/exit"

    def test_constraints_not_required(self, two_classified_clusters, dummy_segments):
        """Old call without constraints works — no selection_constraints needed."""
        tc = {
            "covered": [_cluster_entry(0, 85.0, ["classified"]),
                        _cluster_entry(1, 90.0, ["classified"])],
            "partial": [],
            "uncovered": [],
        }
        route1 = _make_compact_route(
            trail_coverage_classified_covered=2,
            trail_coverage_classified_pct=87.0,
            trail_coverage_total_pct=87.0,
            trail_clusters=tc,
        )
        mock = MockPlanRoute([route1])
        result = asyncio.run(_run_feedback_loop(
            self.INITIAL_COORDS, "mtb", dummy_segments, two_classified_clusters,
            mock, max_iterations=3,
        ))
        assert result["status"] == "feedback_complete"
        assert mock.call_count == 1


class TestIsQualitativelyBetterDistanceConstraints:
    """_is_qualitatively_better with distance constraints (min_distance_km, max_distance_km)."""

    def _with_sc(self, route, constraints):
        """Attach selection_constraints to a route copy."""
        r = dict(route)
        r["selection_constraints"] = constraints
        r["_attempt"] = r.get("_attempt", 1)
        return r

    # ── In-range beats out-of-range ────────────────────────────────

    def test_min_distance_prefers_feasible_over_violation(self):
        """With min=35, 36km (feasible) beats 34km (violation)."""
        a = _make_compact_route(distance_km=36.0)
        b = _make_compact_route(distance_km=34.0)
        a = self._with_sc(a, {"min_distance_km": 35})
        b = self._with_sc(b, {"min_distance_km": 35})
        assert _is_qualitatively_better(a, b) is True

    def test_max_distance_prefers_feasible_over_violation(self):
        """With max=40, 39km (feasible) beats 41km (violation)."""
        a = _make_compact_route(distance_km=39.0)
        b = _make_compact_route(distance_km=41.0)
        a = self._with_sc(a, {"max_distance_km": 40})
        b = self._with_sc(b, {"max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is True

    def test_min_distance_violation_loses_to_any_feasible(self):
        """34km (violation with min=35) loses to 37km (feasible)."""
        a = _make_compact_route(distance_km=23.75)
        b = _make_compact_route(distance_km=37.0)
        a = self._with_sc(a, {"min_distance_km": 35, "max_distance_km": 40})
        b = self._with_sc(b, {"min_distance_km": 35, "max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is False
        assert _is_qualitatively_better(b, a) is True

    # ── Both in-range: distance does NOT decide ─────────────────────

    def test_both_in_range_shorter_not_preferred_by_distance(self):
        """With min=35, max=40, 36km does NOT beat 38km just because shorter."""
        a = _make_compact_route(distance_km=36.0)
        b = _make_compact_route(distance_km=38.0)
        a = self._with_sc(a, {"min_distance_km": 35, "max_distance_km": 40})
        b = self._with_sc(b, {"min_distance_km": 35, "max_distance_km": 40})
        # Both in-range, same coverage — earlier attempt wins
        a["_attempt"] = 1
        b["_attempt"] = 2
        assert _is_qualitatively_better(a, b) is True  # earlier

    def test_both_in_range_trail_coverage_still_decides(self):
        """Trail coverage still beats distance when both in-range."""
        a = _make_compact_route(
            distance_km=38.0,
            trail_coverage_classified_covered=2, trail_coverage_classified_pct=80.0,
            trail_coverage_total_pct=80.0,
            trail_clusters={
                "covered": [_cluster_entry(0, 85.0, ["classified"]),
                            _cluster_entry(1, 75.0, ["classified"])],
                "partial": [], "uncovered": [],
            },
        )
        b = _make_compact_route(
            distance_km=36.0,
            trail_coverage_classified_covered=0, trail_coverage_classified_pct=10.0,
            trail_coverage_total_pct=10.0,
        )
        a = self._with_sc(a, {"min_distance_km": 35, "max_distance_km": 40})
        b = self._with_sc(b, {"min_distance_km": 35, "max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is True

    # ── Both out-of-range: closer to range wins ────────────────────

    def test_both_out_of_range_closer_wins(self):
        """With 35..40, 34km (excess 1) beats 30km (excess 5)."""
        a = _make_compact_route(distance_km=34.0)
        b = _make_compact_route(distance_km=30.0)
        a = self._with_sc(a, {"min_distance_km": 35, "max_distance_km": 40})
        b = self._with_sc(b, {"min_distance_km": 35, "max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is True

    def test_both_too_long_closer_wins(self):
        """With 35..40, 41km beats 45km."""
        a = _make_compact_route(distance_km=41.0)
        b = _make_compact_route(distance_km=45.0)
        a = self._with_sc(a, {"min_distance_km": 35, "max_distance_km": 40})
        b = self._with_sc(b, {"min_distance_km": 35, "max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is True

    # ── Without constraints: old behavior preserved ────────────────

    def test_no_constraints_shorter_wins_old_behavior(self):
        """Without distance constraints, 30km beats 40km (diff > 2km − shorter wins)."""
        a = _make_compact_route(distance_km=30.0)
        b = _make_compact_route(distance_km=40.0)
        a["_attempt"], b["_attempt"] = 2, 1
        assert _is_qualitatively_better(a, b) is True

    # ── Only one constraint bound ──────────────────────────────────

    def test_min_only_feasible_wins(self):
        """Only min=35: 36 (feasible) beats 34 (violation)."""
        a = _make_compact_route(distance_km=36.0)
        b = _make_compact_route(distance_km=34.0)
        a = self._with_sc(a, {"min_distance_km": 35})
        b = self._with_sc(b, {"min_distance_km": 35})
        assert _is_qualitatively_better(a, b) is True

    def test_max_only_feasible_wins(self):
        """Only max=40: 39 (feasible) beats 41 (violation)."""
        a = _make_compact_route(distance_km=39.0)
        b = _make_compact_route(distance_km=41.0)
        a = self._with_sc(a, {"max_distance_km": 40})
        b = self._with_sc(b, {"max_distance_km": 40})
        assert _is_qualitatively_better(a, b) is True


class TestRunFeedbackLoopConstraints:
    """_run_feedback_loop with constraint parameters."""

    INITIAL_COORDS = [[47.5, 10.0], [47.6, 10.1]]

    def test_constraints_attached_to_each_iteration(self, dummy_segments, two_classified_clusters):
        """Constraints dict is passed to plan_route_fn and recorded in call history."""
        tc = {
            "covered": [],
            "partial": [],
            "uncovered": [_cluster_entry(0, 3.0, ["classified"]),
                          _cluster_entry(1, 2.0, ["classified"])],
        }
        route = _make_compact_route(trail_clusters=tc)
        mock = MockPlanRoute([route, route, route])

        from komoot_mcp.tools.routing import _run_feedback_loop as fb
        result = asyncio.run(fb(
            self.INITIAL_COORDS, "mtb",
            dummy_segments, two_classified_clusters, mock,
            max_iterations=3,
            selection_constraints={"min_distance_km": 35, "max_distance_km": 40},
        ))

        for rec in mock.call_history:
            kwargs = rec.get("kwargs", {})
            assert kwargs.get("min_distance_km") == 35
            assert kwargs.get("max_distance_km") == 40
            assert mock.call_count <= 3


# ══════════════════════════════════════════════════════════════════════
# Route_ref compatibility
# ══════════════════════════════════════════════════════════════════════


class TestPlanRouteWithFeedbackParameter:
    """plan_route(compact=False/True) unaffected by feedback_loop parameter."""

    COORDS = [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]]

    def test_compact_false_still_works(self):
        """Plain plan_route(compact=False) unchanged."""
        from conftest import FakeClient, FakeMCP, load_fixture
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        result = asyncio.run(plan_route(self.COORDS, sport="mtb", compact=False))
        assert "route_ref" in result
        assert result["route_ref"].startswith("route_")

    def test_compact_true_still_works(self):
        """Plain plan_route(compact=True) unchanged."""
        from conftest import FakeClient, FakeMCP, load_fixture
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        result = asyncio.run(plan_route(self.COORDS, sport="mtb", compact=True))
        assert "route_ref" not in result
        assert result["distance_km"] > 0