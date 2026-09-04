"""Tests for route overlap detection and trail coverage.

Tests _compute_route_overlap and _compute_trail_coverage
with synthetic and real-fixture data.

_coverage computes actual trail-length coverage via sampling:
  - Sample points every ~5 m along each discovered segment
  - Count samples where the route passes within proximity_km
  - Deduplicate across overlapping segments via a global grid
  - covered_km = covered_samples × seg_len / (num_samples-1)
"""

from __future__ import annotations

from typing import Any

import pytest

from komoot_mcp.tools.routing import (
    _compute_route_overlap,
    _compute_trail_coverage,
)

from conftest import load_fixture


# ── Helpers: build synthetic route geometries ─────────────────────────


def _coord(lat: float, lng: float) -> dict[str, float]:
    return {"lat": lat, "lng": lng}


def _straight_line(
    lat_start: float, lng_start: float,
    lat_end: float, lng_end: float,
    steps: int = 50,
) -> list[dict[str, float]]:
    """Generate evenly spaced coords from start to end."""
    coords = []
    for i in range(steps + 1):
        t = i / steps
        coords.append(_coord(
            lat_start + (lat_end - lat_start) * t,
            lng_start + (lng_end - lng_start) * t,
        ))
    return coords


def _out_and_back(
    lat_start: float, lng_start: float,
    lat_turn: float, lng_turn: float,
    steps: int = 30,
) -> list[dict[str, float]]:
    """A->B->A route: forward then identical return."""
    forward = _straight_line(lat_start, lng_start, lat_turn, lng_turn, steps)
    backward = _straight_line(lat_turn, lng_turn, lat_start, lng_start, steps)
    return forward + backward[1:]  # don't duplicate the turn point


def _haversine_km(lat1, lon1, lat2, lon2):
    """Standalone haversine for test helpers."""
    from math import radians, sin, cos, sqrt, asin
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


# ── Route Overlap Tests ───────────────────────────────────────────────


class TestComputeRouteOverlap:
    """_compute_route_overlap unit tests."""

    def test_no_overlap_straight_line(self):
        """Single straight line must have 0 overlap."""
        coords = _straight_line(47.5, 10.0, 47.6, 10.3, steps=100)
        result = _compute_route_overlap(coords)
        assert result["overlap_km"] == 0.0
        assert result["overlap_percentage"] == 0.0

    def test_out_and_back_overlap(self):
        """Exact out-and-back must detect overlap via the *longer* route."""
        coords = _out_and_back(47.5, 10.0, 47.6, 10.3, steps=80)
        result = _compute_route_overlap(coords, proximity_km=0.010)
        assert result["overlap_km"] > 0.0
        assert result["overlap_percentage"] > 20.0

    def test_slightly_offset_return(self):
        """Return path slightly offset (5 m) should still count as overlap."""
        lat_start, lng_start = 47.5, 10.0
        lat_turn, lng_turn = 47.6, 10.3
        forward = _straight_line(lat_start, lng_start, lat_turn, lng_turn, 80)
        offset = 5.0 / 111_320
        backward = _straight_line(
            lat_turn + offset, lng_turn,
            lat_start + offset, lng_start,
            80,
        )
        coords = forward + backward[1:]
        result = _compute_route_overlap(coords, proximity_km=0.020)
        assert result["overlap_km"] > 0.0
        assert result["overlap_percentage"] > 10.0

    def test_parallel_paths_no_overlap(self):
        """Two parallel distinct paths 100 m apart must NOT overlap."""
        path_a = _straight_line(47.5, 10.0, 47.6, 10.0, 50)
        path_b = _straight_line(47.5009, 10.0, 47.6009, 10.0, 50)
        coords = path_a + path_b[1:]
        result = _compute_route_overlap(coords, proximity_km=0.025)
        assert result["overlap_km"] < 0.001

    def test_crossing_no_false_positive(self):
        """Route crossing itself at a single point must NOT flag substantial
        overlap."""
        ew = _straight_line(47.5, 9.9, 47.5, 10.1, 50)
        ns = _straight_line(47.48, 10.0, 47.52, 10.0, 50)
        coords = ew + ns[1:]
        result = _compute_route_overlap(coords, proximity_km=0.010)
        assert result["overlap_percentage"] < 2.0

    def test_min_coords_no_overlap(self):
        """Fewer than 2*min_index_gap coords returns 0."""
        coords = [_coord(47.5, 10.0), _coord(47.6, 10.0)]
        result = _compute_route_overlap(coords)
        assert result["overlap_km"] == 0.0
        assert result["overlap_percentage"] == 0.0

    def test_empty_coords(self):
        """Empty coord list returns 0."""
        result = _compute_route_overlap([])
        assert result["overlap_km"] == 0.0
        assert result["overlap_percentage"] == 0.0


# ── Trail Coverage Tests ──────────────────────────────────────────────


def _segment(
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    category: str,
    length_km: float = 0.0,
) -> dict[str, Any]:
    """Build a discovered segment dict."""
    if length_km == 0.0:
        length_km = round(
            _haversine_km(start_lat, start_lng, end_lat, end_lng), 4
        )
    return {
        "start": {"lat": start_lat, "lng": start_lng},
        "end": {"lat": end_lat, "lng": end_lng},
        "length_km": length_km,
        "trail_category": category,
        "way_type": "trail_d1" if category == "classified" else "trail_unclassified",
    }


class TestComputeTrailCoverage:
    """_compute_trail_coverage unit tests.

    Coverage uses ~5 m sampling along each segment and measures what
    portion of the trail the route actually traverses (not binary).
    """

    def test_empty_discovered(self):
        """Empty discovered segments returns zero coverage."""
        route = _straight_line(47.5, 10.0, 47.6, 10.0, 20)
        result = _compute_trail_coverage(route, [])
        assert result["classified"]["discovered"] == 0
        assert result["unclassified"]["discovered"] == 0
        assert result["total_coverage_percentage"] == 0.0

    def test_route_covers_complete_segment(self):
        """Route directly along the entire segment with sufficient coordinate
        density: ~100 % coverage."""
        # 4.45 km segment, route steps=120 → coord spacing ≈ 37 m
        # proximity=0.025 (25m) → 50 m coverage per coord → continuous
        route = _straight_line(47.52, 10.0, 47.56, 10.0, steps=120)
        seg = _segment(47.52, 10.0, 47.56, 10.0, "classified")
        result = _compute_trail_coverage(route, [seg], proximity_km=0.025)
        assert result["classified"]["covered"] == 1
        assert result["classified"]["discovered"] == 1
        assert result["classified"]["coverage_percentage"] > 90.0

    def test_long_trail_only_partially_covered(self):
        """Route covers only ~25 % of a long trail segment.
        Segment: 47.50-47.54 (≈4.45 km N‑S)
        Route:   47.51-47.515 (≈0.56 km, same line)
        Expected: ~12.5% coverage.
        """
        seg = _segment(47.50, 10.0, 47.54, 10.0, "classified")
        # Dense route coords (steps=30 for 0.56 km → ~19 m spacing)
        route = _straight_line(47.51, 10.0, 47.515, 10.0, steps=30)
        result = _compute_trail_coverage(route, [seg], proximity_km=0.020)
        assert result["classified"]["covered"] == 1
        assert result["classified"]["discovered"] == 1
        covered_pct = result["classified"]["coverage_percentage"]
        covered_km = result["classified"]["covered_km"]
        total_km = result["classified"]["total_km"]
        # ~ 0.56 km / 4.45 km ≈ 12.5%
        assert 5.0 < covered_pct < 30.0
        assert 0.05 < covered_km < total_km

    def test_long_trail_crossed_only_once(self):
        """Route crosses a long trail at a single point; coverage is very low.
        Trail: (47.535, 10.0)-(47.565, 10.0)  ≈ 3.3 km N‑S
        Route: (47.55, 9.99)-(47.55, 10.01)   ≈ 1.1 km E‑W, crosses at
               approx (47.55, 10.0).
        Only samples within ~50 m of the crossing should register.
        """
        seg = _segment(47.535, 10.0, 47.565, 10.0, "classified")
        route = _straight_line(47.55, 9.99, 47.55, 10.01, 40)
        result = _compute_trail_coverage(route, [seg], proximity_km=0.050)
        assert result["classified"]["covered"] == 1  # segment IS used
        cov_pct = result["classified"]["coverage_percentage"]
        # Crossing only: 50m each way along trail = 100m / 3.3km ≈ 3%
        assert cov_pct < 10.0
        assert cov_pct > 0.5

    def test_multiple_sub_sections_covered(self):
        """Route covers several discontiguous sections of a single trail.
        Trail (47.50,10.0)-(47.58,10.0) ≈ 8.9 km

        Two sections: 47.51-47.52 (1.11 km) and 47.56-47.57 (1.11 km),
        both with enough density (steps=30) for continuous coverage.
        Covered: 2.22 km / 8.9 km ≈ 25 %.
        """
        seg = _segment(47.50, 10.0, 47.58, 10.0, "classified")
        section_a = _straight_line(47.51, 10.0, 47.52, 10.0, steps=30)
        section_b = _straight_line(47.56, 10.0, 47.57, 10.0, steps=30)
        route = section_a + section_b
        result = _compute_trail_coverage(route, [seg], proximity_km=0.020)
        assert result["classified"]["covered"] == 1
        cov_pct = result["classified"]["coverage_percentage"]
        # Two ~1.1 km sections out of ~8.9 km ≈ 25 %
        assert 15.0 < cov_pct < 40.0

    def test_classified_unclassified_separate(self):
        """Classified and unclassified breakdowns are separate."""
        route = _straight_line(47.5, 10.0, 47.6, 10.0, 30)
        segments = [
            _segment(47.52, 10.0, 47.54, 10.0, "classified"),
            _segment(47.55, 10.0, 47.58, 10.0, "unclassified"),
        ]
        result = _compute_trail_coverage(route, segments, proximity_km=0.020)
        assert result["classified"]["covered"] == 1
        assert result["unclassified"]["covered"] == 1
        assert result["classified"]["coverage_percentage"] > 0.0
        assert result["unclassified"]["coverage_percentage"] > 0.0

    def test_adjacent_segments_no_double_count(self):
        """Two overlapping segments are not double counted.
        Seg A: (47.52,10.0)-(47.57,10.0)  ≈ 5.6 km
        Seg B: (47.55,10.0)-(47.58,10.0)  ≈ 3.3 km, overlaps with A
        Route covers all with enough density.
        covered_km must not exceed the unique geographic coverage.
        """
        seg_a = _segment(47.52, 10.0, 47.57, 10.0, "classified")
        seg_b = _segment(47.55, 10.0, 47.58, 10.0, "classified")
        route = _straight_line(47.5, 10.0, 47.6, 10.0, steps=200)
        result = _compute_trail_coverage(
            route, [seg_a, seg_b], proximity_km=0.025
        )
        assert result["classified"]["covered"] == 2
        # Without dedup: ~5.6 + ~3.3 ≈ 8.9 km
        # With dedup: unique coverage 47.52-47.58 ≈ 6.7 km
        unique_km = _haversine_km(47.52, 10.0, 47.58, 10.0)
        covered_km = result["classified"]["covered_km"]
        # Dedup should prevent counting the overlap (47.55-47.57) twice
        assert covered_km < unique_km * 1.25, (
            f"covered_km {covered_km} too close to undeduped total "
            f"{_haversine_km(47.52,10.0,47.57,10.0) + _haversine_km(47.55,10.0,47.58,10.0):.4f}"
        )

    def test_small_routing_deviation_tolerated(self):
        """Route offset by ~10 m parallel to the segment, with sufficient
        coord density, still counts as fully covered."""
        seg = _segment(47.50, 10.0, 47.60, 10.0, "classified")
        # 11.13 km segment, route steps=200 → ~56 m spacing
        # proximity=0.050 → 100 m coverage per coord → continuous
        route = _straight_line(47.50, 10.00009, 47.60, 10.00009, steps=200)
        result = _compute_trail_coverage(route, [seg], proximity_km=0.050)
        assert result["classified"]["covered"] == 1
        assert result["classified"]["coverage_percentage"] > 80.0

    def test_route_entirely_outside(self):
        """Route nowhere near the segment: 0 coverage."""
        seg = _segment(47.5, 10.0, 47.6, 10.0, "classified")
        route = _straight_line(48.0, 10.0, 48.1, 10.0, 20)
        result = _compute_trail_coverage(route, [seg], proximity_km=0.020)
        assert result["classified"]["covered"] == 0
        assert result["classified"]["coverage_percentage"] == 0.0

    def test_zero_length_segment_handled(self):
        """Zero-length segment doesn't break coverage calculation."""
        route = _straight_line(47.5, 10.0, 47.6, 10.0, 20)
        segments = [
            {"start": {"lat": 47.55, "lng": 10.0},
             "end": {"lat": 47.55, "lng": 10.0},
             "length_km": 0.0,
             "trail_category": "classified",
             "way_type": "trail_d1"},
        ]
        result = _compute_trail_coverage(route, segments, proximity_km=0.010)
        assert result["classified"]["covered"] == 1


# ── Integration: real fixture ─────────────────────────────────────────


class TestRealFixtureRouteOverlap:
    """Plan_route fixture (may have some genuine overlap)."""

    def test_real_route_overlap_present(self):
        """Compact=False output includes route_overlap with valid keys."""
        from conftest import FakeClient, FakeMCP

        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod

        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio

        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb",
            compact=True,
        ))
        assert "route_overlap" in result
        ov = result["route_overlap"]
        assert "overlap_km" in ov
        assert "overlap_percentage" in ov
        assert isinstance(ov["overlap_km"], float)
        assert isinstance(ov["overlap_percentage"], float)

    def test_real_route_singletrail_unchanged(self):
        """Existing singletrail computation unchanged after adding overlap."""
        from conftest import FakeClient, FakeMCP

        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod

        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio

        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb",
            compact=True,
        ))
        st = result["singletrail"]
        assert st.get("singletrail_total_km", 0) > 0
        assert st.get("singletrail_percentage", 0) > 0


# ── Cluster Coverage & Feedback-Loop Tests ─────────────────────────────


def _cluster(
    seg_ct: int, length_km: float,
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic cluster dict mimicking cluster_trail_segments()."""
    return {
        "segments": seg_ct,
        "total_length_km": length_km,
        "source_tour_ids": [12345],
        "way_types": ["trail_d1" if "classified" in (categories or ["classified"]) else "trail_unclassified"],
        "trail_categories": categories or ["classified"],
        "start": {"lat": start_lat, "lng": start_lng},
        "end": {"lat": end_lat, "lng": end_lng},
    }


class TestSampleSegmentAlong:
    """_sample_segment_along unit tests."""

    def test_sample_on_trail_counts(self):
        from komoot_mcp.tools.routing import _sample_segment_along
        seg = _segment(47.50, 10.0, 47.51, 10.0, "classified")
        route = _straight_line(47.50, 10.0, 47.51, 10.0, steps=50)
        cnt, km = _sample_segment_along(seg, route, proximity_km=0.025)
        assert cnt > 5
        assert km > 0.0

    def test_sample_off_trail_zero(self):
        from komoot_mcp.tools.routing import _sample_segment_along
        seg = _segment(47.50, 10.0, 47.51, 10.0, "classified")
        route = _straight_line(48.0, 10.0, 48.1, 10.0, steps=50)
        cnt, km = _sample_segment_along(seg, route, proximity_km=0.025)
        assert cnt == 0
        assert km == 0.0


class TestAnalyzeClusterCoverage:
    """analyze_cluster_coverage unit tests."""

    def test_empty_clusters(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        route = _straight_line(47.5, 10.0, 47.6, 10.0, 50)
        assert analyze_cluster_coverage([], route) == []

    def test_cluster_fully_covered(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        cluster = _cluster(2, 5.0, 47.5, 10.0, 47.59, 10.0)
        route = _straight_line(47.5, 10.0, 47.59, 10.0, steps=100)
        result = analyze_cluster_coverage([cluster], route, proximity_km=0.050)
        assert len(result) == 1
        assert result[0]["covered"] is True
        assert result[0]["coverage_percentage"] > 70.0

    def test_cluster_crossed_only_once(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        cluster = _cluster(1, 3.0, 47.50, 10.0, 47.53, 10.0)
        route = _straight_line(47.515, 9.99, 47.515, 10.01, steps=30)
        result = analyze_cluster_coverage([cluster], route, proximity_km=0.050)
        assert len(result) == 1
        assert result[0]["covered"] is False
        assert result[0]["coverage_percentage"] < 10.0

    def test_cluster_partially_covered(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        cluster = _cluster(1, 5.0, 47.50, 10.0, 47.55, 10.0)
        route = _straight_line(47.50, 10.0, 47.515, 10.0, steps=30)
        result = analyze_cluster_coverage([cluster], route, proximity_km=0.025)
        assert len(result) == 1
        assert result[0]["covered"] is False
        assert result[0]["partial"] is True
        pct = result[0]["coverage_percentage"]
        assert 10.0 <= pct < 70.0

    def test_multiple_clusters_independent(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        clusters = [
            _cluster(1, 2.0, 47.50, 10.0, 47.52, 10.0),
            _cluster(1, 2.0, 47.60, 10.0, 47.62, 10.0),
        ]
        route = _straight_line(47.50, 10.0, 47.52, 10.0, steps=50)
        result = analyze_cluster_coverage(clusters, route, proximity_km=0.025)
        assert len(result) == 2
        assert result[0]["covered"] is True
        assert result[1]["covered"] is False
        assert result[1]["coverage_percentage"] < 10.0

    def test_classified_and_unclassified_separate(self):
        from komoot_mcp.tools.routing import analyze_cluster_coverage
        clusters = [
            _cluster(1, 2.0, 47.50, 10.0, 47.52, 10.0, categories=["classified"]),
            _cluster(1, 2.0, 47.55, 10.0, 47.57, 10.0, categories=["unclassified"]),
        ]
        route = _straight_line(47.50, 10.0, 47.57, 10.0, steps=70)
        result = analyze_cluster_coverage(clusters, route, proximity_km=0.050)
        assert len(result) == 2
        assert "classified" in result[0]["trail_categories"]
        assert "unclassified" in result[1]["trail_categories"]


class TestFindUncoveredClusters:
    """find_uncovered_clusters unit tests."""

    def test_categorization(self):
        from komoot_mcp.tools.routing import find_uncovered_clusters
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 85.3,
             "total_length_km": 2.0, "covered_km": 1.7,
             "trail_categories": ["classified"], "way_types": ["trail_d1"],
             "covered": True, "partial": False},
            {"cluster_index": 1, "coverage_percentage": 45.0,
             "total_length_km": 2.0, "covered_km": 0.9,
             "trail_categories": ["classified"], "way_types": ["trail_d1"],
             "covered": False, "partial": True},
            {"cluster_index": 2, "coverage_percentage": 3.2,
             "total_length_km": 2.0, "covered_km": 0.06,
             "trail_categories": ["unclassified"], "way_types": ["trail_unclassified"],
             "covered": False, "partial": False},
        ]
        result = find_uncovered_clusters(analyzed)
        assert len(result["covered"]) == 1
        assert len(result["partial"]) == 1
        assert len(result["uncovered"]) == 1
        assert result["covered"][0]["cluster_index"] == 0
        assert result["partial"][0]["cluster_index"] == 1
        assert result["uncovered"][0]["cluster_index"] == 2

    def test_empty_analyzed(self):
        from komoot_mcp.tools.routing import find_uncovered_clusters
        result = find_uncovered_clusters([])
        assert result["covered"] == []
        assert result["partial"] == []
        assert result["uncovered"] == []

    def test_thresholds_sensible(self):
        """Covered≥70%, uncovered<10%, partial in between."""
        from komoot_mcp.tools.routing import find_uncovered_clusters
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 69.9,
             "trail_categories": ["classified"], "total_length_km": 1.0,
             "covered_km": 0.7, "way_types": ["trail_d1"],
             "covered": False, "partial": True},
            {"cluster_index": 1, "coverage_percentage": 70.0,
             "trail_categories": ["classified"], "total_length_km": 1.0,
             "covered_km": 0.7, "way_types": ["trail_d1"],
             "covered": True, "partial": False},
        ]
        r = find_uncovered_clusters(analyzed)
        assert len(r["partial"]) == 1
        assert len(r["covered"]) == 1


class TestSuggestUncoveredWaypoints:
    """suggest_uncovered_waypoints unit tests."""

    def test_empty_no_suggestions(self):
        from komoot_mcp.tools.routing import suggest_uncovered_waypoints
        assert suggest_uncovered_waypoints([], []) == []

    def test_uncovered_cluster_gets_entry_exit(self):
        from komoot_mcp.tools.routing import suggest_uncovered_waypoints
        clusters = [_cluster(1, 2.0, 47.50, 10.0, 47.52, 10.0)]
        analyzed = [{"cluster_index": 0, "coverage_percentage": 3.0,
                     "trail_categories": ["classified"], "total_length_km": 2.0,
                     "covered_km": 0.06, "way_types": ["trail_d1"],
                     "covered": False, "partial": False}]
        wps = suggest_uncovered_waypoints(clusters, analyzed, max_new=2)
        assert len(wps) == 2
        assert round(wps[0][0], 4) == 47.5
        assert round(wps[0][1], 4) == 10.0

    def test_max_new_respected(self):
        from komoot_mcp.tools.routing import suggest_uncovered_waypoints
        clusters = [
            _cluster(1, 2.0, 47.5, 10.0, 47.52, 10.0),
            _cluster(1, 2.0, 47.6, 10.0, 47.62, 10.0),
        ]
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 2.0,
             "trail_categories": ["classified"], "total_length_km": 2.0,
             "covered_km": 0.04, "way_types": ["trail_d1"],
             "covered": False, "partial": False},
            {"cluster_index": 1, "coverage_percentage": 1.0,
             "trail_categories": ["classified"], "total_length_km": 2.0,
             "covered_km": 0.02, "way_types": ["trail_d1"],
             "covered": False, "partial": False},
        ]
        wps = suggest_uncovered_waypoints(clusters, analyzed, max_new=1)
        assert len(wps) == 1

    def test_classified_before_unclassified(self):
        from komoot_mcp.tools.routing import suggest_uncovered_waypoints
        clusters = [
            _cluster(1, 2.0, 47.5, 10.0, 47.52, 10.0, categories=["unclassified"]),
            _cluster(1, 2.5, 47.6, 10.0, 47.625, 10.0, categories=["classified"]),
        ]
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 3.0,
             "trail_categories": ["unclassified"], "total_length_km": 2.0,
             "covered_km": 0.06, "way_types": ["trail_unclassified"],
             "covered": False, "partial": False},
            {"cluster_index": 1, "coverage_percentage": 2.0,
             "trail_categories": ["classified"], "total_length_km": 2.5,
             "covered_km": 0.05, "way_types": ["trail_d1"],
             "covered": False, "partial": False},
        ]
        wps = suggest_uncovered_waypoints(clusters, analyzed, max_new=2)
        # Classified cluster (idx 1) gets entry+exit (both slots with max_new=2)
        # Unclassified cluster (idx 0) has no remaining budget
        assert len(wps) == 2
        classified_lat = round(47.6, 4)
        wps_lats = [round(w[0], 4) for w in wps]
        assert classified_lat in wps_lats

    def test_classified_first_with_budget(self):
        """With max_new=4, classified waypoints appear first in ordering."""
        from komoot_mcp.tools.routing import suggest_uncovered_waypoints
        clusters = [
            _cluster(1, 2.0, 47.5, 10.0, 47.52, 10.0, categories=["unclassified"]),
            _cluster(1, 2.5, 47.6, 10.0, 47.625, 10.0, categories=["classified"]),
        ]
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 3.0,
             "trail_categories": ["unclassified"], "total_length_km": 2.0,
             "covered_km": 0.06, "way_types": ["trail_unclassified"],
             "covered": False, "partial": False},
            {"cluster_index": 1, "coverage_percentage": 2.0,
             "trail_categories": ["classified"], "total_length_km": 2.5,
             "covered_km": 0.05, "way_types": ["trail_d1"],
             "covered": False, "partial": False},
        ]
        wps = suggest_uncovered_waypoints(clusters, analyzed, max_new=4)
        assert len(wps) == 4
        # First two should be classified (lat=47.6), then unclassified (lat=47.5)
        assert round(wps[0][0], 4) == 47.6  # classified entry
        assert round(wps[2][0], 4) == 47.5  # unclassified entry


class TestCompactClusterInfo:
    """_compact_cluster_info unit tests."""

    def test_compact_format_no_coords(self):
        from komoot_mcp.tools.routing import _compact_cluster_info
        analyzed = [
            {"cluster_index": 0, "coverage_percentage": 85.3,
             "total_length_km": 2.0, "covered_km": 1.7,
             "trail_categories": ["classified"], "way_types": ["trail_d1"],
             "covered": True, "partial": False},
        ]
        compact = _compact_cluster_info(analyzed)
        assert "covered" in compact
        assert "partial" in compact
        assert "uncovered" in compact
        entry = compact["covered"][0]
        assert "cluster_index" in entry
        assert "total_length_km" in entry
        assert "covered_km" in entry
        assert "coverage_percentage" in entry
        assert "trail_categories" in entry
        # No coordinate data in compact output
        assert "start" not in entry
        assert "end" not in entry
        assert "way_types" not in entry


class TestPlanRouteWithDiscoveredSegments:
    """plan_route with discovered_segments integration tests."""

    def test_plan_route_without_discovered_unchanged(self):
        """plan_route without discovered_segments works exactly as before."""
        from conftest import FakeClient, FakeMCP
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio
        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb", compact=True,
        ))
        assert "route_overlap" in result
        assert "trail_coverage" not in result

    def test_plan_route_with_discovered_includes_coverage(self):
        """When discovered_segments is provided, trail_coverage appears."""
        from conftest import FakeClient, FakeMCP
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio
        segments = [_segment(47.50, 10.0, 47.55, 10.0, "classified")]
        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb", compact=True,
            discovered_segments=segments,
        ))
        assert "trail_coverage" in result
        assert "classified" in result["trail_coverage"]
        assert "unclassified" in result["trail_coverage"]
        assert "total_coverage_percentage" in result["trail_coverage"]

    def test_plan_route_compact_false_with_discovered(self):
        """compact=False + discovered_segments: coverage still present, route_ref works."""
        from conftest import FakeClient, FakeMCP
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio
        segments = [_segment(47.50, 10.0, 47.55, 10.0, "classified")]
        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb", compact=False,
            discovered_segments=segments,
        ))
        assert "route_ref" in result
        assert "trail_coverage" in result

    def test_plan_route_with_clusters_and_segments(self):
        """With both discovered_segments and discovered_clusters, trail_clusters appears."""
        from conftest import FakeClient, FakeMCP
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        import asyncio
        segments = [_segment(47.50, 10.0, 47.55, 10.0, "classified")]
        clusters = [_cluster(1, 2.0, 47.50, 10.0, 47.52, 10.0)]
        result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb", compact=True,
            discovered_segments=segments,
            discovered_clusters=clusters,
        ))
        assert "trail_clusters" in result
        assert "covered" in result["trail_clusters"]
        assert "partial" in result["trail_clusters"]
        assert "uncovered" in result["trail_clusters"]
        assert "trail_coverage_suggestions" in result

    def test_ref_integration_still_works(self):
        """create_planned_tour with route_ref still works."""
        from conftest import FakeClient, FakeMCP
        mcp = FakeMCP()
        client = FakeClient(route_response=load_fixture("plan_route_full.json"))
        import komoot_mcp.tools.routing as routing_mod
        routing_mod.register(mcp, client)
        plan_route = mcp.tools["plan_route"]
        create = mcp.tools["create_planned_tour"]
        import asyncio

        route_result = asyncio.run(plan_route(
            [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]],
            sport="mtb", compact=False,
        ))
        ref = route_result["route_ref"]
        created = asyncio.run(create(
            route_ref=ref, name="Test Tour", sport="mtb",
        ))
        # Should succeed: status is 'success' or at least has 'id'
        assert created.get("status") != "error"