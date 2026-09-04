"""Tests for unclassified trail (wt#trail without trail_d*) handling.

Tests the new capability to detect and handle wt#trail elements
that are NOT trail_d1..d5 classified singletrail.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from komoot_mcp.tools.trail_discovery import (
    _trail_category,
    cluster_trail_segments,
    derive_routing_waypoints,
    extract_trail_segments,
)
from komoot_mcp.tools.routing import _compute_singletrail


# ── Test _trail_category helper ─────────────────────────────────────────

class TestTrailCategory:
    def test_classified_singletrail_d1(self):
        assert _trail_category("wt#trail_d1") == "classified"

    def test_classified_singletrail_d5(self):
        assert _trail_category("wt#trail_d5") == "classified"

    def test_unclassified_trail(self):
        assert _trail_category("wt#trail") == "unclassified"

    def test_non_trail_way_type(self):
        assert _trail_category("wt#street") is None
        assert _trail_category("wt#way") is None
        assert _trail_category("wt#cycleway") is None
        assert _trail_category("wt#minor_road") is None
        assert _trail_category("wt#primary") is None

    def test_invalid_input(self):
        assert _trail_category(None) is None
        assert _trail_category(42) is None


# ── Test extract_trail_segments with unclassified trail ─────────────────

_LAT_BASE = 48.0
_LNG_BASE = 10.0


@pytest.fixture
def mixed_wt_items() -> list[dict[str, Any]]:
    """Way type items containing both classified and unclassified trails."""
    return [
        {"from": 0, "to": 5, "element": "wt#minor_road"},
        {"from": 5, "to": 10, "element": "wt#trail_d2"},       # classified
        {"from": 10, "to": 20, "element": "wt#street"},
        {"from": 20, "to": 28, "element": "wt#trail"},          # unclassified
        {"from": 28, "to": 35, "element": "wt#trail_d1"},       # classified
        {"from": 35, "to": 40, "element": "wt#cycleway"},
        {"from": 40, "to": 45, "element": "wt#trail"},          # unclassified
    ]


@pytest.fixture
def mixed_coords() -> list[dict[str, Any]]:
    """45 coordinates, each ~0.001 deg apart (~111m at 48N)."""
    return [
        {"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE, "alt": 700.0, "t": i * 10000}
        for i in range(45)
    ]


class TestExtractUnclassifiedTrail:
    def test_extract_mixed_categories(self, mixed_wt_items, mixed_coords):
        """Both classified (trail_d) and unclassified (trail) segments extracted."""
        segs = extract_trail_segments(mixed_wt_items, mixed_coords, 2001)
        # Expected: trail_d2 (5..10), trail_unclassified (20..28), trail_d1 (28..35), trail_unclassified (40..45)
        assert len(segs) == 4

        # First: classified trail_d2
        assert segs[0]["way_type"] == "trail_d2"
        assert segs[0]["trail_category"] == "classified"
        assert segs[0]["from_index"] == 5

        # Second: unclassified trail
        assert segs[1]["way_type"] == "trail_unclassified"
        assert segs[1]["trail_category"] == "unclassified"
        assert segs[1]["from_index"] == 20

        # Third: classified trail_d1
        assert segs[2]["way_type"] == "trail_d1"
        assert segs[2]["trail_category"] == "classified"
        assert segs[2]["from_index"] == 28

        # Fourth: unclassified trail
        assert segs[3]["way_type"] == "trail_unclassified"
        assert segs[3]["trail_category"] == "unclassified"
        assert segs[3]["from_index"] == 40

    def test_extract_only_unclassified(self):
        """Only wt#trail (non-d) elements → all extracted as unclassified."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail"},
            {"from": 10, "to": 15, "element": "wt#trail"},
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(20)]
        segs = extract_trail_segments(wt, coords, 3001)
        assert len(segs) == 2
        assert all(s["trail_category"] == "unclassified" for s in segs)
        assert all(s["way_type"] == "trail_unclassified" for s in segs)

    def test_extract_length_unclassified(self):
        """Length calculation for unclassified trail segments."""
        # 10 coords spaced 0.001 deg → ~1.0 km
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        wt = [{"from": 0, "to": 10, "element": "wt#trail"}]
        segs = extract_trail_segments(wt, coords, 4001)
        assert len(segs) == 1
        assert segs[0]["length_km"] > 0
        assert segs[0]["way_type"] == "trail_unclassified"


# ── Test clustering with unclassified trail ──────────────────────────────

class TestClusterUnclassified:
    def test_cluster_mixed_categories(self, mixed_wt_items, mixed_coords):
        """Clusters can contain both classified and unclassified trail segments."""
        segs = extract_trail_segments(mixed_wt_items, mixed_coords, 5001)
        # Gap between trail_d2 (end=10) and trail_unclassified (start=20) is ~1.1 km
        # Gap between trail_unclassified (end=28) and trail_d1 (start=28) is 0 km → merged
        # Gap between trail_d1 (end=35) and trail_unclassified (start=40) is ~0.55 km
        # With max_gap_km=0.5: trail_d1 and trail_unclassified [28..45] might merge
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        assert len(clusters) >= 1

        # Check trail_categories field exists
        for c in clusters:
            assert "trail_categories" in c
            assert isinstance(c["trail_categories"], list)

    def test_cluster_unclassified_only(self):
        """Only unclassified trail → cluster with trail_categories=['unclassified']."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail"},
            {"from": 5, "to": 10, "element": "wt#trail"},  # adjacent → merged
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        segs = extract_trail_segments(wt, coords, 6001)
        clusters = cluster_trail_segments(segs)
        assert len(clusters) >= 1
        assert clusters[0]["trail_categories"] == ["unclassified"]

    def test_cluster_separated_mixed(self):
        """Separated classified and unclassified clusters → distinct entries."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},    # classified
            # gap of ~2.2 km
            {"from": 25, "to": 30, "element": "wt#trail"},      # unclassified
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(35)]
        segs = extract_trail_segments(wt, coords, 7001)
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        assert len(clusters) == 2  # gap ~2.2 km > 0.5 km
        assert clusters[0]["trail_categories"] == ["classified"]
        assert clusters[1]["trail_categories"] == ["unclassified"]


# ── Test derive_routing_waypoints priority ──────────────────────────────

class TestDerivePriority:
    def _make_cluster(self, start_lat, start_lng, end_lat, end_lng,
                       length_km=1.0, tid=1001, categories=None):
        categories = categories or ["classified"]
        return {
            "segments": 1, "total_length_km": length_km,
            "source_tour_ids": [tid], "way_types": ["trail_d1"],
            "trail_categories": categories,
            "start": {"lat": start_lat, "lng": start_lng},
            "end": {"lat": end_lat, "lng": end_lng},
        }

    def test_classified_before_unclassified_equal_length(self):
        """Equal-length clusters: classified (A) gets pair before unclassified (B)."""
        cls = [
            self._make_cluster(48.1, 10.0, 48.104, 10.004,
                               length_km=0.5, tid=1001,
                               categories=["unclassified"]),
            self._make_cluster(48.0, 10.0, 48.004, 10.004,
                               length_km=0.5, tid=1002,
                               categories=["classified"]),
        ]
        wps = derive_routing_waypoints(cls, max_points=3)
        # Classified gets pair (2 pts), unclassified gets centroid (1 pt)
        assert len(wps) == 3
        # First pair should be classified cluster entry/exit
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.004, 10.004]
        # Centroid of unclassified
        assert wps[2] == [48.102, 10.002]

    def test_classified_same_length_pair_before_unclassified_single(self):
        """max_points=2: classified gets pair, unclassified gets nothing."""
        cls = [
            self._make_cluster(48.1, 10.0, 48.104, 10.004,
                               length_km=0.5, tid=1001,
                               categories=["unclassified"]),
            self._make_cluster(48.0, 10.0, 48.004, 10.004,
                               length_km=0.5, tid=1002,
                               categories=["classified"]),
        ]
        wps = derive_routing_waypoints(cls, max_points=2)
        assert len(wps) == 2
        assert wps[0] == [48.0, 10.0]  # classified entry
        assert wps[1] == [48.004, 10.004]  # classified exit

    def test_classified_beats_longer_unclassified_cluster(self):
        """Classification priority wins even when unclassified is longer."""
        cls = [
            self._make_cluster(48.1, 10.0, 48.104, 10.004,
                               length_km=0.5, tid=1001,
                               categories=["unclassified"]),
            self._make_cluster(48.0, 10.0, 48.004, 10.004,
                               length_km=0.4, tid=1002,
                               categories=["classified"]),
        ]
        wps = derive_routing_waypoints(cls, max_points=2)
        assert wps == [[48.0, 10.0], [48.004, 10.004]]

    def test_unclassified_only_no_priority_change(self):
        """Only unclassified clusters: sorted by length, no priority shift."""
        cls = [
            self._make_cluster(48.1, 10.0, 48.104, 10.004,
                               length_km=0.3, tid=1001,
                               categories=["unclassified"]),
            self._make_cluster(48.0, 10.0, 48.004, 10.004,
                               length_km=0.5, tid=1002,
                               categories=["unclassified"]),
        ]
        wps = derive_routing_waypoints(cls, max_points=4)
        assert len(wps) == 4
        # Longer cluster first (0.5 km)
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.004, 10.004]
        # Shorter cluster second (0.3 km)
        assert wps[2] == [48.1, 10.0]
        assert wps[3] == [48.104, 10.004]


# ── Test _compute_singletrail with unclassified trail ──────────────────

class TestComputeSingletrailUnclassified:
    def test_unclassified_only(self):
        """Only wt#trail (non-d) → trail_unclassified entry with distance."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail"},
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(5)]
        result = _compute_singletrail(wt, coords, total_km=0.5)
        assert result != {"available": False}
        assert "trail_unclassified" in result
        assert result["trail_unclassified"]["segments"] == 1
        assert "distance_km" in result["trail_unclassified"]
        assert result["trail_unclassified"]["distance_km"] > 0
        assert "unclassified_trail_total_km" in result

    def test_mixed_classified_and_unclassified(self):
        """Both trail_d* and wt#trail present."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},
            {"from": 10, "to": 15, "element": "wt#trail"},
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(20)]
        result = _compute_singletrail(wt, coords, total_km=2.0)

        # Classified singletrail preserved
        assert "trail_d1" in result
        assert "singletrail_total_km" in result
        assert result["trail_d1"]["segments"] == 1

        # Unclassified trail added
        assert "trail_unclassified" in result
        assert "unclassified_trail_total_km" in result

        # Ordered: trail_d1 before trail_unclassified
        keys = list(result.keys())
        d1_idx = keys.index("trail_d1")
        un_idx = keys.index("trail_unclassified")
        assert d1_idx < un_idx

    def test_no_trail_returns_available_false(self):
        """No trail data at all → {'available': False}."""
        wt = [{"from": 0, "to": 5, "element": "wt#street"}]
        coords = [{"lat": 48.0, "lng": 10.0} for _ in range(5)]
        result = _compute_singletrail(wt, coords, total_km=0.5)
        assert result == {"available": False}