"""Tests for trail_discovery feature -- no live Komoot API calls."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from komoot_mcp.tools.trail_discovery import (
    _extract_coord_items,
    _extract_way_type_items,
    _haversine_km,
    _segment_km,
    cluster_trail_segments,
    derive_routing_waypoints,
    discover_trail_hotspots,
    extract_trail_segments,
)

# -- Fixtures -------------------------------------------------------------

# Single-tour fixture: 3 trail segments (trail_d1, trail_d2, trail_d2)
# Coordinates form a simple north-south line at roughly constant longitude.
# This lets us predict distances easily.
_LAT_BASE = 48.0
_LNG_BASE = 10.0

@pytest.fixture
def simple_tour_wt() -> list[dict[str, Any]]:
    """Way type items for a simple tour: 3 trail segments, some other types."""
    return [
        {"from": 0, "to": 10, "element": "wt#minor_road"},
        {"from": 10, "to": 15, "element": "wt#trail_d2"},
        {"from": 15, "to": 20, "element": "wt#street"},
        {"from": 20, "to": 30, "element": "wt#trail_d1"},
        {"from": 30, "to": 35, "element": "wt#cycleway"},
        {"from": 35, "to": 38, "element": "wt#trail_d2"},
        {"from": 38, "to": 40, "element": "wt#minor_road"},
        {"from": 40, "to": 50, "element": "wt#path"},
    ]


@pytest.fixture
def simple_tour_coords() -> list[dict[str, Any]]:
    """50 coordinates, each ~0.001 deg apart (~111 m at 48N)."""
    return [
        {"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE, "alt": 700.0, "t": i * 10000}
        for i in range(50)
    ]


@pytest.fixture
def simple_tour_data(simple_tour_wt, simple_tour_coords) -> dict[str, Any]:
    return {
        "tour_id": 1001,
        "wt_items": simple_tour_wt,
        "coord_items": simple_tour_coords,
    }


# Two-tour fixture: second tour has some overlapping trail segments
@pytest.fixture
def second_tour_wt() -> list[dict[str, Any]]:
    return [
        {"from": 0, "to": 5, "element": "wt#street"},
        {"from": 5, "to": 8, "element": "wt#trail_d2"},
        {"from": 8, "to": 12, "element": "wt#cycleway"},
        {"from": 12, "to": 22, "element": "wt#trail_d1"},
        {"from": 22, "to": 30, "element": "wt#minor_road"},
    ]


@pytest.fixture
def second_tour_coords() -> list[dict[str, Any]]:
    """30 coordinates, offset by ~0.05 deg east to simulate nearby but not same route."""
    return [
        {"lat": _LAT_BASE + i * 0.0008, "lng": _LNG_BASE + 0.05, "alt": 700.0, "t": i * 10000}
        for i in range(30)
    ]


# -- Tests: extract_trail_segments ---------------------------------------

class TestExtractTrailSegments:
    def test_single_segment(self):
        """Erkennung einzelner Trailabschnitte: nur ein trail_d1 Segment."""
        wt = [{"from": 0, "to": 5, "element": "wt#trail_d1"}]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(5)]
        segs = extract_trail_segments(wt, coords, 42)
        assert len(segs) == 1
        seg = segs[0]
        assert seg["way_type"] == "trail_d1"
        assert seg["source_tour_id"] == 42
        assert seg["start"]["lat"] == _LAT_BASE
        assert seg["end"]["lat"] == _LAT_BASE + 4 * 0.001
        assert seg["length_km"] > 0
        assert seg["from_index"] == 0
        assert seg["to_index"] == 5

    def test_multiple_consecutive_trail_points(self, simple_tour_wt, simple_tour_coords):
        """Mehrere aufeinanderfolgende Trailpunkte: Segmente dicht beieinander."""
        segs = extract_trail_segments(simple_tour_wt, simple_tour_coords, 1001)
        # trail_d2 [10..15], trail_d1 [20..30], trail_d2 [35..38]
        assert len(segs) == 3
        # Check they are in order
        assert segs[0]["way_type"] == "trail_d2"
        assert segs[0]["from_index"] == 10
        assert segs[1]["way_type"] == "trail_d1"
        assert segs[1]["from_index"] == 20
        assert segs[2]["way_type"] == "trail_d2"
        assert segs[2]["from_index"] == 35

    def test_multiple_separated_segments(self):
        """Mehrere getrennte Trailabschnitte mit Lcken dazwischen."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},
            {"from": 10, "to": 15, "element": "wt#trail_d2"},
            {"from": 25, "to": 30, "element": "wt#trail_d1"},
        ]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(35)]
        segs = extract_trail_segments(wt, coords, 1002)
        assert len(segs) == 3
        assert segs[0]["from_index"] == 0
        assert segs[1]["from_index"] == 10
        assert segs[2]["from_index"] == 25

    def test_segment_length(self):
        """Lngenberechnung: bekannter Abstand zwischen Koordinaten."""
        # 10 coords spaced 0.001 deg (~0.111 km each at 48N) = ~1.0 km total
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        wt = [{"from": 0, "to": 9, "element": "wt#trail_d1"}]
        segs = extract_trail_segments(wt, coords, 1003)
        assert len(segs) == 1
        # 9 segments of ~0.111 km each = ~0.999 km approx
        expected = round(9 * _haversine_km(_LAT_BASE, _LNG_BASE, _LAT_BASE + 0.001, _LNG_BASE), 4)
        assert segs[0]["length_km"] == pytest.approx(expected, rel=0.01)

    def test_no_trail_segments(self):
        """Keine Trailabschnitte: nur Strassen/Wegen."""
        wt = [
            {"from": 0, "to": 10, "element": "wt#minor_road"},
            {"from": 10, "to": 20, "element": "wt#street"},
        ]
        coords = [{"lat": 0.0, "lng": 0.0} for _ in range(20)]
        segs = extract_trail_segments(wt, coords, 1004)
        assert segs == []

    def test_empty_input(self):
        """Leere/ungltige Daten."""
        assert extract_trail_segments([], [], 42) == []
        assert extract_trail_segments([], [{"lat": 1.0, "lng": 2.0}], 42) == []
        assert extract_trail_segments(None, None, 42) == []

    def test_invalid_way_type_structure(self):
        """Ungltige way_type Strukturen werden ignoriert."""
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},  # valid
            {"element": "wt#trail_d2"},  # missing from/to
            {},  # empty
            "string_item",  # not a dict
            42,  # not a dict
        ]
        coords = [{"lat": _LAT_BASE, "lng": _LNG_BASE} for _ in range(6)]
        segs = extract_trail_segments(wt, coords, 1005)
        assert len(segs) == 1  # only the first one is valid

    def test_index_out_of_range(self):
        """from_index > to_index or beyond coord list is skipped."""
        wt = [
            {"from": 5, "to": 3, "element": "wt#trail_d1"},  # invalid range
            {"from": 0, "to": 999, "element": "wt#trail_d2"},  # beyond coord count
        ]
        coords = [{"lat": 48.0, "lng": 10.0} for _ in range(10)]
        segs = extract_trail_segments(wt, coords, 1006)
        assert len(segs) == 0  # both are invalid


# -- Tests: cluster_trail_segments ---------------------------------------

class TestClusterTrailSegments:
    def test_single_tour_clusters_separated(self):
        """Single tour: 2 separated trail clusters with a gap between them."""
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(40)]
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},
            # gap of 15 coords (~1.6 km)
            {"from": 20, "to": 25, "element": "wt#trail_d2"},
        ]
        segs = extract_trail_segments(wt, coords, 1001)
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        assert len(clusters) == 2  # gap is ~1.6 km > 0.5 km max_gap

    def test_single_tour_clusters_merged(self):
        """Single tour: 2 trail segments close enough to merge into one cluster."""
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(40)]
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d1"},
            # gap of 1 coord (~0.111 km)
            {"from": 6, "to": 10, "element": "wt#trail_d2"},
        ]
        segs = extract_trail_segments(wt, coords, 1001)
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        assert len(clusters) == 1  # gap ~0.111 km < 0.5 km max_gap

    def test_multi_tour_clustering(self):
        """Mehrere Touren: Segmente von zwei Touren rumlich nah -> ein Cluster."""
        coords1 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        # Second tour starts at same lat as first tour's end, slightly offset east
        coords2 = [{"lat": _LAT_BASE + 4 * 0.001 + i * 0.001, "lng": _LNG_BASE + 0.001} for i in range(10)]

        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords1, 1001
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords2, 1002
        )

        # segs1[0] ends at (48.004, 10.0)
        # segs2[0] starts at (48.004, 10.001) -- gap ~74m

        all_segs = segs1 + segs2
        clusters = cluster_trail_segments(all_segs, max_gap_km=0.15)
        assert len(clusters) == 1  # gap ~74m < 0.15 km
        assert clusters[0]["source_tour_ids"] == [1001, 1002]

    def test_multi_tour_separated(self):
        """Mehrere Touren: Segmente weit genug entfernt -> separate Cluster."""
        coords1 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        coords2 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE + 1.0} for i in range(10)]

        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d2"}], coords1, 1001
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d2"}], coords2, 1002
        )

        all_segs = segs1 + segs2
        clusters = cluster_trail_segments(all_segs, max_gap_km=0.2)
        assert len(clusters) == 2  # ~111 km apart > 0.2 km

    def test_empty_segments(self):
        """Keine Segmente -> kein Cluster."""
        assert cluster_trail_segments([]) == []

    def test_duplicate_segments_from_same_tour(self):
        """Duplikaterkennung: selbe Tour-ID -> korrekt in cluster source_ids."""
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(20)]
        wt = [
            {"from": 0, "to": 5, "element": "wt#trail_d2"},
            {"from": 5, "to": 10, "element": "wt#trail_d1"},
        ]
        segs = extract_trail_segments(wt, coords, 2001)
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        assert len(clusters) == 1
        # Same tour, so source_tour_ids should have only one entry
        assert clusters[0]["source_tour_ids"] == [2001]


# -- Tests: discover_trail_hotspots --------------------------------------

class TestDiscoverTrailHotspots:
    def test_single_tour_no_hotspot(self):
        """Einzelne Tour: kein Hotspot (min_overlap=2, nur eine Quelle)."""
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        segs = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords, 1001
        )
        hotspots = discover_trail_hotspots(segs, min_overlap=2)
        assert hotspots == []

    def test_two_tours_hotspot(self):
        """Zwei Touren berschneiden sich nahe -> ein Hotspot."""
        coords1 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        coords2 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE + 0.05} for i in range(10)]

        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords1, 1001
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords2, 1002
        )

        all_segs = segs1 + segs2
        # segs1[0] start: (48.0, 10.0), end: (48.004, 10.0)
        # segs2[0] start: (48.0, 10.05), end: (48.004, 10.05)
        # They are ~5.5 km apart, so no overlap with radius_km=0.15
        hotspots = discover_trail_hotspots(all_segs, min_overlap=2, radius_km=10.0)
        assert len(hotspots) >= 1
        assert [1001, 1002] in [h["source_tour_ids"] for h in hotspots]

    def test_hotspot_within_radius(self):
        """Zwei Touren starten am selben Punkt -> Hotspot wird erkannt."""
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords, 1001
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d2"}], coords, 1002
        )
        all_segs = segs1 + segs2
        hotspots = discover_trail_hotspots(all_segs, min_overlap=2, radius_km=0.1)
        # Both start at same coords -> should have hotspot at (48.0, 10.0)
        assert len(hotspots) >= 1

    def test_empty_input(self):
        """Leere Eingabe -> keine Hotspots."""
        assert discover_trail_hotspots([]) == []


# -- Tests: multi-tour integration ---------------------------------------

class TestMultiTour:
    def test_two_tours_combined(self):
        """Mehrere Touren als Quelle: Segmente beider Touren in der Ausgabe."""
        coords1 = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(10)]
        coords2 = [{"lat": _LAT_BASE + i * 0.002, "lng": _LNG_BASE + 1.0} for i in range(10)]

        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords1, 2001
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d2"}], coords2, 2002
        )
        all_segs = segs1 + segs2

        assert len(all_segs) == 2
        assert all_segs[0]["source_tour_id"] == 2001
        assert all_segs[1]["source_tour_id"] == 2002

        clusters = cluster_trail_segments(all_segs, max_gap_km=10.0)
        assert len(clusters) == 2  # far apart even with generous gap
        assert [2001, 2002] == sorted([
            tid for c in clusters for tid in c["source_tour_ids"]
        ])

    def test_trail_segments_from_two_tours_preserve_source(self):
        """Source-tour-IDs bleiben erhalten: kein Vermischen."""
        coords1 = [{"lat": 48.0 + i * 0.001, "lng": 10.0} for i in range(10)]
        coords2 = [{"lat": 48.0 + i * 0.001, "lng": 10.0} for i in range(10)]

        segs1 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d1"}], coords1, 1501
        )
        segs2 = extract_trail_segments(
            [{"from": 0, "to": 5, "element": "wt#trail_d2"}], coords2, 1502
        )

        assert all(s["source_tour_id"] == 1501 for s in segs1)
        assert all(s["source_tour_id"] == 1502 for s in segs2)


# -- Tests: compact output -----------------------------------------------

class TestCompactOutput:
    def test_compact_structure(self, simple_tour_wt, simple_tour_coords):
        """Kompakte Ausgabe: clusters + hotspots, keine per-segment Details."""
        segs = extract_trail_segments(simple_tour_wt, simple_tour_coords, 1001)
        clusters = cluster_trail_segments(segs, max_gap_km=0.5)
        hotspots = discover_trail_hotspots(segs, min_overlap=2)
        # In compact mode we'd only return clusters + hotspots + total_segments
        # Verify the data exists
        assert len(clusters) >= 1  # trail_d2 [10..15] + trail_d1 [20..30] may merge with 0.5 gap
        assert len(segs) == 3

    def test_segments_all_contain_endpoints(self):
        """Jeder Trailabschnitt hat start/end Koordinaten."""
        wt = [{"from": 0, "to": 5, "element": "wt#trail_d2"}]
        coords = [{"lat": _LAT_BASE + i * 0.001, "lng": _LNG_BASE} for i in range(5)]
        segs = extract_trail_segments(wt, coords, 200)
        assert "start" in segs[0]
        assert "end" in segs[0]
        assert "lat" in segs[0]["start"]
        assert "lng" in segs[0]["end"]


# -- Tests: helper / edge cases ------------------------------------------

class TestHelpers:
    def test_haversine_short_distance(self):
        """Haversine: kurze Distanz ~74m bei 0.001 longitude at 48N."""
        d = _haversine_km(48.0, 10.0, 48.0, 10.001)
        assert d == pytest.approx(0.0744, abs=0.001)

    def test_haversine_latitude_distance(self):
        """Haversine: ~111km pro Grad."""
        d = _haversine_km(48.0, 10.0, 49.0, 10.0)
        assert d == pytest.approx(111.0, abs=1.0)

    def test_extract_way_type_items_valid(self):
        """_extract_way_type_items: korrekte Extraktion."""
        tour = {"_embedded": {"way_types": {"items": [{"element": "wt#trail_d1"}]}}}
        assert _extract_way_type_items(tour) == [{"element": "wt#trail_d1"}]

    def test_extract_way_type_items_empty(self):
        """_extract_way_type_items: fehlende/leere Daten."""
        assert _extract_way_type_items({}) == []
        assert _extract_way_type_items({"__embedded": {}}) == []
        assert _extract_way_type_items({"embedded": None}) == []

    def test_extract_coord_items_valid(self):
        """_extract_coord_items: korrekte Extraktion."""
        tour = {"_embedded": {"coordinates": {"items": [{"lat": 48.0, "lng": 10.0}]}}}
        assert _extract_coord_items(tour) == [{"lat": 48.0, "lng": 10.0}]

    def test_extract_coord_items_empty(self):
        """_extract_coord_items: fehlende/leere Daten."""
        assert _extract_coord_items({}) == []
        assert _extract_coord_items({"embedded": None}) == []

    def test_segment_km_zero_length(self):
        """_segment_km: einzelner Punkt -> 0 km."""
        coords = [{"lat": 48.0, "lng": 10.0}, {"lat": 48.0, "lng": 10.0}]
        assert _segment_km(coords, 0, 1) == 0.0

    def test_segment_km_full_path(self):
        """_segment_km: gesamte Strecke."""
        coords = [
            {"lat": 48.0, "lng": 10.0},
            {"lat": 48.001, "lng": 10.0},
            {"lat": 48.002, "lng": 10.0},
        ]
        d = _segment_km(coords, 0, 2)
        expected = _haversine_km(48.0, 10.0, 48.001, 10.0) + _haversine_km(48.001, 10.0, 48.002, 10.0)
        assert d == pytest.approx(expected, rel=0.001)

    def test_segment_km_out_of_bounds(self):
        """_segment_km: Indizes auerhalb des Bereichs -> kein Fehler."""
        coords = [{"lat": 48.0, "lng": 10.0}]
        assert _segment_km(coords, 0, 10) == 0.0


# -- Tests: real-data fixture --------------------------------------------

class TestWithRealFixture:
    """Teste mit der echten plan_route_full.json Fixture (eingebettete way_types/coords)."""

    @pytest.fixture
    def plan_route_full(self):
        path = Path(__file__).parent / "fixtures" / "plan_route_full.json"
        return json.loads(path.read_text())

    def test_extract_from_real_data(self, plan_route_full):
        """Aus echten Komoot-Daten Trailabschnitte extrahieren."""
        embedded = plan_route_full["_embedded"]
        wt_items = embedded["way_types"]["items"]
        coord_items = embedded["coordinates"]["items"]

        segs = extract_trail_segments(wt_items, coord_items, 9999)
        # The fixture has 9 trail items (3 trail_d1, 6 trail_d2)
        assert len(segs) <= 9  # some may merge due to consecutive trail items
        assert len(segs) >= 3
        for s in segs:
            assert s["way_type"] in ("trail_d1", "trail_d2")
            assert s["length_km"] > 0
            assert s["source_tour_id"] == 9999

    def test_cluster_real_data(self, plan_route_full):
        """Trailabschnitte aus echten Daten clustern."""
        embedded = plan_route_full["_embedded"]
        segs = extract_trail_segments(
            embedded["way_types"]["items"],
            embedded["coordinates"]["items"],
            7777,
        )
        clusters = cluster_trail_segments(segs, max_gap_km=0.2)
        assert len(clusters) >= 1
        for c in clusters:
            assert c["total_length_km"] > 0
            assert isinstance(c["segments"], int)
            assert c["source_tour_ids"] == [7777]

    def test_hotspot_real_data_single_tour(self):
        """Einzelne Tour erzeugt keinen Hotspot bei min_overlap=2."""
        path = Path(__file__).parent / "fixtures" / "plan_route_full.json"
        data = json.loads(path.read_text())
        embedded = data["_embedded"]
        segs = extract_trail_segments(
            embedded["way_types"]["items"],
            embedded["coordinates"]["items"],
            8888,
        )
        hotspots = discover_trail_hotspots(segs, min_overlap=2)
        assert hotspots == []  # single tour, no multi-tour overlap


# -- Tests: derive_routing_waypoints -----------------------------------

class TestDeriveRoutingWaypoints:
    """derive_routing_waypoints: clusters/hotspots -> [[lat, lng], ...] for plan_route."""

    def _make_cluster(self, start_lat, start_lng, end_lat, end_lng, length_km=1.0, tid=1001):
        return {
            "segments": 1, "total_length_km": length_km,
            "source_tour_ids": [tid], "way_types": ["trail_d1"],
            "start": {"lat": start_lat, "lng": start_lng},
            "end": {"lat": end_lat, "lng": end_lng},
        }

    def _make_hotspot(self, lat, lng, tids=None):
        return {
            "center": {"lat": lat, "lng": lng},
            "overlapping_segments": 2,
            "source_tour_ids": tids or [1001, 1002],
            "way_types": ["trail_d1", "trail_d2"],
        }

    # -- Entry+Exit basics -------------------------------------------

    def test_one_cluster_returns_entry_exit(self):
        """Ein Cluster -> 2 Waypoints (Entry + Exit)."""
        cl = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5)]
        wps = derive_routing_waypoints(cl)
        assert len(wps) == 2
        assert wps[0] == [48.0, 10.0]        # entry
        assert wps[1] == [48.004, 10.004]     # exit

    def test_two_clusters_two_pairs(self):
        """Zwei getrennte Cluster -> 4 Waypoints (2 Paare)."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5, tid=1001),
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.3, tid=1002),
        ]
        wps = derive_routing_waypoints(cls)
        assert len(wps) == 4
        assert wps[0] == [48.0, 10.0]        # entry of longest cluster
        assert wps[1] == [48.004, 10.004]     # exit of longest cluster
        assert wps[2] == [48.1, 10.0]         # entry of second cluster
        assert wps[3] == [48.104, 10.004]     # exit of second cluster

    def test_four_clusters_eight_waypoints(self):
        """Vier getrennte Cluster -> 8 Waypoints (4 Paare, default max_points=8)."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5, tid=1001),
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.5, tid=1002),
            self._make_cluster(48.2, 10.0, 48.204, 10.004, length_km=0.5, tid=1003),
            self._make_cluster(48.3, 10.0, 48.304, 10.004, length_km=0.5, tid=1004),
        ]
        wps = derive_routing_waypoints(cls)
        assert len(wps) == 8  # 4 clusters x 2 = 8, fits default max_points=8

    def test_long_cluster_gets_pair(self):
        """Cluster > 1 km bekommt Entry+Exit-Paar."""
        cl = [self._make_cluster(48.0, 10.0, 48.02, 10.02, length_km=2.5)]
        wps = derive_routing_waypoints(cl)
        assert len(wps) == 2
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.02, 10.02]

    # -- max_points edge cases ---------------------------------------

    def test_max_points_2(self):
        """max_points=2 -> nur das lngste Cluster bekommt ein Paar."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=1.0, tid=1001),
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.5, tid=1002),
        ]
        wps = derive_routing_waypoints(cls, max_points=2)
        assert len(wps) == 2
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.004, 10.004]

    def test_max_points_3(self):
        """max_points=3 -> 1 Paar (2 Pts) + 1 Centroid (1 Pt)."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=1.0, tid=1001),
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.5, tid=1002),
        ]
        wps = derive_routing_waypoints(cls, max_points=3)
        assert len(wps) == 3
        assert wps[0] == [48.0, 10.0]        # entry of longest
        assert wps[1] == [48.004, 10.004]     # exit of longest
        assert wps[2] == [48.102, 10.002]     # centroid of second

    def test_default_max_points_8_six_clusters(self):
        """Default max_points=8, 6 Cluster -> 4 Paare = 8 Pts."""
        cls = [
            self._make_cluster(48.0 + i * 0.1, 10.0, 48.004 + i * 0.1, 10.004,
                               length_km=0.5, tid=1000 + i)
            for i in range(6)
        ]
        wps = derive_routing_waypoints(cls)
        assert len(wps) == 8  # 4 pairs fit in max_points=8, 2 clusters skipped

    def test_max_points_10(self):
        """max_points=10, 10 Cluster -> 5 Paare = 10 Pts."""
        cls = [
            self._make_cluster(48.0 + i * 0.1, 10.0, 48.004 + i * 0.1, 10.004,
                               length_km=0.5, tid=1000 + i)
            for i in range(10)
        ]
        wps = derive_routing_waypoints(cls, max_points=10)
        assert len(wps) == 10  # 5 pairs

    def test_max_points_15(self):
        """max_points=15, 15 Cluster -> 7 Paare (14 Pts) + 1 Centroid (1 Pt)."""
        cls = [
            self._make_cluster(48.0 + i * 0.08, 10.0, 48.004 + i * 0.08, 10.004,
                               length_km=0.5, tid=1000 + i)
            for i in range(15)
        ]
        wps = derive_routing_waypoints(cls, max_points=15)
        assert len(wps) == 15  # 7 pairs (14) + 1 centroid (1) = 15

    def test_max_points_zero_returns_empty(self):
        """max_points=0 -> leere Liste."""
        cl = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5)]
        assert derive_routing_waypoints(cl, max_points=0) == []

    def test_max_points_negative_returns_empty(self):
        """max_points=-1 -> leere Liste."""
        cl = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5)]
        assert derive_routing_waypoints(cl, max_points=-1) == []

    # -- Hotspots ----------------------------------------------------

    def test_hotspot_added_after_pairs(self):
        """1 Cluster + 1 Hotspot -> Paar (2 Pts) + Hotspot (1 Pt) = 3."""
        cls = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5, tid=1001)]
        hots = [self._make_hotspot(48.05, 10.05)]
        wps = derive_routing_waypoints(cls, hots, max_points=3)
        assert len(wps) == 3
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.004, 10.004]
        assert wps[2] == [48.05, 10.05]  # hotspot is last

    def test_hotspot_too_close_skipped(self):
        """Hotspot am Einstiegspunkt -> zu nah -> bersprungen."""
        cls = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5, tid=1001)]
        # Hotspot exactly at entry -> 0 km distance -> skipped
        hots = [self._make_hotspot(48.0, 10.0)]
        wps = derive_routing_waypoints(cls, hots, max_points=3)
        assert len(wps) == 2  # only cluster pair, hotspot skipped

    # -- Sorting & diversity -----------------------------------------

    def test_sort_by_length(self):
        """Cluster werden nach Lnge sortiert (lngste zuerst)."""
        cls = [
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.1, tid=1001),
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=2.0, tid=1002),
            self._make_cluster(48.2, 10.0, 48.203, 10.003, length_km=0.5, tid=1003),
        ]
        wps = derive_routing_waypoints(cls, max_points=5)
        # 2.0 km cluster gets pair first -> (48.0, 10.0) + (48.004, 10.004)
        # 0.5 km cluster gets pair next  -> (48.2, 10.0) + (48.203, 10.003)
        # 0.1 km cluster gets centroid    -> (48.102, 10.002)
        assert len(wps) == 5
        assert wps[0] == [48.0, 10.0]        # entry of longest cluster
        assert wps[1] == [48.004, 10.004]     # exit of longest cluster

    def test_spatial_diversity_partial_overlap(self):
        """Teilberlappende Cluster (verschiedene Ein-/Ausstiege) -> beide Paare."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=1.0, tid=1001),
            self._make_cluster(48.001, 10.001, 48.003, 10.003, length_km=0.5, tid=1002),
        ]
        wps = derive_routing_waypoints(cls, max_points=5)
        # Entry1=(48.0, 10.0), Exit1=(48.004, 10.004) => intra-pair ~0.53 km
        # Entry2=(48.001, 10.001) is 0.16 km from Entry1 > 0.1 -> diverse
        # Exit2=(48.003, 10.003) is 0.16 km from Exit1 > 0.1 -> diverse
        # Both pairs included (4 waypoints)
        assert len(wps) == 4

    def test_spatial_diversity_overlapping_clusters(self):
        """Zwei Cluster mit identischem Start/Ende -> nur erstes Paar."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.001, 10.001, length_km=0.15, tid=1001),
            self._make_cluster(48.0, 10.0, 48.001, 10.001, length_km=0.15, tid=1002),
        ]
        wps = derive_routing_waypoints(cls, max_points=5)
        # Zweites Cluster: Entry/Exit identisch -> 0 km -> nicht diverse
        # Centroid (48.0005, 10.0005): ~0.078 km < min_diversity=0.1 -> nicht diverse
        # -> komplett berschlagen
        assert len(wps) == 2
        assert wps[0] == [48.0, 10.0]
        assert wps[1] == [48.001, 10.001]

    def test_many_clusters_fewer_than_max(self):
        """15 Cluster, max_points=10 -> 5 Paare = 10 Pts, alle diverse."""
        cls = [
            self._make_cluster(48.0 + i * 0.08, 10.0, 48.004 + i * 0.08, 10.004,
                               length_km=0.8, tid=1000 + i)
            for i in range(15)
        ]
        wps = derive_routing_waypoints(cls, max_points=10)
        assert len(wps) == 10  # 5 pairs = 10 points
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                d = _haversine_km(wps[i][0], wps[i][1], wps[j][0], wps[j][1])
                assert d >= 0.08, f"Waypoints {i} and {j} too close: {d*1000:.0f}m"

    # -- Format & invariants -----------------------------------------

    def test_empty_clusters(self):
        """Leere Liste -> []."""
        assert derive_routing_waypoints([]) == []
        assert derive_routing_waypoints([], [{"center": {"lat": 48.0, "lng": 10.0}}]) == []

    def test_output_format_plan_route_ready(self):
        """Waypoints sind [lat, lng] - Format fr plan_route(coordinates=...)."""
        cl = [self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5)]
        wps = derive_routing_waypoints(cl)
        assert isinstance(wps, list)
        assert len(wps) == 2
        for wp in wps:
            assert isinstance(wp, list)
            assert len(wp) == 2
            assert isinstance(wp[0], float)
            assert isinstance(wp[1], float)
            assert -90 <= wp[0] <= 90
            assert -180 <= wp[1] <= 180

    def test_no_filler_points(self):
        """2 Cluster + max_points=8 -> 2 Paare = 4 Pts, keine Filler."""
        cls = [
            self._make_cluster(48.0, 10.0, 48.004, 10.004, length_km=0.5, tid=1001),
            self._make_cluster(48.1, 10.0, 48.104, 10.004, length_km=0.3, tid=1002),
        ]
        wps = derive_routing_waypoints(cls, max_points=8)
        assert len(wps) == 4  # 2 pairs, no artificial filling