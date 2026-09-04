"""Tests for tour_search — pure functions only (no live API).

Covers:
  - _is_roundtrip: roundtrip detection with relative tolerance
  - _cross_track_distance_km / _point_to_line_segment_distance_km
  - _generate_corridor_centers
  - _deduplicate_tours
  - _make_compact_result
  - search_tours mode dispatch via FakeMCP + FakeClient
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from komoot_mcp.tools.tour_search import (
    _cross_track_distance_km,
    _deduplicate_tours,
    _generate_corridor_centers,
    _is_roundtrip,
    _make_compact_result,
    _point_to_line_segment_distance_km,
    register,
)
from komoot_mcp.tools.trail_discovery import _haversine_km
from conftest import FakeClient, FakeMCP, load_fixture

# ── Test helpers ────────────────────────────────────────────────────────

_LAT_BASE = 48.0
_LNG_BASE = 10.0


def _make_item(
    tour_id: int = 1001,
    name: str = "Test Tour",
    sport: str = "mtb",
    distance: float = 12000.0,
    elevation_up: float = 400.0,
    duration: int = 3600,
    **overrides: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": tour_id,
        "name": name,
        "sport": sport,
        "type": "tour_recorded",
        "status": "public",
        "distance": distance,
        "elevation_up": elevation_up,
        "duration": duration,
        "lat": 48.0,
        "lng": 10.0,
    }
    item.update(overrides)
    return item


def _tours_response(tour_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mock tours API response in the correct HAL format."""
    return {
        "_embedded": {"tours": tour_items},
        "page": {"size": len(tour_items), "totalElements": len(tour_items), "totalPages": 1, "number": 0},
    }


# ── Tests: _is_roundtrip ───────────────────────────────────────────────

class TestIsRoundtrip:
    def test_perfect_loop(self):
        """Same start and end point -> always roundtrip."""
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.0, 10.0, 20.0)
        assert is_rt is True
        assert dev == 0.0

    def test_small_deviation_5km_tour(self):
        """5 km tour, end 0.3 km from start -> roundtrip (tol = 0.5 km)."""
        # 0.3 km at 48N is roughly 0.0027 deg lat or 0.004 deg lng
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.002, 10.001, 5.0)
        assert is_rt is True
        assert dev < 0.5

    def test_small_deviation_40km_tour(self):
        """40 km tour, end 0.3 km from start -> roundtrip (tol = 2.0 km)."""
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.003, 10.001, 40.0)
        assert is_rt is True

    def test_large_deviation_40km_tour(self):
        """40 km tour, end 10 km from start -> NOT roundtrip."""
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.1, 10.05, 40.0)
        assert is_rt is False
        assert dev > 2.0

    def test_relative_tolerance_scales(self):
        """100 km tour can tolerate a larger start-end gap than 5 km."""
        # 4 km deviation -> for 5 km tour: 4 > 0.5 -> not roundtrip
        #                   for 100 km tour: 4 < 5.0 -> roundtrip
        lat_offset = 4.0 / 111.0  # ~4 km
        is_rt_short, _ = _is_roundtrip(48.0, 10.0, 48.0 + lat_offset, 10.0, 5.0)
        is_rt_long, _ = _is_roundtrip(48.0, 10.0, 48.0 + lat_offset, 10.0, 100.0)
        assert is_rt_short is False
        assert is_rt_long is True

    def test_zero_length_tour(self):
        """0 km tour falls back to min tolerance 0.5 km."""
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.004, 10.0, 0.0)
        # 0.004 deg lat ~ 0.44 km -> within 0.5 km tolerance
        assert is_rt is True
        assert dev == pytest.approx(0.444, abs=0.01)

    def test_boundary_exact_tolerance(self):
        """Start-end distance within tolerance boundary -> roundtrip."""
        # tolerance = max(10 * 0.05, 0.5) = 0.5
        # Use start-end ~0.45 km to stay clearly within 0.5 km tolerance
        lat_delta = 0.004 / 1.0  # ~0.444 km at 48N
        is_rt, dev = _is_roundtrip(48.0, 10.0, 48.0 + lat_delta, 10.0, 10.0)
        assert is_rt is True
        assert dev == pytest.approx(0.444, abs=0.01)


# ── Tests: cross-track distance ─────────────────────────────────────────

class TestCrossTrackDistance:
    def test_point_on_line(self):
        """Point exactly on line AB -> 0 distance."""
        d = _cross_track_distance_km(48.0, 10.0, 48.0, 10.0, 48.0, 11.0)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_point_midway_between_endpoints(self):
        """Point on a line North of the corridor midpoint."""
        # Corridor: (48.0, 10.0) -> (48.0, 10.1) (east-west along 48N)
        # Point: (48.01, 10.05) -> ~1.11 km north of the line
        d = _cross_track_distance_km(48.01, 10.05, 48.0, 10.0, 48.0, 10.1)
        # ~1.11 km north (1 deg lat = 111 km, 0.01 deg = 1.11 km)
        assert d == pytest.approx(1.11, abs=0.01)

    def test_point_beyond_endpoint(self):
        """Cross-track may be large for points far from the line."""
        d = _cross_track_distance_km(49.0, 10.05, 48.0, 10.0, 48.0, 10.1)
        assert d > 100  # ~111 km north

    def test_zero_length_line(self):
        """Start == end -> distance to that single point."""
        d = _cross_track_distance_km(48.01, 10.0, 48.0, 10.0, 48.0, 10.0)
        # This should be ~1.11 km (haversine from 48.0 to 48.01)
        assert d == pytest.approx(1.11, abs=0.01)


class TestPointToLineSegmentDistance:
    def test_point_on_segment(self):
        """Point exactly on the line segment -> 0."""
        d = _point_to_line_segment_distance_km(48.0, 10.05, 48.0, 10.0, 48.0, 10.1)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_point_before_start(self):
        """Point before the start of segment -> distance to start."""
        d = _point_to_line_segment_distance_km(48.0, 9.9, 48.0, 10.0, 48.0, 10.1)
        # Should be ~11.1 km (0.1 deg longitude at 48N)
        assert d == pytest.approx(7.44, abs=0.1)  # ~7.44 km

    def test_point_after_end(self):
        """Point after the end of segment -> distance to end."""
        d = _point_to_line_segment_distance_km(48.0, 10.2, 48.0, 10.0, 48.0, 10.1)
        # Should be ~7.44 km (0.1 deg longitude at 48N)
        assert d == pytest.approx(7.44, abs=0.1)

    def test_point_off_to_the_side(self):
        """Point off the side of the segment -> perpendicular distance."""
        # Segment: (48.0, 10.0) -> (48.01, 10.0) (~1.11 km north)
        # Point: (48.005, 10.005) -> offset ~0.37 km east of segment midpoint
        d = _point_to_line_segment_distance_km(48.005, 10.005, 48.0, 10.0, 48.01, 10.0)
        assert d == pytest.approx(0.37, abs=0.02)


# ── Tests: _generate_corridor_centers ──────────────────────────────────

class TestGenerateCorridorCenters:
    def test_zero_distance(self):
        """Start == end -> single center point."""
        centers = _generate_corridor_centers(48.0, 10.0, 48.0, 10.0, 5.0)
        assert len(centers) == 1
        assert centers[0] == (48.0, 10.0)

    def test_short_distance(self):
        """~10 km corridor with 5 km radius -> 2-3 centers."""
        centers = _generate_corridor_centers(48.0, 10.0, 48.0, 10.1, 5.0)
        assert 2 <= len(centers) <= 4

    def test_medium_distance(self):
        """~50 km corridor with 5 km radius -> ~12 centers capped at 8."""
        centers = _generate_corridor_centers(48.0, 10.0, 48.0, 10.5, 5.0)
        assert 2 <= len(centers) <= 8

    def test_endpoints_match(self):
        """First center = start, last center = end."""
        centers = _generate_corridor_centers(47.7, 9.75, 47.65, 9.6, 3.0)
        assert len(centers) >= 2
        assert centers[0] == (pytest.approx(47.7, abs=0.001), pytest.approx(9.75, abs=0.001))
        assert centers[-1] == (pytest.approx(47.65, abs=0.001), pytest.approx(9.6, abs=0.001))

    def test_minimum_two_centers(self):
        """Even very short corridor gets at least 2 centers."""
        centers = _generate_corridor_centers(48.0, 10.0, 48.001, 10.001, 1.0)
        assert len(centers) >= 2


# ── Tests: _deduplicate_tours ──────────────────────────────────────────

class TestDeduplicateTours:
    def test_no_duplicates(self):
        """Distinct IDs remain unchanged."""
        items = [_make_item(1), _make_item(2), _make_item(3)]
        result = _deduplicate_tours(items)
        assert len(result) == 3

    def test_with_duplicates(self):
        """Duplicate IDs are removed, first occurrence preserved."""
        items = [_make_item(1, name="first"), _make_item(2), _make_item(1, name="duplicate")]
        result = _deduplicate_tours(items)
        assert len(result) == 2
        assert result[0]["name"] == "first"

    def test_empty_list(self):
        """Empty input -> empty output."""
        assert _deduplicate_tours([]) == []

    def test_all_same_id(self):
        """All same ID -> single result."""
        items = [_make_item(42, name="a"), _make_item(42, name="b"), _make_item(42, name="c")]
        result = _deduplicate_tours(items)
        assert len(result) == 1
        assert result[0]["name"] == "a"


# ── Tests: _make_compact_result ────────────────────────────────────────

class TestMakeCompactResult:
    def test_basic_fields(self):
        """Core fields present: tour_id, name, sport, type, distance_km."""
        item = _make_item(1001, name="Alpine Trail", sport="mtb")
        result = _make_compact_result(item)
        assert result["tour_id"] == 1001
        assert result["name"] == "Alpine Trail"
        assert result["sport"] == "mtb"
        assert result["distance_km"] == 12.0

    def test_distance_conversion(self):
        """Distance in meters -> km, rounded to 2 places."""
        item = _make_item(distance=33450)
        result = _make_compact_result(item)
        assert result["distance_km"] == 33.45

    def test_distance_from_center(self):
        """When provided, includes distance_from_center_km."""
        result = _make_compact_result(_make_item(), distance_from_center_km=3.456)
        assert result["distance_from_center_km"] == 3.456

    def test_roundtrip_score(self):
        """When provided, includes roundtrip fields."""
        score = {"is_roundtrip": True, "deviation_km": 0.123}
        result = _make_compact_result(_make_item(), roundtrip_score=score)
        assert result["is_roundtrip"] is True
        assert result["roundtrip_deviation_km"] == 0.123

    def test_start_end_distances(self):
        """When provided, includes start/end distance fields."""
        result = _make_compact_result(
            _make_item(),
            start_distance_km=0.5,
            end_distance_km=0.8,
        )
        assert result["start_distance_km"] == 0.5
        assert result["end_distance_km"] == 0.8

    def test_corridor_distance(self):
        """When provided, includes corridor_distance_km."""
        result = _make_compact_result(_make_item(), corridor_distance_km=1.234)
        assert result["corridor_distance_km"] == 1.234

    def test_no_spurious_fields(self):
        """Without optional args, no extra distance fields appear."""
        result = _make_compact_result(_make_item())
        assert "distance_from_center_km" not in result
        assert "roundtrip_deviation_km" not in result
        assert "is_roundtrip" not in result
        assert "start_distance_km" not in result
        assert "end_distance_km" not in result
        assert "corridor_distance_km" not in result

    def test_elevation_and_duration_preserved(self):
        """elevation_up, elevation_down, duration preserved."""
        item = _make_item(elevation_up=500, elevation_down=480, duration=7200)
        result = _make_compact_result(item)
        assert result["elevation_up"] == 500
        assert result["elevation_down"] == 480
        assert result["duration"] == 7200

    def test_missing_distance(self):
        """None or missing distance -> 0."""
        result = _make_compact_result({})
        assert result["distance_km"] == 0.0


# ── Tests: search mode dispatch via FakeMCP + FakeClient ────────────────

class FakeClientSearch:
    """Fake client that captures requests and returns canned data."""

    def __init__(self) -> None:
        self.user_id = "test_user"
        self.last_get_path: str | None = None
        self.last_get_params: dict[str, Any] = {}
        self._get_responses: dict[str, dict[str, Any]] = {}
        self._call_count = 0
        self._get_calls: list[tuple[str, dict[str, Any]]] = []

    def set_get_response(self, path: str, response: dict[str, Any]) -> None:
        self._get_responses[path] = response

    async def get(self, path: str, **params: Any) -> Any:
        self._call_count += 1
        self._get_calls.append((path, params))
        self.last_get_path = path
        self.last_get_params = params

        # Try exact match first, then prefix match
        if path in self._get_responses:
            return self._get_responses[path]

        # Extract tour_id from /tours/{id} paths
        if path.startswith("/tours/"):
            return self._get_responses.get("/tours/", {"_embedded": {"coordinates": {"items": []}}})

        # Default: empty results in correct HAL format
        return {"_embedded": {"tours": []}, "page": {"size": 0, "totalElements": 0, "totalPages": 0, "number": 0}}

    @property
    def call_count(self) -> int:
        return self._call_count


@pytest.fixture
def fake_client() -> FakeClientSearch:
    return FakeClientSearch()


@pytest.fixture
def registered(
    fake_client: FakeClientSearch,
) -> tuple[Any, FakeClientSearch]:
    """Register search_tours with a FakeMCP + FakeClientSearch."""
    mcp = FakeMCP()
    register(mcp, fake_client)  # type: ignore[arg-type]
    return mcp, fake_client


def _run(coro: Any) -> Any:
    import asyncio
    return asyncio.run(coro)


class TestModeDispatch:
    def test_no_params_returns_error(self, registered: tuple[Any, FakeClientSearch]):
        mcp, _fc = registered
        fn = mcp.tools["search_tours"]
        result = _run(fn())
        assert result["status"] == "error"
        assert "center_lat" in result["message"]

    def test_radius_mode(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001), _make_item(1002)]),
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(center_lat=48.0, center_lng=10.0, radius_km=10.0))
        assert result["status"] == "success"
        assert result["search_mode"] == "radius"
        assert result["total_found"] == 2

    def test_radius_mode_passes_params(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=47.7, center_lng=9.75, radius_km=15.0, sport="mtb"))
        params = fc.last_get_params
        assert params["center"] == "47.7,9.75"
        assert params["max_distance"] == 15000
        assert params["sport_types"] == "mtb"
        assert params["sort_field"] == "proximity"
        assert params["sort_direction"] == "asc"

    def test_radius_mode_with_tour_type(
        self, registered: tuple[Any, FakeClientSearch]
    ):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(
            center_lat=48.0, center_lng=10.0, radius_km=5.0,
            tour_type="tour_recorded",
        ))
        assert fc.last_get_params.get("type") == "tour_recorded"

    def test_roundtrip_mode(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        # Respond to radius search with 1 candidate
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        # Respond to get_tour with a roundtrip (same start/end coords)
        fc.set_get_response(
            "/tours/",
            {
                "id": 1001,
                "distance": 15000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": 48.0, "lng": 10.0},
                            {"lat": 48.005, "lng": 10.002},
                            {"lat": 48.0, "lng": 10.0},
                        ]
                    }
                },
            },
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=10.0, route_type="roundtrip",
        ))
        assert result["status"] == "success"
        assert result["search_mode"] == "roundtrip"
        assert result["total_found"] == 1
        assert result["results"][0]["is_roundtrip"] is True

    def test_start_to_end_mode(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fc.set_get_response(
            "/tours/",
            {
                "id": 1001,
                "distance": 25000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": 47.705, "lng": 9.755},
                            {"lat": 47.700, "lng": 9.750},
                            {"lat": 47.65, "lng": 9.60},
                        ]
                    }
                },
            },
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            start_lat=47.7059, start_lng=9.7565,
            end_lat=47.65, end_lng=9.60,
        ))
        assert result["status"] == "success"
        assert result["search_mode"] == "start_to_end"
        assert result["total_found"] == 1
        assert "start_distance_km" in result["results"][0]
        assert "end_distance_km" in result["results"][0]

    def test_corridor_mode(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        # Corridor mode makes multiple radius searches
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fc.set_get_response(
            "/tours/",
            {
                "id": 1001,
                "distance": 20000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": 47.705, "lng": 9.756},
                            {"lat": 47.700, "lng": 9.730},
                            {"lat": 47.680, "lng": 9.660},
                            {"lat": 47.650, "lng": 9.600},
                        ]
                    }
                },
            },
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            start_lat=47.7059, start_lng=9.7565,
            end_lat=47.65, end_lng=9.60,
            corridor_km=5.0,
        ))
        assert result["status"] == "success"
        assert result["search_mode"] == "corridor"
        assert "corridor_centers" in result
        assert result["corridor_centers"] >= 2


class TestPagination:
    def test_page_param_passed(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0, page=2))
        assert fc.last_get_params.get("page") == 2

    def test_limit_capped(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0, limit=200))
        assert fc.last_get_params.get("limit") == 50

    def test_limit_default(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0))
        assert fc.last_get_params.get("limit") == 20


class TestEmptyResults:
    def test_radius_empty(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([]),
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0))
        assert result["status"] == "success"
        assert result["total_found"] == 0

    def test_roundtrip_empty_results(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([]),
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=5.0, route_type="roundtrip",
        ))
        assert result["status"] == "success"
        assert result["total_found"] == 0

    def test_start_end_empty(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([]),
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            start_lat=48.0, start_lng=10.0,
            end_lat=48.1, end_lng=10.1,
        ))
        assert result["status"] == "success"
        assert result["total_found"] == 0


class TestSportTypes:
    def test_default_mtb(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0))
        assert fc.last_get_params.get("sport_types") == "mtb"

    def test_e_mtb(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=5.0, sport="e_mtb",
        ))
        assert fc.last_get_params.get("sport_types") == "e_mtb"

    def test_multiple_sports(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=5.0, sport="mtb,touringbicycle",
        ))
        assert fc.last_get_params.get("sport_types") == "mtb,touringbicycle"


class TestTourTypeFilter:
    def test_tour_planned(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=5.0, tour_type="tour_planned",
        ))
        assert fc.last_get_params.get("type") == "tour_planned"

    def test_tour_recorded(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(
            center_lat=48.0, center_lng=10.0,
            radius_km=5.0, tour_type="tour_recorded",
        ))
        assert fc.last_get_params.get("type") == "tour_recorded"

    def test_no_type_filter(self, registered: tuple[Any, FakeClientSearch]):
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fn = mcp.tools["search_tours"]
        _run(fn(center_lat=48.0, center_lng=10.0, radius_km=5.0))
        assert "type" not in fc.last_get_params


class TestDeduplicationInCorridor:
    def test_corridor_dedup(self, registered: tuple[Any, FakeClientSearch]):
        """Corridor search with duplicate tour IDs across centers."""
        mcp, fc = registered
        # Both center searches return the same tour
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        fc.set_get_response(
            "/tours/",
            {
                "id": 1001,
                "distance": 15000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": 47.705, "lng": 9.756},
                            {"lat": 47.680, "lng": 9.700},
                        ]
                    }
                },
            },
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            start_lat=47.7059, start_lng=9.7565,
            end_lat=47.65, end_lng=9.60,
            corridor_km=5.0,
        ))
        # Should only have 1 result despite multiple API calls
        assert result["total_found"] == 1
        # Should have made N+1 calls (N center searches + 1 get_tour)
        assert fc.call_count >= 3


class TestRoundtripScoring:
    def test_non_roundtrip_filtered(self, registered: tuple[Any, FakeClientSearch]):
        """route_type=roundtrip filters out non-roundtrips."""
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001)]),
        )
        # Tour whose start and end are far apart (10+ km)
        fc.set_get_response(
            "/tours/",
            {
                "id": 1001,
                "distance": 30000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": 48.0, "lng": 10.0},
                            {"lat": 48.02, "lng": 10.02},
                            {"lat": 48.1, "lng": 10.1},
                        ]
                    }
                },
            },
        )
        fn = mcp.tools["search_tours"]
        result = _run(fn(
            center_lat=48.05, center_lng=10.05,
            radius_km=20.0, route_type="roundtrip",
        ))
        # Should be filtered out (not a roundtrip)
        assert result["total_found"] == 0


class TestStartEndRanking:
    def test_results_sorted_by_combined_distance(
        self, registered: tuple[Any, FakeClientSearch]
    ):
        """Start-to-end results sorted by (start_dist, end_dist)."""
        mcp, fc = registered
        fc.set_get_response(
            f"/users/{fc.user_id}/tours/",
            _tours_response([_make_item(1001), _make_item(1002)]),
        )

        def get_response_for_tour(tid: int, s_lat: float, s_lng: float,
                                   e_lat: float, e_lng: float) -> dict:
            return {
                "id": tid,
                "distance": 20000,
                "_embedded": {
                    "coordinates": {
                        "items": [
                            {"lat": s_lat, "lng": s_lng},
                            {"lat": e_lat, "lng": e_lng},
                        ]
                    }
                },
            }

        # Keep the radius-search response, add individual tour responses
        fc._get_responses[f"/tours/1001"] = get_response_for_tour(1001, 47.706, 9.757, 47.65, 9.60)  # closer start
        fc._get_responses[f"/tours/1002"] = get_response_for_tour(1002, 47.71, 9.76, 47.65, 9.60)   # further start

        fn = mcp.tools["search_tours"]
        result = _run(fn(
            start_lat=47.7059, start_lng=9.7565,
            end_lat=47.65, end_lng=9.60,
        ))

        assert result["total_found"] == 2
        # First result should have smaller start_distance_km
        assert (
            result["results"][0]["start_distance_km"]
            <= result["results"][1]["start_distance_km"]
        )


class TestInvalidParameters:
    def test_missing_radius(self, registered: tuple[Any, FakeClientSearch]):
        mcp, _fc = registered
        fn = mcp.tools["search_tours"]
        result = _run(fn(center_lat=48.0, center_lng=10.0))  # no radius
        assert result["status"] == "error"

    def test_missing_center_lat(self, registered: tuple[Any, FakeClientSearch]):
        mcp, _fc = registered
        fn = mcp.tools["search_tours"]
        result = _run(fn(center_lng=10.0, radius_km=5.0))
        assert result["status"] == "error"

    def test_missing_end(self, registered: tuple[Any, FakeClientSearch]):
        mcp, _fc = registered
        fn = mcp.tools["search_tours"]
        result = _run(fn(start_lat=48.0, start_lng=10.0))
        assert result["status"] == "error"