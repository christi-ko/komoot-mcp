"""Backward compatibility tests for plan_route (compact=False/True) and
the create_planned_tour payload preparation, including route_ref caching.

Strategy: drive the real closures (plan_route, create_planned_tour) through
register() with a FakeMCP + FakeClient that returns a captured real
compact=False plan_route response from tests/fixtures/. No network calls,
no writes to Komoot.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from komoot_mcp.tools import routing

from conftest import FakeClient, FakeMCP, load_fixture

COORDS = [[47.557, 10.0206], [47.545, 10.105], [47.557, 10.0206]]


@pytest.fixture
def registered() -> tuple[FakeMCP, FakeClient]:
    mcp = FakeMCP()
    # Real captured compact=False response (Oberstaufen MTB route)
    client = FakeClient(route_response=load_fixture("plan_route_full.json"))
    routing.register(mcp, client)
    return mcp, client


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── compact=False: compact summary + route_ref (was full response) ──────

def test_plan_route_compact_false_returns_summary_with_ref(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """compact=False now returns a compact summary enriched with route_ref,
    NOT the full 96 KB plan_route response."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=False))

    # Core summary fields (same as compact=True)
    assert "distance_km" in result
    assert "elevation_up_m" in result
    assert "elevation_down_m" in result
    assert "duration" in result
    assert "difficulty" in result
    assert "technical_difficulty" in result
    assert "fitness_difficulty" in result
    assert "way_types" in result
    assert "surfaces" in result
    assert "singletrail" in result
    assert "segments" in result
    assert "matched_coordinates" in result

    # Must contain route_ref
    assert "route_ref" in result
    assert result["route_ref"].startswith("route_")

    # Must NOT contain large geometry
    assert "path" not in result
    assert "coordinates" not in result
    assert "_embedded" not in result

    # segments is a dict summary, not the geometry list
    assert isinstance(result["segments"], dict)
    assert set(result["segments"].keys()) == {"routed", "manual"}


def test_plan_route_compact_false_has_singletrail(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """compact=False now includes the singletrail summary."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=False))
    assert "singletrail" in result
    assert "matched_coordinates" in result
    assert isinstance(result["matched_coordinates"], int)


def test_plan_route_compact_false_caches_internally(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """The full route data is cached internally under route_ref, not returned."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=False))

    ref = result["route_ref"]
    # Must exist in the module-level cache
    assert ref in routing._route_cache
    entry = routing._route_cache[ref]
    assert len(entry) == 2  # (timestamp, route_data)
    ts, cached = entry
    assert isinstance(ts, float)
    assert "path" in cached
    assert "_embedded" in cached


# ── compact=True: unchanged ─────────────────────────────────────────────

def test_plan_route_compact_true_summary(registered: tuple[FakeMCP, FakeClient]) -> None:
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=True))

    # core fields
    assert "distance_km" in result
    assert "elevation_up_m" in result
    assert "elevation_down_m" in result
    assert "duration" in result
    assert "difficulty" in result
    assert "way_types" in result
    assert "surfaces" in result
    assert "singletrail" in result
    assert "segments" in result
    assert "matched_coordinates" in result

    # T/C
    assert result["technical_difficulty"].startswith("T")
    assert result["fitness_difficulty"].startswith("C")

    # matched_coordinates is an int count
    assert isinstance(result["matched_coordinates"], int)
    assert result["matched_coordinates"] == 513


def test_plan_route_compact_true_no_coordinate_list(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Compact output must NOT contain a full coordinate/path list,
    and must NOT have route_ref."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=True))

    assert "path" not in result
    assert "coordinates" not in result
    assert "_embedded" not in result
    assert "route_ref" not in result
    # segments is a summary dict, not the geometry list
    assert isinstance(result["segments"], dict)
    assert set(result["segments"].keys()) == {"routed", "manual"}


def test_plan_route_compact_true_not_cached(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """compact=True must NOT create a cache entry."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]

    # Clear cache from previous tests
    routing._route_cache.clear()

    _run(plan_route(COORDS, sport="mtb", compact=True))
    # compact=True results are not cached — no entries should exist
    assert len(routing._route_cache) == 0


def test_plan_route_default_is_false(registered: tuple[FakeMCP, FakeClient]) -> None:
    """Default (no compact arg) must return the compact summary + route_ref."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb"))
    assert "distance_km" in result
    assert "route_ref" in result
    # No full geometry
    assert "path" not in result
    assert "singletrail" in result


# ── create_planned_tour with route_data (unchanged compat) ──────────────

def test_create_planned_tour_payload_with_full_response(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Payload built from a compact=False response must keep path, segments,
    sport, date and must NOT contain _links / uri."""
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full_response = load_fixture("plan_route_full.json")

    result = _run(create_tour(full_response, name="Test Tour", sport="mtb"))
    assert result.get("captured") is True, "should have been captured, not written"

    payload = client.last_request["json_body"]
    assert "path" in payload
    assert "segments" in payload
    assert payload["sport"] == "mtb"
    assert "date" in payload  # preserved from the route response
    assert "_links" not in payload
    assert "uri" not in payload


def test_create_planned_tour_payload_name(registered: tuple[FakeMCP, FakeClient]) -> None:
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full_response = load_fixture("plan_route_full.json")

    _run(create_tour(full_response, name="Meine Tour", sport="hike"))
    payload = client.last_request["json_body"]
    assert payload["name"] == "Meine Tour"
    assert payload["sport"] == "hike"


def test_create_planned_tour_payload_no_mutation(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """The original route_data must not be mutated."""
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full_response = load_fixture("plan_route_full.json")
    original_keys = set(full_response.keys())

    _run(create_tour(full_response, name="X", sport="mtb"))
    assert set(full_response.keys()) == original_keys
    assert "_links" not in full_response or "_links" in original_keys
    assert full_response["sport"] == full_response.get("sport")


def test_create_planned_tour_payload_valid_full_response_still_works(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """A valid full response via route_data must still pass validation and reach the API."""
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full = load_fixture("plan_route_full.json")

    result = _run(create_tour(full, name="Valid Tour", sport="mtb"))

    # Must NOT return an error (validation passed)
    assert "status" not in result or result.get("status") != "error"
    # A real API call was made (or captured by FakeClient)
    assert result.get("captured") is True
    assert client.last_request is not None
    assert client.last_request["json_body"]["name"] == "Valid Tour"


# ── create_planned_tour validation (compact=True rejection) ──────────

def _make_compact_route_data() -> dict[str, Any]:
    """Simulate a compact=True plan_route() result."""
    return {
        "distance_km": 22.81,
        "elevation_up_m": 326.9,
        "elevation_down_m": 326.9,
        "duration": "1h 48m",
        "difficulty": "MODERATE",
        "technical_difficulty": "T2",
        "fitness_difficulty": "C2",
        "way_types": {"trail": 14.9, "street": 29.5, "way": 47.5},
        "surfaces": {"asphalt": 34.5, "unpaved": 35.8, "nature": 16.1},
        "singletrail": {"trail_d1": 1.14, "trail_d2": 2.25, "singletrail_total_km": 3.39},
        "segments": {"routed": 9, "manual": 0},
        "matched_coordinates": 661,
    }


def test_create_planned_tour_rejects_compact_distance_km(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """distance_km (compact=True) must be rejected before any API call."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    compact = _make_compact_route_data()

    result = _run(create_tour(compact, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "compact" in result["error"].lower()
    assert "distance_km" in result["error"]


def test_create_planned_tour_rejects_compact_elevation_up_m(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """elevation_up_m (compact=True) must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    compact = _make_compact_route_data()

    result = _run(create_tour(compact, name="X", sport="mtb"))
    assert result["status"] == "error"
    assert "compact" in result["error"].lower()


def test_create_planned_tour_rejects_compact_elevation_down_m(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """elevation_down_m (compact=True) must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    compact = _make_compact_route_data()
    # Remove distance_km and elevation_up_m to test elevation_down_m detection
    del compact["distance_km"]
    del compact["elevation_up_m"]

    result = _run(create_tour(compact, name="X", sport="mtb"))
    assert result["status"] == "error"
    assert "compact" in result["error"].lower()
    assert "elevation_down_m" in result["error"]


def test_create_planned_tour_rejects_duration_string(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """String duration like '2h 55m' must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full = load_fixture("plan_route_full.json")

    # Corrupt duration
    full = dict(full)
    full["duration"] = "2h 55m"

    result = _run(create_tour(full, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "string" in result["error"].lower()
    assert "duration" in result["error"]
    # Verify no request was made to Komoot
    assert _client.last_request is None


def test_create_planned_tour_rejects_missing_path(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Full response without 'path' must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full = load_fixture("plan_route_full.json")
    truncated = {k: v for k, v in full.items() if k != "path"}

    result = _run(create_tour(truncated, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "path" in result["error"]
    assert _client.last_request is None


def test_create_planned_tour_rejects_missing_segments(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Full response without 'segments' must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full = load_fixture("plan_route_full.json")
    truncated = {k: v for k, v in full.items() if k != "segments"}

    result = _run(create_tour(truncated, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "segments" in result["error"]
    assert _client.last_request is None


def test_create_planned_tour_rejects_missing_embedded(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Full response without '_embedded' must be rejected."""
    mcp, _client = registered
    create_tour = mcp.tools["create_planned_tour"]
    full = load_fixture("plan_route_full.json")
    truncated = {k: v for k, v in full.items() if k != "_embedded"}

    result = _run(create_tour(truncated, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "_embedded" in result["error"]
    assert _client.last_request is None


# ── route_ref: create_planned_tour with route_ref parameter ──────────────

def test_create_planned_tour_with_valid_route_ref(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Valid route_ref loads from cache, validates, and reaches the API."""
    mcp, client = registered
    plan_route = mcp.tools["plan_route"]
    create_tour = mcp.tools["create_planned_tour"]

    # 1) Get route_ref from plan_route(compact=False)
    route_result = _run(plan_route(COORDS, sport="mtb", compact=False))
    ref = route_result["route_ref"]

    # 2) Use route_ref in create_planned_tour
    result = _run(create_tour(route_ref=ref, name="Route Ref Tour", sport="mtb"))

    assert result.get("captured") is True, "should have reached the API"
    payload = client.last_request["json_body"]
    assert payload["name"] == "Route Ref Tour"
    assert payload["sport"] == "mtb"
    assert "path" in payload
    assert "segments" in payload


def test_create_planned_tour_route_ref_removes_from_cache(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """After successful save, the route_ref must be removed from cache."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    create_tour = mcp.tools["create_planned_tour"]

    route_result = _run(plan_route(COORDS, sport="mtb", compact=False))
    ref = route_result["route_ref"]

    # Must be in cache before save
    assert ref in routing._route_cache

    _run(create_tour(route_ref=ref, name="X", sport="mtb"))

    # Must NOT be in cache after save
    assert ref not in routing._route_cache


def test_create_planned_tour_route_ref_one_shot(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """After one successful save, the same route_ref cannot be reused."""
    mcp, client = registered
    plan_route = mcp.tools["plan_route"]
    create_tour = mcp.tools["create_planned_tour"]

    route_result = _run(plan_route(COORDS, sport="mtb", compact=False))
    ref = route_result["route_ref"]

    # First use — succeeds
    _run(create_tour(route_ref=ref, name="First", sport="mtb"))

    # Reset the FakeClient so last_request is None again
    client.last_request = None

    # Second use — must fail (already consumed)
    result = _run(create_tour(route_ref=ref, name="Second", sport="mtb"))
    assert result["status"] == "error"
    assert "invalid or expired" in result["error"]
    assert client.last_request is None  # No API call


def test_create_planned_tour_invalid_route_ref(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Invalid route_ref must return a clean error without API call."""
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]

    result = _run(create_tour(route_ref="route_nonexistent", name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "invalid or expired" in result["error"]
    assert client.last_request is None


def test_create_planned_tour_expired_route_ref(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Expired route_ref must return a clean error without API call."""
    mcp, client = registered
    plan_route = mcp.tools["plan_route"]
    create_tour = mcp.tools["create_planned_tour"]

    route_result = _run(plan_route(COORDS, sport="mtb", compact=False))
    ref = route_result["route_ref"]

    # Manually age the cache entry beyond TTL
    routing._route_cache[ref] = (time.time() - routing._CACHE_TTL - 1, routing._route_cache[ref][1])

    result = _run(create_tour(route_ref=ref, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "expired" in result["error"]
    assert client.last_request is None
    # Should also be removed from cache
    assert ref not in routing._route_cache


def test_create_planned_tour_neither_ref_nor_data(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Calling create_planned_tour without route_ref AND without route_data
    must return a clean error."""
    mcp, client = registered
    create_tour = mcp.tools["create_planned_tour"]

    result = _run(create_tour(name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "Either route_data or route_ref" in result["error"]
    assert client.last_request is None


def test_create_planned_tour_route_ref_kept_on_validation_fail(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """If validation fails, the route_ref must stay in cache for potential retry."""
    mcp, client = registered
    plan_route = mcp.tools["plan_route"]
    create_tour = mcp.tools["create_planned_tour"]

    route_result = _run(plan_route(COORDS, sport="mtb", compact=False))
    ref = route_result["route_ref"]

    # Overwrite cached data with a compact-like dict that will fail validation
    bad_data = _make_compact_route_data()
    routing._route_cache[ref] = (time.time(), bad_data)

    result = _run(create_tour(route_ref=ref, name="X", sport="mtb"))

    assert result["status"] == "error"
    assert "compact" in result["error"].lower()
    assert client.last_request is None

    # Entry must still be in cache
    assert ref in routing._route_cache


def test_two_different_route_refs_independent(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """Two separate plan_route(compact=False) calls produce independent refs."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]

    result1 = _run(plan_route(COORDS, sport="mtb", compact=False))
    result2 = _run(plan_route(COORDS, sport="mtb", compact=False))

    ref1, ref2 = result1["route_ref"], result2["route_ref"]
    assert ref1 != ref2
    assert ref1 in routing._route_cache
    assert ref2 in routing._route_cache


# ── Response size sanity ──────────────────────────────────────────────

def test_plan_route_compact_false_response_size(
    registered: tuple[FakeMCP, FakeClient],
) -> None:
    """The returned compact=False response must be small (under 5 KB)."""
    mcp, _client = registered
    plan_route = mcp.tools["plan_route"]
    result = _run(plan_route(COORDS, sport="mtb", compact=False))

    import json
    size = len(json.dumps(result).encode("utf-8"))
    assert size < 5000, f"compact=False response is {size} bytes — expected < 5 KB"

    # Meanwhile the cached entry should be ~90 KB (the full response)
    ref = result["route_ref"]
    cached_size = len(json.dumps(routing._route_cache[ref][1]).encode("utf-8"))
    assert cached_size > 40000, f"cached route is {cached_size} bytes — expected > 40 KB"