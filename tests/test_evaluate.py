"""Tests for postureopt.evaluate module."""

import pytest
from postureopt.core import Asset, AssetType, Location, SustainmentAction
from postureopt.evaluate import (
    readiness_score,
    coverage_score,
    sustainment_cost,
    posture_efficiency,
)


def make_asset(asset_id: str, loc_id: str, readiness: float, qty: int = 1) -> Asset:
    return Asset(
        asset_id=asset_id,
        asset_type=AssetType.AIRCRAFT,
        location_id=loc_id,
        quantity=qty,
        readiness_rate=readiness,
    )


def make_location(loc_id: str) -> Location:
    return Location(location_id=loc_id, name=loc_id, lat=0.0, lon=0.0)


def test_readiness_score_uniform() -> None:
    assets = [make_asset("A1", "L1", 0.8), make_asset("A2", "L2", 0.6)]
    score = readiness_score(assets)
    assert score == pytest.approx(0.7, abs=1e-9)


def test_readiness_score_empty() -> None:
    assert readiness_score([]) == 0.0


def test_readiness_score_weighted() -> None:
    assets = [
        make_asset("A1", "L1", 1.0, qty=3),
        make_asset("A2", "L2", 0.0, qty=1),
    ]
    score = readiness_score(assets)
    assert score == pytest.approx(0.75, abs=1e-9)


def test_coverage_score_full_coverage() -> None:
    locs = [make_location("L1"), make_location("L2")]
    assets = [make_asset("A1", "L1", 1.0), make_asset("A2", "L2", 1.0)]
    assert coverage_score(assets, locs) == pytest.approx(1.0)


def test_coverage_score_partial() -> None:
    locs = [make_location("L1"), make_location("L2"), make_location("L3")]
    assets = [make_asset("A1", "L1", 1.0)]
    assert coverage_score(assets, locs) == pytest.approx(1 / 3, abs=1e-9)


def test_coverage_score_no_locations() -> None:
    assert coverage_score([], []) == 0.0


def test_sustainment_cost_known_actions() -> None:
    actions = [
        SustainmentAction.REPOSITION,
        SustainmentAction.RESUPPLY,
        SustainmentAction.MAINTAIN,
        SustainmentAction.HOLD,
    ]
    assert sustainment_cost(actions) == pytest.approx(17.0)


def test_sustainment_cost_empty() -> None:
    assert sustainment_cost([]) == 0.0


def test_posture_efficiency_basic() -> None:
    import math
    eff = posture_efficiency(1.0, 1.0, 9.0)
    assert eff == pytest.approx(1.0 / math.log1p(9.0), abs=1e-9)


def test_posture_efficiency_zero_cost() -> None:
    # Zero cost -> log1p(0) == 0 -> return readiness * coverage directly
    eff = posture_efficiency(0.8, 0.5, 0.0)
    assert eff == pytest.approx(0.4, abs=1e-9)
