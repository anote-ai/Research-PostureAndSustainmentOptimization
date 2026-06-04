"""Tests for postureopt.core module."""

import pytest
from postureopt.core import (
    AssetType,
    SustainmentAction,
    Location,
    Asset,
    PostureState,
    haversine_distance,
    PostureOptimizer,
)


def test_asset_type_enum() -> None:
    assert AssetType.AIRCRAFT == "AIRCRAFT"
    assert AssetType.FUEL_DEPOT == "FUEL_DEPOT"
    assert len(AssetType) == 5


def test_sustainment_action_enum() -> None:
    assert SustainmentAction.REPOSITION == "REPOSITION"
    assert SustainmentAction.HOLD == "HOLD"
    assert len(SustainmentAction) == 4


def test_location_construction() -> None:
    loc = Location(location_id="LOC01", name="Ramstein AB", lat=49.4369, lon=7.6003, capacity=50)
    assert loc.location_id == "LOC01"
    assert loc.capacity == 50


def test_location_invalid_lat() -> None:
    with pytest.raises(ValueError):
        Location(location_id="BAD", name="Bad", lat=100.0, lon=0.0)


def test_asset_construction() -> None:
    asset = Asset(
        asset_id="A001",
        asset_type=AssetType.AIRCRAFT,
        location_id="LOC01",
        quantity=4,
        readiness_rate=0.85,
    )
    assert asset.asset_type == AssetType.AIRCRAFT
    assert asset.readiness_rate == pytest.approx(0.85)


def test_asset_invalid_readiness() -> None:
    with pytest.raises(ValueError):
        Asset(asset_id="BAD", asset_type=AssetType.MEDICAL, location_id="LOC01", readiness_rate=1.5)


def test_posture_state_construction() -> None:
    loc = Location(location_id="LOC01", name="Base Alpha", lat=35.0, lon=-80.0)
    asset = Asset(asset_id="A001", asset_type=AssetType.FUEL_DEPOT, location_id="LOC01")
    state = PostureState(state_id="S001", assets=[asset], locations=[loc], time_step=3)
    assert state.time_step == 3
    assert len(state.assets) == 1


def test_haversine_nyc_to_la() -> None:
    # NYC: 40.7128N, 74.0060W; LA: 34.0522N, 118.2437W
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    # Expected ~3940 km, within 5%
    assert abs(dist - 3940) / 3940 < 0.05


def test_haversine_same_point() -> None:
    dist = haversine_distance(45.0, 90.0, 45.0, 90.0)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_posture_optimizer_instantiation() -> None:
    opt = PostureOptimizer(seed=7)
    assert opt.seed == 7


def test_posture_optimizer_optimize_placement_stub() -> None:
    opt = PostureOptimizer()
    loc = Location(location_id="LOC01", name="Base", lat=0.0, lon=0.0)
    asset = Asset(asset_id="A001", asset_type=AssetType.AIRCRAFT, location_id="LOC01")
    result = opt.optimize_placement([asset], [loc], demand={"LOC01": 1})
    assert isinstance(result, dict)
    assert "A001" in result


def test_posture_optimizer_replenishment_policy() -> None:
    opt = PostureOptimizer()
    assets = [
        Asset(asset_id="A1", asset_type=AssetType.AIRCRAFT, location_id="LOC01", readiness_rate=0.3),
        Asset(asset_id="A2", asset_type=AssetType.AIRCRAFT, location_id="LOC01", readiness_rate=0.9),
    ]
    state = PostureState(state_id="S1", assets=assets)
    actions = opt.replenishment_policy(state)
    assert len(actions) == 2
    from postureopt.core import SustainmentAction
    assert actions[0] == SustainmentAction.MAINTAIN
    assert actions[1] == SustainmentAction.HOLD
