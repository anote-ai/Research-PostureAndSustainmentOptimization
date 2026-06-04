"""Tests for postureopt.core."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import (
    AssetType, SustainmentAction, Location, Asset, PostureState,
    haversine_distance, ReplenishmentPolicy, PostureOptimizer,
)


def test_asset_type_values():
    values = {a.value for a in AssetType}
    assert values == {"AIRCRAFT", "FUEL_DEPOT", "MAINTENANCE_CREW", "MUNITION", "MEDICAL"}


def test_asset_type_count():
    assert len(AssetType) == 5


def test_sustainment_action_values():
    values = {a.value for a in SustainmentAction}
    assert values == {"REPOSITION", "RESUPPLY", "MAINTAIN", "HOLD"}


def test_sustainment_action_count():
    assert len(SustainmentAction) == 4


def test_location_construction():
    loc = Location(location_id="L001", name="Test", lat=10.0, lon=20.0, capacity=5)
    assert loc.strategic_value == 0.5


def test_asset_readiness_validation():
    with pytest.raises(ValueError):
        Asset(asset_id="A1", asset_type=AssetType.AIRCRAFT,
              location_id="L001", quantity=3, readiness_rate=1.5)


def test_posture_state_assets_at():
    a1 = Asset("A1", AssetType.AIRCRAFT, "L001", 5, 0.9)
    a2 = Asset("A2", AssetType.FUEL_DEPOT, "L002", 3, 0.8)
    state = PostureState(assets=[a1, a2], locations=[])
    assert state.assets_at("L001") == [a1]
    assert len(state.assets_at("L002")) == 1


def test_posture_state_total_readiness():
    a1 = Asset("A1", AssetType.AIRCRAFT, "L001", 4, 1.0)
    a2 = Asset("A2", AssetType.FUEL_DEPOT, "L002", 4, 0.5)
    state = PostureState(assets=[a1, a2], locations=[])
    # (4*1.0 + 4*0.5) / 8 = 6/8 = 0.75
    assert abs(state.total_readiness() - 0.75) < 1e-9


def test_haversine_nyc_la():
    # NYC ~40.71N 74.01W, LA ~34.05N 118.24W
    d = haversine_distance(40.71, -74.01, 34.05, -118.24)
    assert 3800 < d < 4100


def test_replenishment_policy_low_readiness():
    policy = ReplenishmentPolicy()
    a = Asset("A1", AssetType.AIRCRAFT, "L001", 5, 0.3)
    assert policy.decide(a) == SustainmentAction.MAINTAIN


def test_replenishment_policy_low_quantity():
    policy = ReplenishmentPolicy()
    a = Asset("A1", AssetType.AIRCRAFT, "L001", 1, 0.9)
    assert policy.decide(a) == SustainmentAction.RESUPPLY


def test_greedy_placement_assigns_all():
    optimizer = PostureOptimizer(seed=42)
    locations = [
        Location("L1", "Alpha", 0.0, 0.0, 5, 0.9),
        Location("L2", "Beta", 1.0, 1.0, 5, 0.7),
    ]
    assets = [
        Asset(f"A{i}", AssetType.AIRCRAFT, "L1", 1, 0.8)
        for i in range(4)
    ]
    assignments = optimizer.greedy_placement(assets, locations)
    assert len(assignments) == 4
