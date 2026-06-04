"""Tests for postureopt.evaluate."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location, PostureState, SustainmentAction
from postureopt.evaluate import (
    readiness_score, coverage_score, sustainment_cost,
    posture_efficiency, placement_quality, simulation_summary,
)


def _make_assets():
    return [
        Asset("A1", AssetType.AIRCRAFT, "L001", 4, 1.0),
        Asset("A2", AssetType.FUEL_DEPOT, "L002", 4, 0.5),
    ]


def _make_locations():
    return [
        Location("L001", "Alpha", 0.0, 0.0, 10, 0.9),
        Location("L002", "Beta", 1.0, 1.0, 10, 0.7),
    ]


def test_readiness_score():
    assets = _make_assets()
    assert abs(readiness_score(assets) - 0.75) < 1e-9


def test_coverage_score_full():
    assets = _make_assets()
    locs = _make_locations()
    assert coverage_score(assets, locs) == 1.0


def test_coverage_score_partial():
    assets = [Asset("A1", AssetType.AIRCRAFT, "L001", 2, 0.8)]
    locs = _make_locations()
    assert coverage_score(assets, locs) == 0.5


def test_sustainment_cost_known():
    actions = [SustainmentAction.REPOSITION, SustainmentAction.HOLD, SustainmentAction.MAINTAIN]
    cost = sustainment_cost(actions)
    assert abs(cost - 12.0) < 1e-9


def test_posture_efficiency():
    eff = posture_efficiency(1.0, 1.0, 0.0)
    import math
    assert abs(eff - 1.0 / math.log1p(1.0)) < 1e-9


def test_placement_quality():
    assets = _make_assets()
    locs = _make_locations()
    assignments = {"A1": "L001", "A2": "L002"}
    pq = placement_quality(assignments, assets, locs)
    assert abs(pq - 0.8) < 1e-9


def test_simulation_summary_structure():
    from postureopt.data import make_posture_state, simulate_degradation
    state = make_posture_state(n_assets=5, seed=1)
    history = simulate_degradation(state, n_steps=3, seed=1)
    summary = simulation_summary(history)
    assert "n_steps" in summary
    assert summary["n_steps"] == 3
    assert "mean_readiness" in summary
