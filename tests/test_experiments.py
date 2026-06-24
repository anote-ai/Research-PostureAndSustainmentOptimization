"""Tests for Experiments 1 & 2: ThreatScenario, ScenarioWeightedOptimizer, EVSS."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from postureopt.core import (
    Asset, AssetType, Location, PostureOptimizer, ScenarioWeightedOptimizer, ThreatScenario,
)
from postureopt.data import make_posture_state, make_threat_scenarios, simulate_degradation
from postureopt.evaluate import evss, scenario_weighted_readiness


# ---------------------------------------------------------------------------
# ThreatScenario
# ---------------------------------------------------------------------------

def test_threat_scenario_construction():
    s = ThreatScenario(scenario_id="S0", threat_levels={"L001": 0.3, "L002": 0.1}, weight=2.0)
    assert s.weight == 2.0
    assert s.threat_levels["L001"] == pytest.approx(0.3)


def test_threat_scenario_default_weight():
    s = ThreatScenario(scenario_id="S0", threat_levels={})
    assert s.weight == 1.0


# ---------------------------------------------------------------------------
# scenario_weighted_readiness
# ---------------------------------------------------------------------------

def _two_locs():
    return [
        Location("L1", "Alpha", 0.0, 0.0, 10, 0.9),
        Location("L2", "Beta", 1.0, 1.0, 10, 0.6),
    ]


def _two_assets():
    return [
        Asset("A1", AssetType.AIRCRAFT, "L1", 4, 1.0),
        Asset("A2", AssetType.FUEL_DEPOT, "L2", 4, 1.0),
    ]


def test_scenario_weighted_readiness_no_threat():
    assets = _two_assets()
    scenarios = [ThreatScenario("S0", {"L1": 0.0, "L2": 0.0}, weight=1.0)]
    # No threat → readiness unchanged
    assert scenario_weighted_readiness(assets, scenarios) == pytest.approx(1.0)


def test_scenario_weighted_readiness_full_threat():
    assets = _two_assets()
    scenarios = [ThreatScenario("S0", {"L1": 1.0, "L2": 1.0}, weight=1.0)]
    assert scenario_weighted_readiness(assets, scenarios) == pytest.approx(0.0)


def test_scenario_weighted_readiness_partial_threat():
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 2, 1.0)]
    # 50% threat → effective readiness = 0.5
    scenarios = [ThreatScenario("S0", {"L1": 0.5}, weight=1.0)]
    assert scenario_weighted_readiness(assets, scenarios) == pytest.approx(0.5)


def test_scenario_weighted_readiness_two_scenarios_equal_weight():
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 1, 1.0)]
    # Scenario 0: threat 0.0 → readiness 1.0
    # Scenario 1: threat 1.0 → readiness 0.0
    # Expected = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    scenarios = [
        ThreatScenario("S0", {"L1": 0.0}, weight=1.0),
        ThreatScenario("S1", {"L1": 1.0}, weight=1.0),
    ]
    assert scenario_weighted_readiness(assets, scenarios) == pytest.approx(0.5)


def test_scenario_weighted_readiness_peaked_weights():
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 1, 1.0)]
    # Weight 3 on threat=0 scenario, weight 1 on threat=1 → expected = 0.75
    scenarios = [
        ThreatScenario("S0", {"L1": 0.0}, weight=3.0),
        ThreatScenario("S1", {"L1": 1.0}, weight=1.0),
    ]
    assert scenario_weighted_readiness(assets, scenarios) == pytest.approx(0.75)


def test_scenario_weighted_readiness_empty():
    assert scenario_weighted_readiness([], []) == 0.0
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 1, 1.0)]
    assert scenario_weighted_readiness(assets, []) == 0.0


# ---------------------------------------------------------------------------
# evss
# ---------------------------------------------------------------------------

def test_evss_positive():
    assert evss(0.80, 0.65) == pytest.approx(0.15)


def test_evss_zero():
    assert evss(0.70, 0.70) == pytest.approx(0.0)


def test_evss_negative():
    # Greedy outperforms (edge case)
    assert evss(0.60, 0.70) == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# ScenarioWeightedOptimizer
# ---------------------------------------------------------------------------

def test_cev_optimizer_assigns_all_assets():
    locs = _two_locs()
    assets = [Asset(f"A{i}", AssetType.AIRCRAFT, "L1", 1, 0.8) for i in range(4)]
    scenarios = [ThreatScenario("S0", {"L1": 0.5, "L2": 0.1}, weight=1.0)]
    opt = ScenarioWeightedOptimizer(seed=42)
    assignment = opt.optimize_placement(assets, locs, scenarios)
    assert len(assignment) == 4
    assert all(v in {"L1", "L2"} for v in assignment.values())


def test_cev_optimizer_prefers_low_threat_location():
    # L1 high strategic value but very high threat; L2 low value but no threat.
    # CEV effective value of L1 = 0.9 * (1 - 0.9) = 0.09
    # CEV effective value of L2 = 0.8 * (1 - 0.0) = 0.80 → CEV should prefer L2.
    locs = [
        Location("L1", "High-threat", 0.0, 0.0, 10, 0.9),
        Location("L2", "Safe", 1.0, 1.0, 10, 0.8),
    ]
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 1, 0.9)]
    scenarios = [ThreatScenario("S0", {"L1": 0.9, "L2": 0.0}, weight=1.0)]
    opt = ScenarioWeightedOptimizer(seed=42)
    assignment = opt.optimize_placement(assets, locs, scenarios)
    assert assignment["A1"] == "L2"


def test_greedy_prefers_high_strategic_value():
    locs = [
        Location("L1", "High-value", 0.0, 0.0, 10, 0.9),
        Location("L2", "Low-value", 1.0, 1.0, 10, 0.5),
    ]
    assets = [Asset("A1", AssetType.AIRCRAFT, "L1", 1, 0.9)]
    opt = PostureOptimizer(seed=42)
    assignment = opt.greedy_placement(assets, locs)
    assert assignment["A1"] == "L1"


def test_cev_outperforms_greedy_under_adversarial_threat():
    state = make_posture_state(n_assets=20, seed=42)
    greedy_opt = PostureOptimizer(seed=42)
    cev_opt = ScenarioWeightedOptimizer(seed=42)

    greedy_assignment = greedy_opt.greedy_placement(state.assets, state.locations)
    scenarios = make_threat_scenarios(
        state.locations,
        distribution="adversarial",
        n_scenarios=20,
        weight_distribution="peaked",
        greedy_assignment=greedy_assignment,
        seed=42,
    )

    greedy_assets = [
        Asset(a.asset_id, a.asset_type, greedy_assignment.get(a.asset_id, a.location_id),
              a.quantity, a.readiness_rate, a.maintenance_days_remaining)
        for a in state.assets
    ]
    greedy_swr = scenario_weighted_readiness(greedy_assets, scenarios)

    cev_assignment = cev_opt.optimize_placement(state.assets, state.locations, scenarios)
    cev_assets = [
        Asset(a.asset_id, a.asset_type, cev_assignment.get(a.asset_id, a.location_id),
              a.quantity, a.readiness_rate, a.maintenance_days_remaining)
        for a in state.assets
    ]
    cev_swr = scenario_weighted_readiness(cev_assets, scenarios)

    assert evss(cev_swr, greedy_swr) > 0, (
        f"CEV ({cev_swr:.4f}) should beat greedy ({greedy_swr:.4f}) under adversarial threat"
    )


# ---------------------------------------------------------------------------
# make_threat_scenarios
# ---------------------------------------------------------------------------

def test_make_threat_scenarios_count():
    locs = _two_locs()
    scenarios = make_threat_scenarios(locs, n_scenarios=10, seed=0)
    assert len(scenarios) == 10


def test_make_threat_scenarios_uniform_threat_range():
    locs = _two_locs()
    scenarios = make_threat_scenarios(locs, distribution="uniform", n_scenarios=50, seed=0)
    for s in scenarios:
        for lid, t in s.threat_levels.items():
            assert 0.0 <= t <= 1.0, f"Threat {t} out of [0,1] for {lid}"


def test_make_threat_scenarios_adversarial_targets_greedy():
    locs = _two_locs()
    greedy_assignment = {"A1": "L1"}  # greedy chose L1
    scenarios = make_threat_scenarios(
        locs, distribution="adversarial", n_scenarios=30,
        greedy_assignment=greedy_assignment, seed=0,
    )
    mean_threat_l1 = sum(s.threat_levels["L1"] for s in scenarios) / len(scenarios)
    mean_threat_l2 = sum(s.threat_levels["L2"] for s in scenarios) / len(scenarios)
    assert mean_threat_l1 > mean_threat_l2, (
        "Adversarial distribution should concentrate threat on greedy-chosen locations"
    )


def test_make_threat_scenarios_peaked_weights_vary():
    locs = _two_locs()
    scenarios = make_threat_scenarios(locs, weight_distribution="peaked", n_scenarios=20, seed=7)
    weights = [s.weight for s in scenarios]
    assert max(weights) > min(weights) * 2, "Peaked weights should have significant spread"


def test_make_threat_scenarios_uniform_weights_constant():
    locs = _two_locs()
    scenarios = make_threat_scenarios(locs, weight_distribution="uniform", n_scenarios=10, seed=0)
    assert all(s.weight == 1.0 for s in scenarios)


# ---------------------------------------------------------------------------
# simulate_degradation with fixed rate
# ---------------------------------------------------------------------------

def test_fixed_degradation_rate_deterministic():
    state = make_posture_state(n_assets=5, seed=1)
    h1 = simulate_degradation(state, n_steps=3, seed=99, degradation_rate=0.10)
    h2 = simulate_degradation(state, n_steps=3, seed=0, degradation_rate=0.10)
    # Different seeds should produce identical results with fixed rate
    for s1, s2 in zip(h1, h2):
        assert s1.total_readiness() == pytest.approx(s2.total_readiness())


def test_fixed_degradation_rate_decreases_readiness():
    state = make_posture_state(n_assets=10, seed=42)
    initial = state.total_readiness()
    history = simulate_degradation(state, n_steps=5, seed=42, degradation_rate=0.05)
    # At least some degradation should occur before maintenance kicks in
    assert history[-1].total_readiness() <= initial + 0.01
