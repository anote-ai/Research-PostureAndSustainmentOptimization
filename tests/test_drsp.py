"""Tests for drsp.py — distributionally robust two-stage posture optimizer."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location, PostureOptimizer
from postureopt.drsp import (
    AdversarialModel,
    CEVOptimizer,
    RobustCEVOptimizer,
    ScenarioSet,
    ThreatScenario,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _locations():
    return [
        Location("L1", "Alpha",   26.0, 127.0, capacity=10, strategic_value=0.95),
        Location("L2", "Beta",    34.0, 132.0, capacity=10, strategic_value=0.90),
        Location("L3", "Gamma",   13.0, 144.0, capacity=10, strategic_value=0.85),
        Location("L4", "Delta",   40.0, 141.0, capacity=10, strategic_value=0.75),
        Location("L5", "Epsilon", 37.0, 127.0, capacity=10, strategic_value=0.80),
    ]


def _assets(n: int = 10, readiness: float = 0.85) -> list:
    return [
        Asset(f"A{i:03d}", AssetType.AIRCRAFT, "L1", quantity=2, readiness_rate=readiness)
        for i in range(n)
    ]


def _uniform_scenarios(locations):
    """Equal threat on all locations — greedy and CEV should behave similarly."""
    loc_ids = [loc.location_id for loc in locations]
    return ScenarioSet(scenarios=[
        ThreatScenario("S1", {l: 0.1 for l in loc_ids}, probability=0.5),
        ThreatScenario("S2", {l: 0.2 for l in loc_ids}, probability=0.5),
    ])


def _skewed_scenarios(locations):
    """High threat concentrated on L1 (the greedy-preferred location) with 80% weight.
    Forces CEV to prefer lower-value but safer locations over L1."""
    loc_ids = [loc.location_id for loc in locations]
    return ScenarioSet(scenarios=[
        ThreatScenario(
            "S_high",
            {"L1": 0.95, "L2": 0.1, "L3": 0.0, "L4": 0.0, "L5": 0.0},
            probability=0.8,
        ),
        ThreatScenario("S_low", {l: 0.05 for l in loc_ids}, probability=0.2),
    ])


# ---------------------------------------------------------------------------
# ThreatScenario
# ---------------------------------------------------------------------------

def test_threat_scenario_rejects_bad_probability():
    with pytest.raises(ValueError):
        ThreatScenario("S1", {}, probability=1.5)


def test_threat_scenario_rejects_bad_weight():
    with pytest.raises(ValueError):
        ThreatScenario("S1", {"L1": -0.1}, probability=1.0)


def test_threat_scenario_effective_value_with_threat():
    loc = Location("L1", "Alpha", 0.0, 0.0, 5, strategic_value=0.9)
    s = ThreatScenario("S1", {"L1": 0.5}, probability=1.0)
    assert abs(s.effective_strategic_value(loc) - 0.45) < 1e-9


def test_threat_scenario_effective_value_no_threat():
    loc = Location("L1", "Alpha", 0.0, 0.0, 5, strategic_value=0.8)
    s = ThreatScenario("S1", {}, probability=1.0)
    assert abs(s.effective_strategic_value(loc) - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# ScenarioSet
# ---------------------------------------------------------------------------

def test_scenario_set_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        ScenarioSet(scenarios=[
            ThreatScenario("S1", {}, probability=0.3),
            ThreatScenario("S2", {}, probability=0.3),
        ])


def test_scenario_set_empty_is_valid():
    ss = ScenarioSet()
    assert ss.scenarios == []


def test_scenario_set_weighted_location_value():
    loc = Location("L1", "Alpha", 0.0, 0.0, 5, strategic_value=1.0)
    ss = ScenarioSet(scenarios=[
        ThreatScenario("S1", {"L1": 0.0}, probability=0.5),
        ThreatScenario("S2", {"L1": 1.0}, probability=0.5),
    ])
    # 0.5*1.0 + 0.5*0.0 = 0.5
    assert abs(ss.weighted_location_value(loc) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# CEVOptimizer — placement structure
# ---------------------------------------------------------------------------

def test_cev_assigns_all_assets():
    locs = _locations()
    assets = _assets(10)
    cev = CEVOptimizer(_uniform_scenarios(locs))
    assignment = cev.optimize_placement(assets, locs)
    assert len(assignment) == len(assets)


def test_cev_respects_capacity():
    locs = [
        Location("L1", "Alpha", 0.0, 0.0, capacity=3, strategic_value=0.9),
        Location("L2", "Beta",  1.0, 0.0, capacity=10, strategic_value=0.5),
    ]
    assets = _assets(5)
    ss = ScenarioSet(scenarios=[ThreatScenario("S1", {}, probability=1.0)])
    assignment = CEVOptimizer(ss).optimize_placement(assets, locs)
    l1_count = sum(1 for v in assignment.values() if v == "L1")
    assert l1_count <= 3


def test_cev_expected_efficiency_positive():
    locs = _locations()
    assets = _assets(5)
    ss = _uniform_scenarios(locs)
    cev = CEVOptimizer(ss)
    assignment = cev.optimize_placement(assets, locs)
    assert cev.expected_posture_efficiency(assets, locs, assignment) > 0.0


# ---------------------------------------------------------------------------
# CEVOptimizer — EVSS (core paper claim)
# ---------------------------------------------------------------------------

def test_cev_beats_greedy_under_skewed_threat():
    """EVSS > 0: CEV outperforms greedy when dominant scenario heavily threatens
    the highest-strategic-value location (the greedy-preferred choice)."""
    locs = _locations()
    assets = _assets(20)
    ss = _skewed_scenarios(locs)

    greedy_assignment = PostureOptimizer(seed=42).greedy_placement(assets, locs)
    cev = CEVOptimizer(ss)
    cev_assignment = cev.optimize_placement(assets, locs)

    greedy_eff = cev.expected_posture_efficiency(assets, locs, greedy_assignment)
    cev_eff = cev.expected_posture_efficiency(assets, locs, cev_assignment)

    evss = cev_eff - greedy_eff
    assert evss > 0.0, f"EVSS={evss:.4f}: CEV must outperform greedy under skewed threat"


def test_cev_not_worse_than_greedy_under_uniform_threat():
    """Under uniform threat, CEV should match greedy (no information advantage)."""
    locs = _locations()
    assets = _assets(5)
    ss = _uniform_scenarios(locs)

    greedy_assignment = PostureOptimizer(seed=42).greedy_placement(assets, locs)
    cev = CEVOptimizer(ss)
    cev_assignment = cev.optimize_placement(assets, locs)

    greedy_eff = cev.expected_posture_efficiency(assets, locs, greedy_assignment)
    cev_eff = cev.expected_posture_efficiency(assets, locs, cev_assignment)

    assert cev_eff >= greedy_eff - 1e-9


# ---------------------------------------------------------------------------
# AdversarialModel
# ---------------------------------------------------------------------------

def test_adversarial_zero_obs_leaves_probabilities_unchanged():
    locs = _locations()
    assets = _assets(5)
    ss = _skewed_scenarios(locs)
    adversary = AdversarialModel(p_obs=0.0)
    assignment = PostureOptimizer(seed=42).greedy_placement(assets, locs)
    updated = adversary.update_scenarios(ss, assignment, locs)
    for orig, upd in zip(ss.scenarios, updated.scenarios):
        assert abs(orig.probability - upd.probability) < 1e-9


def test_adversarial_full_obs_increases_weight_on_threatened_location():
    """With p_obs=1 and all assets at L1, adversary shifts toward the scenario
    that most heavily threatens L1."""
    locs = _locations()
    assets = _assets(20)
    ss = _skewed_scenarios(locs)
    adversary = AdversarialModel(p_obs=1.0, rationality=1.0)
    assignment = {a.asset_id: "L1" for a in assets}
    updated = adversary.update_scenarios(ss, assignment, locs)
    # S_high threatens L1 at 0.95 — its probability should increase
    assert updated.scenarios[0].probability >= ss.scenarios[0].probability


def test_adversarial_updated_probabilities_sum_to_one():
    locs = _locations()
    assets = _assets(10)
    ss = _skewed_scenarios(locs)
    adversary = AdversarialModel(p_obs=0.75, rationality=0.8)
    assignment = {a.asset_id: "L2" for a in assets}
    updated = adversary.update_scenarios(ss, assignment, locs)
    assert abs(sum(s.probability for s in updated.scenarios) - 1.0) < 1e-9


def test_adversarial_model_rejects_bad_p_obs():
    with pytest.raises(ValueError):
        AdversarialModel(p_obs=1.5)


def test_adversarial_model_rejects_bad_rationality():
    with pytest.raises(ValueError):
        AdversarialModel(p_obs=0.5, rationality=-0.1)


# ---------------------------------------------------------------------------
# RobustCEVOptimizer
# ---------------------------------------------------------------------------

def test_robust_cev_returns_valid_placement():
    locs = _locations()
    assets = _assets(10)
    ss = _skewed_scenarios(locs)
    robust = RobustCEVOptimizer(ss, AdversarialModel(p_obs=0.5))
    assignment, final_ss = robust.optimize_placement(assets, locs)
    assert len(assignment) == len(assets)
    assert abs(sum(s.probability for s in final_ss.scenarios) - 1.0) < 1e-9


def test_robust_cev_maintains_floor_vs_naive_under_full_adversarial_observation():
    """Robust CEV evaluated under its adversarial scenario distribution outperforms
    naive CEV evaluated under the adversarial distribution the adversary imposes
    on the naive placement.

    Setup: high-value L1 is nearly always attacked when observed. Naive CEV still
    places at L1 (high prior expected value). Adversary perfectly observes and
    concentrates attack on L1. Robust CEV iterates away from L1 and achieves
    a higher efficiency floor under the adversary's response.
    """
    locs = [
        Location("L1", "Alpha",   26.0, 127.0, capacity=10, strategic_value=1.0),
        Location("L2", "Beta",    34.0, 132.0, capacity=10, strategic_value=0.7),
        Location("L3", "Gamma",   13.0, 144.0, capacity=10, strategic_value=0.6),
        Location("L4", "Delta",   40.0, 141.0, capacity=10, strategic_value=0.5),
        Location("L5", "Epsilon", 37.0, 127.0, capacity=10, strategic_value=0.4),
    ]
    assets = _assets(10)
    prior = ScenarioSet(scenarios=[
        ThreatScenario("S_safe",   {}, probability=0.9),
        ThreatScenario("S_atk_L1", {"L1": 0.99}, probability=0.1),
    ])
    adversary = AdversarialModel(p_obs=1.0, rationality=1.0)

    # Naive CEV: optimizes against prior, ignoring adversarial response
    naive_cev = CEVOptimizer(prior)
    naive_assignment = naive_cev.optimize_placement(assets, locs)
    # Adversary observes naive placement and concentrates attack accordingly
    adv_for_naive = adversary.update_scenarios(prior, naive_assignment, locs)
    naive_eff = CEVOptimizer(adv_for_naive).expected_posture_efficiency(
        assets, locs, naive_assignment
    )

    # Robust CEV: iterates until stable against adversarial response
    robust = RobustCEVOptimizer(prior, adversary)
    robust_assignment, adv_for_robust = robust.optimize_placement(assets, locs)
    robust_eff = robust.expected_posture_efficiency(
        assets, locs, robust_assignment, adv_for_robust
    )

    assert robust_eff > naive_eff, (
        f"Robust ({robust_eff:.4f}) must exceed naive ({naive_eff:.4f}) "
        "under full adversarial observation"
    )
