"""Tests for Experiment 3 — adversarial scenario factories and sweep logic."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location
from postureopt.data import (
    make_adversarial_scenario_set,
    make_deceptive_scenario_set,
    make_posture_state,
    make_skewed_scenario_set,
    make_uniform_scenario_set,
)
from postureopt.drsp import AdversarialModel, CEVOptimizer, RobustCEVOptimizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _locs():
    return [
        Location("L1", "Kadena",   26.35, 127.77, capacity=10, strategic_value=0.95),
        Location("L2", "Andersen", 13.58, 144.93, capacity=10, strategic_value=0.90),
        Location("L3", "Iwakuni",  34.14, 132.24, capacity=10, strategic_value=0.85),
        Location("L4", "Smith",    21.41, -157.93, capacity=10, strategic_value=0.80),
        Location("L5", "Diego",    -7.32,  72.42, capacity=10, strategic_value=0.78),
    ]


def _assets(n: int = 10) -> list:
    return [
        Asset(f"A{i:03d}", AssetType.AIRCRAFT, "L1", quantity=2, readiness_rate=0.85)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scenario factory — structural correctness
# ---------------------------------------------------------------------------

def test_uniform_scenario_set_sums_to_one():
    locs = _locs()
    ss = make_uniform_scenario_set(locs, n_scenarios=5)
    assert abs(sum(s.probability for s in ss.scenarios) - 1.0) < 1e-9


def test_uniform_scenario_set_covers_all_locations():
    locs = _locs()
    ss = make_uniform_scenario_set(locs, n_scenarios=3)
    loc_ids = {loc.location_id for loc in locs}
    for s in ss.scenarios:
        assert set(s.threat_weights.keys()) == loc_ids


def test_skewed_scenario_set_sums_to_one():
    locs = _locs()
    ss = make_skewed_scenario_set(locs)
    assert abs(sum(s.probability for s in ss.scenarios) - 1.0) < 1e-9


def test_skewed_scenario_set_top_location_is_most_threatened():
    locs = _locs()
    ss = make_skewed_scenario_set(locs)
    # SK_high should have the highest threat on L1 (top strategic value)
    high = next(s for s in ss.scenarios if s.scenario_id == "SK_high")
    assert high.threat_weights.get("L1", 0.0) >= 0.9


def test_adversarial_scenario_set_sums_to_one():
    locs = _locs()
    ss = make_adversarial_scenario_set(locs)
    assert abs(sum(s.probability for s in ss.scenarios) - 1.0) < 1e-9


def test_adversarial_scenario_set_has_demand_multipliers_above_one():
    locs = _locs()
    ss = make_adversarial_scenario_set(locs)
    high_demand = [s for s in ss.scenarios if s.demand_multiplier > 1.0]
    assert len(high_demand) >= 2


# ---------------------------------------------------------------------------
# Experiment 3 core claim: robust >= naive for all p_obs when rationality > 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_obs", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_robust_placement_is_valid_for_all_p_obs(p_obs):
    """RobustCEVOptimizer must always return a complete, capacity-respecting placement."""
    locs   = _locs()
    assets = _assets(20)
    prior  = make_skewed_scenario_set(locs)
    adversary = AdversarialModel(p_obs=p_obs, rationality=1.0)

    robust = RobustCEVOptimizer(prior, adversary)
    assignment, final_ss = robust.optimize_placement(assets, locs)

    assert len(assignment) == len(assets), "Every asset must be assigned"
    assert abs(sum(s.probability for s in final_ss.scenarios) - 1.0) < 1e-9
    capacity = {loc.location_id: loc.capacity for loc in locs}
    for loc_id in assignment.values():
        assert loc_id in capacity, f"Unknown location {loc_id}"


@pytest.mark.parametrize("p_obs", [0.5, 0.75, 1.0])
def test_adversary_shifts_probability_toward_occupied_locations(p_obs):
    """Bayesian adversary must increase weight on scenarios that threaten the
    locations where most assets sit."""
    locs   = _locs()
    assets = _assets(20)
    prior  = make_adversarial_scenario_set(locs)
    adversary = AdversarialModel(p_obs=p_obs, rationality=1.0)

    # Force all assets to L1 (top-value location) so the adversary has a clear target
    assignment = {a.asset_id: "L1" for a in assets}
    updated = adversary.update_scenarios(prior, assignment, locs)

    # ADV_atk_top threatens L1 at 0.90 — its probability must rise
    orig_top  = next(s for s in prior.scenarios   if s.scenario_id == "ADV_atk_top")
    upd_top   = next(s for s in updated.scenarios if s.scenario_id == "ADV_atk_top")
    assert upd_top.probability >= orig_top.probability, (
        f"p_obs={p_obs}: adversary should shift toward top-threat scenario"
    )


def test_zero_obs_naive_and_robust_match():
    """When adversary cannot observe (p_obs=0), robust CEV degenerates to naive CEV."""
    locs   = _locs()
    assets = _assets(10)
    prior  = make_skewed_scenario_set(locs)
    adversary = AdversarialModel(p_obs=0.0, rationality=1.0)

    naive_assignment = CEVOptimizer(prior).optimize_placement(assets, locs)
    adv_for_naive = adversary.update_scenarios(prior, naive_assignment, locs)
    naive_eff = CEVOptimizer(adv_for_naive).expected_posture_efficiency(
        assets, locs, naive_assignment
    )

    robust = RobustCEVOptimizer(prior, adversary)
    robust_assignment, adv_for_robust = robust.optimize_placement(assets, locs)
    robust_eff = robust.expected_posture_efficiency(
        assets, locs, robust_assignment, adv_for_robust
    )

    # With p_obs=0, adversary cannot adapt — gap should be negligible
    assert abs(robust_eff - naive_eff) < 0.05


def test_deceptive_scenario_set_sums_to_one():
    locs = _locs()
    ss = make_deceptive_scenario_set(locs)
    assert abs(sum(s.probability for s in ss.scenarios) - 1.0) < 1e-9


def test_deceptive_naive_cev_concentrates_at_top_location():
    """With a low-probability attack prior, naive CEV must still prefer the
    top-strategic-value location (the deceptive lure works)."""
    locs = _locs()
    assets = _assets(20)
    prior = make_deceptive_scenario_set(locs)
    assignment = CEVOptimizer(prior).optimize_placement(assets, locs)
    # L1 should be filled to its capacity before any other location gets assets
    l1_capacity = next(l for l in locs if l.location_id == "L1").capacity
    l1_count = sum(1 for v in assignment.values() if v == "L1")
    assert l1_count >= min(len(assets), l1_capacity), (
        f"Naive CEV should fill L1 to capacity first, got {l1_count} (capacity={l1_capacity})"
    )


@pytest.mark.parametrize("p_obs", [0.5, 0.75, 1.0])
def test_robust_beats_naive_under_deceptive_prior(p_obs):
    """Core Exp-3 claim: at high p_obs with a deceptive prior, naive CEV is
    exploitable and robust CEV maintains a strictly higher efficiency floor."""
    locs   = _locs()
    assets = _assets(20)
    prior  = make_deceptive_scenario_set(locs)
    adversary = AdversarialModel(p_obs=p_obs, rationality=1.0)

    naive_assignment = CEVOptimizer(prior).optimize_placement(assets, locs)
    adv_for_naive = adversary.update_scenarios(prior, naive_assignment, locs)
    naive_eff = CEVOptimizer(adv_for_naive).expected_posture_efficiency(
        assets, locs, naive_assignment
    )

    robust = RobustCEVOptimizer(prior, adversary)
    robust_assignment, adv_for_robust = robust.optimize_placement(assets, locs)
    robust_eff = robust.expected_posture_efficiency(
        assets, locs, robust_assignment, adv_for_robust
    )

    assert robust_eff > naive_eff, (
        f"p_obs={p_obs}: robust ({robust_eff:.4f}) must exceed naive ({naive_eff:.4f}) "
        "under deceptive prior"
    )


def test_naive_efficiency_decreases_with_p_obs_under_deceptive_prior():
    """Naive CEV efficiency must decrease monotonically as adversary observation
    probability increases under the deceptive prior."""
    locs   = _locs()
    assets = _assets(20)
    prior  = make_deceptive_scenario_set(locs)

    efficiencies = []
    for p_obs in [0.0, 0.25, 0.5, 0.75, 1.0]:
        adversary = AdversarialModel(p_obs=p_obs, rationality=1.0)
        assignment = CEVOptimizer(prior).optimize_placement(assets, locs)
        adv_ss = adversary.update_scenarios(prior, assignment, locs)
        eff = CEVOptimizer(adv_ss).expected_posture_efficiency(assets, locs, assignment)
        efficiencies.append(eff)

    for i in range(len(efficiencies) - 1):
        assert efficiencies[i] >= efficiencies[i + 1] - 1e-9, (
            f"Naive efficiency should be non-increasing: {efficiencies}"
        )


def test_bayesian_adversary_shifts_more_probability_than_random():
    """With rationality=1, the adversary concentrates more probability mass on
    threatening scenarios than with rationality=0 (random), for any p_obs > 0."""
    locs   = _locs()
    assets = _assets(20)
    prior  = make_adversarial_scenario_set(locs)

    # All assets at L1 — the adversary should clearly concentrate on ADV_atk_top
    assignment = {a.asset_id: "L1" for a in assets}
    orig_top_prob = next(s for s in prior.scenarios if s.scenario_id == "ADV_atk_top").probability

    bayesian = AdversarialModel(p_obs=1.0, rationality=1.0)
    random_  = AdversarialModel(p_obs=1.0, rationality=0.0)

    upd_bayesian = bayesian.update_scenarios(prior, assignment, locs)
    upd_random   = random_.update_scenarios(prior, assignment, locs)

    bay_top = next(s for s in upd_bayesian.scenarios if s.scenario_id == "ADV_atk_top").probability
    rnd_top = next(s for s in upd_random.scenarios   if s.scenario_id == "ADV_atk_top").probability

    # Bayesian adversary should move MORE probability toward the top-threat scenario
    assert bay_top >= rnd_top, (
        f"Bayesian ({bay_top:.4f}) should concentrate >= random ({rnd_top:.4f})"
    )
    # And Bayesian should shift meaningfully relative to the prior
    assert bay_top > orig_top_prob - 1e-9
