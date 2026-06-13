"""Distributionally robust two-stage stochastic optimizer for force posture.

Stage 1 (here-and-now): asset placement decisions before scenarios are revealed.
Stage 2 (recourse): sustainment actions after scenario realization.

Adversarial Bayesian model: adversary observes posture with probability p_obs
and shifts their attack distribution toward the most exposed locations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core import Asset, Location, ReplenishmentPolicy, SustainmentAction
from .evaluate import coverage_score, posture_efficiency, readiness_score, sustainment_cost


@dataclass
class ThreatScenario:
    """A single threat scenario with per-location threat intensities.

    threat_weights: location_id -> threat intensity in [0, 1].
        1.0 = fully contested; 0.0 = safe. Omitted locations are implicitly 0.
    probability: prior probability of this scenario; must sum to 1 across ScenarioSet.
    demand_multiplier: scales sustainment cost under this scenario.
    """

    scenario_id: str
    threat_weights: Dict[str, float]
    probability: float
    demand_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")
        if self.demand_multiplier <= 0:
            raise ValueError("demand_multiplier must be > 0")
        for loc_id, w in self.threat_weights.items():
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"threat_weight for {loc_id} must be in [0, 1], got {w}")

    def effective_strategic_value(self, location: Location) -> float:
        """Location strategic value discounted by threat intensity."""
        threat = self.threat_weights.get(location.location_id, 0.0)
        return location.strategic_value * (1.0 - threat)


@dataclass
class ScenarioSet:
    """Collection of threat scenarios whose probabilities must sum to 1."""

    scenarios: List[ThreatScenario] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scenarios:
            total = sum(s.probability for s in self.scenarios)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Scenario probabilities must sum to 1.0, got {total:.6f}")

    def weighted_location_value(self, location: Location) -> float:
        """Expected strategic value of a location across all scenarios."""
        return sum(
            s.probability * s.effective_strategic_value(location)
            for s in self.scenarios
        )


def _recourse_actions(
    assets: List[Asset],
    scenario: ThreatScenario,
) -> List[SustainmentAction]:
    """Second-stage recourse: reposition assets from high-threat locations, else use
    standard replenishment policy."""
    policy = ReplenishmentPolicy()
    actions: List[SustainmentAction] = []
    for asset in assets:
        if scenario.threat_weights.get(asset.location_id, 0.0) > 0.7:
            actions.append(SustainmentAction.REPOSITION)
        else:
            actions.append(policy.decide(asset))
    return actions


def _apply_placement(assets: List[Asset], assignment: Dict[str, str]) -> List[Asset]:
    """Return copies of assets with locations updated per assignment."""
    result = []
    for a in assets:
        loc_id = assignment.get(a.asset_id, a.location_id)
        result.append(Asset(
            asset_id=a.asset_id,
            asset_type=a.asset_type,
            location_id=loc_id,
            quantity=a.quantity,
            readiness_rate=a.readiness_rate,
            maintenance_days_remaining=a.maintenance_days_remaining,
        ))
    return result


class CEVOptimizer:
    """Scenario-weighted expected-value (CEV) posture optimizer.

    Stage 1: rank locations by their scenario-weighted expected strategic value,
    then assign assets greedily to highest-ranked locations with remaining capacity.
    Stage 2: per scenario, apply threat-aware recourse actions.
    """

    def __init__(self, scenario_set: ScenarioSet) -> None:
        self.scenario_set = scenario_set

    def optimize_placement(
        self, assets: List[Asset], locations: List[Location]
    ) -> Dict[str, str]:
        """Assign assets to maximize expected posture efficiency across scenarios."""
        sorted_locs = sorted(
            locations,
            key=lambda loc: -self.scenario_set.weighted_location_value(loc),
        )
        capacity_remaining = {loc.location_id: loc.capacity for loc in sorted_locs}
        assignment: Dict[str, str] = {}
        for asset in assets:
            for loc in sorted_locs:
                if capacity_remaining[loc.location_id] > 0:
                    assignment[asset.asset_id] = loc.location_id
                    capacity_remaining[loc.location_id] -= 1
                    break
        return assignment

    def expected_posture_efficiency(
        self,
        assets: List[Asset],
        locations: List[Location],
        assignment: Dict[str, str],
    ) -> float:
        """Compute E_s[posture_efficiency(assignment, recourse_s, scenario_s)]."""
        located = _apply_placement(assets, assignment)
        total = 0.0
        for scenario in self.scenario_set.scenarios:
            actions = _recourse_actions(located, scenario)
            cost = sustainment_cost(actions) * scenario.demand_multiplier
            r = readiness_score(located)
            c = coverage_score(located, locations)
            total += scenario.probability * posture_efficiency(r, c, cost)
        return total


class AdversarialModel:
    """Bayesian counter-move model for an A2/AD adversary.

    The adversary observes the defender's placement with probability p_obs and,
    if rational, shifts attack distribution toward scenarios that threaten the
    most occupied locations (maximizing disruption potential).
    """

    def __init__(self, p_obs: float, rationality: float = 1.0) -> None:
        if not (0.0 <= p_obs <= 1.0):
            raise ValueError(f"p_obs must be in [0, 1], got {p_obs}")
        if not (0.0 <= rationality <= 1.0):
            raise ValueError(f"rationality must be in [0, 1], got {rationality}")
        self.p_obs = p_obs
        self.rationality = rationality

    def update_scenarios(
        self,
        scenario_set: ScenarioSet,
        assignment: Dict[str, str],
        locations: List[Location],
    ) -> ScenarioSet:
        """Return adversarially-updated scenario weights given observed placement.

        Blend = p_obs * rationality interpolates prior toward the adversary's
        best-response distribution (concentrate on scenarios that threaten
        the most-occupied locations).
        """
        location_counts: Dict[str, int] = {loc.location_id: 0 for loc in locations}
        for loc_id in assignment.values():
            if loc_id in location_counts:
                location_counts[loc_id] += 1

        scores = [
            sum(
                s.threat_weights.get(loc_id, 0.0) * count
                for loc_id, count in location_counts.items()
            )
            for s in scenario_set.scenarios
        ]
        total_score = sum(scores)
        if total_score == 0.0:
            n = len(scenario_set.scenarios)
            adversarial_probs = [1.0 / n] * n
        else:
            adversarial_probs = [sc / total_score for sc in scores]

        blend = self.p_obs * self.rationality
        updated = []
        for s, adv_p in zip(scenario_set.scenarios, adversarial_probs):
            new_prob = (1.0 - blend) * s.probability + blend * adv_p
            updated.append(ThreatScenario(
                scenario_id=s.scenario_id,
                threat_weights=s.threat_weights,
                probability=new_prob,
                demand_multiplier=s.demand_multiplier,
            ))
        return ScenarioSet(scenarios=updated)


class RobustCEVOptimizer:
    """Adversarially-robust CEV optimizer.

    Iterates: place assets → adversary updates scenario distribution → re-place,
    until placement converges (or max_iter iterations). Returns the converged
    placement and the adversarial scenario distribution it faced.
    """

    def __init__(
        self,
        scenario_set: ScenarioSet,
        adversary: AdversarialModel,
        max_iter: int = 10,
    ) -> None:
        self.scenario_set = scenario_set
        self.adversary = adversary
        self.max_iter = max_iter

    def optimize_placement(
        self, assets: List[Asset], locations: List[Location]
    ) -> Tuple[Dict[str, str], ScenarioSet]:
        """Return (placement, adversarial_scenario_set) at convergence."""
        current_scenarios = self.scenario_set
        assignment: Dict[str, str] = {}
        for _ in range(self.max_iter):
            new_assignment = CEVOptimizer(current_scenarios).optimize_placement(
                assets, locations
            )
            updated = self.adversary.update_scenarios(
                current_scenarios, new_assignment, locations
            )
            if new_assignment == assignment:
                break
            assignment = new_assignment
            current_scenarios = updated
        return assignment, current_scenarios

    def expected_posture_efficiency(
        self,
        assets: List[Asset],
        locations: List[Location],
        assignment: Dict[str, str],
        adversarial_scenarios: ScenarioSet,
    ) -> float:
        """Evaluate placement under adversarially-updated scenarios."""
        return CEVOptimizer(adversarial_scenarios).expected_posture_efficiency(
            assets, locations, assignment
        )
