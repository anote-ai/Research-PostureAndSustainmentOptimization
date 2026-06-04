"""Evaluation metrics for posture and sustainment optimization."""

from __future__ import annotations

import math

from postureopt.core import Asset, Location, SustainmentAction

# Cost model for sustainment actions
_ACTION_COSTS: dict[SustainmentAction, float] = {
    SustainmentAction.REPOSITION: 10.0,
    SustainmentAction.RESUPPLY: 5.0,
    SustainmentAction.MAINTAIN: 2.0,
    SustainmentAction.HOLD: 0.0,
}


def readiness_score(assets: list[Asset]) -> float:
    """Compute the weighted mean readiness rate across all assets.

    Weights are proportional to asset quantity.

    Args:
        assets: List of Asset objects.

    Returns:
        Weighted mean readiness rate in [0, 1], or 0.0 if list is empty.
    """
    if not assets:
        return 0.0
    total_qty = sum(a.quantity for a in assets)
    if total_qty == 0:
        return 0.0
    return sum(a.readiness_rate * a.quantity for a in assets) / total_qty


def coverage_score(assets: list[Asset], locations: list[Location]) -> float:
    """Compute fraction of locations that have at least one asset assigned.

    Args:
        assets: List of Asset objects with location_id assignments.
        locations: All possible Location objects.

    Returns:
        Coverage fraction in [0, 1], or 0.0 if no locations.
    """
    if not locations:
        return 0.0
    covered_ids = {a.location_id for a in assets}
    covered = sum(1 for loc in locations if loc.location_id in covered_ids)
    return covered / len(locations)


def sustainment_cost(actions: list[SustainmentAction]) -> float:
    """Compute total cost of a list of sustainment actions.

    Args:
        actions: List of SustainmentAction values.

    Returns:
        Total cost as float.
    """
    return sum(_ACTION_COSTS[action] for action in actions)


def posture_efficiency(readiness: float, coverage: float, cost: float) -> float:
    """Compute composite posture efficiency score.

    Efficiency = (readiness * coverage) / log1p(cost)

    Higher is better. Returns 0.0 when cost makes the denominator undefined
    (cost < -1) or when readiness/coverage are zero.

    Args:
        readiness: Readiness score in [0, 1].
        coverage: Coverage score in [0, 1].
        cost: Total sustainment cost (non-negative).

    Returns:
        Efficiency scalar.
    """
    denom = math.log1p(max(cost, 0.0))
    if denom == 0.0:
        # cost == 0 -> log1p(0) == 0; avoid divide by zero; return raw product
        return readiness * coverage
    return (readiness * coverage) / denom
