"""Evaluation metrics for postureopt."""
from __future__ import annotations

import math
from typing import List, Dict

from .core import Asset, Location, SustainmentAction, PostureState

ACTION_COSTS: Dict[str, float] = {
    "REPOSITION": 10.0,
    "RESUPPLY": 5.0,
    "MAINTAIN": 2.0,
    "HOLD": 0.0,
}


def readiness_score(assets: List[Asset]) -> float:
    """Quantity-weighted mean readiness rate."""
    total_qty = sum(a.quantity for a in assets)
    if total_qty == 0:
        return 0.0
    return sum(a.readiness_rate * a.quantity for a in assets) / total_qty


def coverage_score(assets: List[Asset], locations: List[Location]) -> float:
    """Fraction of locations with at least one asset."""
    if not locations:
        return 0.0
    covered = {a.location_id for a in assets}
    loc_ids = {loc.location_id for loc in locations}
    return len(covered & loc_ids) / len(loc_ids)


def sustainment_cost(actions: List[SustainmentAction]) -> float:
    """Sum of action costs."""
    return sum(ACTION_COSTS.get(a.value, 0.0) for a in actions)


def posture_efficiency(readiness: float, coverage: float, cost: float) -> float:
    """Composite efficiency metric."""
    return (readiness * coverage) / math.log1p(cost + 1.0)


def placement_quality(
    assignments: Dict[str, str],
    assets: List[Asset],
    locations: List[Location],
) -> float:
    """Mean strategic value of assigned locations."""
    loc_map = {loc.location_id: loc.strategic_value for loc in locations}
    values = [loc_map[lid] for lid in assignments.values() if lid in loc_map]
    if not values:
        return 0.0
    return sum(values) / len(values)


def simulation_summary(states: List[PostureState]) -> Dict:
    """Aggregate readiness across simulation steps."""
    if not states:
        return {"n_steps": 0, "mean_readiness": 0.0, "min_readiness": 0.0, "final_readiness": 0.0}
    readiness_vals = [s.total_readiness() for s in states]
    return {
        "n_steps": len(states),
        "mean_readiness": sum(readiness_vals) / len(readiness_vals),
        "min_readiness": min(readiness_vals),
        "final_readiness": readiness_vals[-1],
    }
