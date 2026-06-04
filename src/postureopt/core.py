"""Core data structures and optimization logic for postureopt."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple


class AssetType(str, Enum):
    AIRCRAFT = "AIRCRAFT"
    FUEL_DEPOT = "FUEL_DEPOT"
    MAINTENANCE_CREW = "MAINTENANCE_CREW"
    MUNITION = "MUNITION"
    MEDICAL = "MEDICAL"


class SustainmentAction(str, Enum):
    REPOSITION = "REPOSITION"
    RESUPPLY = "RESUPPLY"
    MAINTAIN = "MAINTAIN"
    HOLD = "HOLD"


@dataclass
class Location:
    location_id: str
    name: str
    lat: float
    lon: float
    capacity: int
    strategic_value: float = 0.5


@dataclass
class Asset:
    asset_id: str
    asset_type: AssetType
    location_id: str
    quantity: int
    readiness_rate: float  # [0, 1]
    maintenance_days_remaining: int = 30

    def __post_init__(self) -> None:
        if not (0.0 <= self.readiness_rate <= 1.0):
            raise ValueError(f"readiness_rate must be in [0,1], got {self.readiness_rate}")


@dataclass
class PostureState:
    state_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    assets: List[Asset] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    time_step: int = 0

    def assets_at(self, location_id: str) -> List[Asset]:
        return [a for a in self.assets if a.location_id == location_id]

    def total_readiness(self) -> float:
        if not self.assets:
            return 0.0
        total_qty = sum(a.quantity for a in self.assets)
        if total_qty == 0:
            return 0.0
        return sum(a.readiness_rate * a.quantity for a in self.assets) / total_qty


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ReplenishmentPolicy:
    """Rule-based replenishment policy."""

    def decide(self, asset: Asset) -> SustainmentAction:
        if asset.readiness_rate < 0.4:
            return SustainmentAction.MAINTAIN
        if asset.quantity < 2:
            return SustainmentAction.RESUPPLY
        if asset.maintenance_days_remaining < 7:
            return SustainmentAction.MAINTAIN
        return SustainmentAction.HOLD


class PostureOptimizer:
    """Greedy posture optimization."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def greedy_placement(
        self, assets: List[Asset], locations: List[Location]
    ) -> Dict[str, str]:
        """Assign each asset to the highest-value location with remaining capacity."""
        sorted_locs = sorted(locations, key=lambda l: -l.strategic_value)
        capacity_remaining = {loc.location_id: loc.capacity for loc in sorted_locs}
        assignment: Dict[str, str] = {}
        for asset in assets:
            for loc in sorted_locs:
                if capacity_remaining[loc.location_id] > 0:
                    assignment[asset.asset_id] = loc.location_id
                    capacity_remaining[loc.location_id] -= 1
                    break
        return assignment

    def optimize_replenishment(
        self, state: PostureState
    ) -> List[Tuple[Asset, SustainmentAction]]:
        """Apply ReplenishmentPolicy to all assets."""
        policy = ReplenishmentPolicy()
        return [(asset, policy.decide(asset)) for asset in state.assets]
