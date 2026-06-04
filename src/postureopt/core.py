"""Core data structures and optimization logic for PostureOpt."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetType(str, Enum):
    """Types of military/logistics assets managed in a posture plan."""

    AIRCRAFT = "AIRCRAFT"
    FUEL_DEPOT = "FUEL_DEPOT"
    MAINTENANCE_CREW = "MAINTENANCE_CREW"
    MUNITION = "MUNITION"
    MEDICAL = "MEDICAL"


class SustainmentAction(str, Enum):
    """Actions available in the sustainment policy."""

    REPOSITION = "REPOSITION"
    RESUPPLY = "RESUPPLY"
    MAINTAIN = "MAINTAIN"
    HOLD = "HOLD"


@dataclass
class Location:
    """A geographic location that can host assets."""

    location_id: str
    name: str
    lat: float
    lon: float
    capacity: int = 100

    def __post_init__(self) -> None:
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"lat must be in [-90, 90], got {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"lon must be in [-180, 180], got {self.lon}")


@dataclass
class Asset:
    """A single asset unit with type, location, and readiness attributes."""

    asset_id: str
    asset_type: AssetType
    location_id: str
    quantity: int = 1
    readiness_rate: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.readiness_rate <= 1.0):
            raise ValueError(
                f"readiness_rate must be in [0, 1], got {self.readiness_rate}"
            )
        if self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")


@dataclass
class PostureState:
    """Snapshot of the overall posture at a given time step."""

    state_id: str
    assets: list[Asset] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    time_step: int = 0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two geographic points in km.

    Uses the Haversine formula.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class PostureOptimizer:
    """Optimizer for asset placement and sustainment policy.

    This class provides stub implementations for placement optimization
    and replenishment policy selection. Production implementations would
    integrate RL agents or combinatorial search backends.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def optimize_placement(
        self,
        assets: list[Asset],
        locations: list[Location],
        demand: dict[str, Any],
    ) -> dict[str, str]:
        """Assign assets to locations to maximize coverage and readiness.

        Args:
            assets: List of Asset objects to place.
            locations: Available Location objects.
            demand: Demand signal per location_id (e.g., {location_id: priority}).

        Returns:
            Mapping of asset_id -> location_id.
        """
        if not locations:
            return {}
        # Stub: assign all assets to first location
        default_loc = locations[0].location_id
        return {asset.asset_id: default_loc for asset in assets}

    def replenishment_policy(self, state: PostureState) -> list[SustainmentAction]:
        """Derive a sustainment action for each asset in the current state.

        Args:
            state: Current PostureState snapshot.

        Returns:
            List of SustainmentAction, one per asset.
        """
        actions: list[SustainmentAction] = []
        for asset in state.assets:
            if asset.readiness_rate < 0.5:
                actions.append(SustainmentAction.MAINTAIN)
            elif asset.readiness_rate < 0.8:
                actions.append(SustainmentAction.RESUPPLY)
            else:
                actions.append(SustainmentAction.HOLD)
        return actions
