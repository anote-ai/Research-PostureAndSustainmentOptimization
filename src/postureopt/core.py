"""Core data structures and optimization logic for postureopt."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


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


# ---------------------------------------------------------------------------
# SLA and infrastructure telemetry
# ---------------------------------------------------------------------------


@dataclass
class SLATarget:
    """Service Level Agreement targets for an AI model deployment."""

    sla_id: str
    max_latency_ms: float  # P99 latency budget
    min_availability: float  # fraction [0, 1]
    max_error_rate: float  # fraction [0, 1]
    cost_per_hour_usd: float  # maximum allowed hourly cost

    def __post_init__(self) -> None:
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be > 0")
        for attr in ("min_availability", "max_error_rate"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass
class TelemetrySnapshot:
    """Observed metrics for one time window."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_s: float = 0.0
    p99_latency_ms: float = 100.0
    availability: float = 0.999
    error_rate: float = 0.001
    cpu_utilization: float = 0.50  # [0, 1]
    memory_utilization: float = 0.60  # [0, 1]
    gpu_utilization: float = 0.0   # [0, 1]
    cost_per_hour_usd: float = 10.0
    requests_per_second: float = 100.0

    def __post_init__(self) -> None:
        for attr in ("availability", "error_rate", "cpu_utilization",
                     "memory_utilization", "gpu_utilization"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass
class ResourceProfile:
    """Describes hardware resources allocated to a deployment."""

    profile_id: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    instance_type: str = "standard"
    # Hourly on-demand cost
    hourly_cost_usd: float = 1.0

    def __post_init__(self) -> None:
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be > 0")


# ---------------------------------------------------------------------------
# Cost-latency trade-off model
# ---------------------------------------------------------------------------


def estimated_latency_ms(
    profile: ResourceProfile,
    requests_per_second: float,
    base_latency_ms: float = 10.0,
) -> float:
    """Simple queueing-theory-inspired latency estimate.

    Models P99 latency as base_latency + load-driven inflation.
    As CPU utilization approaches 1, latency grows nonlinearly.
    """
    capacity_rps = profile.cpu_cores * 50.0 + profile.gpu_count * 500.0
    utilization = min(requests_per_second / max(capacity_rps, 1e-9), 0.999)
    # M/M/1 queueing approximation
    return base_latency_ms / (1.0 - utilization)


def estimated_hourly_cost(
    profile: ResourceProfile,
    gpu_surcharge_per_hour: float = 2.5,
) -> float:
    """Estimate total hourly cost including GPU surcharge."""
    return profile.hourly_cost_usd + profile.gpu_count * gpu_surcharge_per_hour


def resource_utilization_score(snapshots: List[TelemetrySnapshot]) -> float:
    """Mean composite resource utilization across snapshots.

    Combines CPU, memory, and GPU into a single [0, 1] value.
    GPU is weighted higher when present.
    """
    if not snapshots:
        return 0.0
    scores: List[float] = []
    for s in snapshots:
        if s.gpu_utilization > 0:
            score = (s.cpu_utilization * 0.3 + s.memory_utilization * 0.3 + s.gpu_utilization * 0.4)
        else:
            score = (s.cpu_utilization * 0.5 + s.memory_utilization * 0.5)
        scores.append(score)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Posture utilities
# ---------------------------------------------------------------------------


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

    def random_placement(
        self, assets: List[Asset], locations: List[Location]
    ) -> Dict[str, str]:
        """Assign each asset to a uniformly random location with remaining capacity."""
        import random as _random
        rng = _random.Random(self.seed)
        capacity_remaining = {loc.location_id: loc.capacity for loc in locations}
        loc_ids = [loc.location_id for loc in locations]
        assignment: Dict[str, str] = {}
        for asset in assets:
            available = [l for l in loc_ids if capacity_remaining[l] > 0]
            if not available:
                break
            chosen = rng.choice(available)
            assignment[asset.asset_id] = chosen
            capacity_remaining[chosen] -= 1
        return assignment

    def greedy_placement(
        self, assets: List[Asset], locations: List[Location]
    ) -> Dict[str, str]:
        """Assign each asset to the highest-value location with remaining capacity."""
        sorted_locs = sorted(locations, key=lambda loc: -loc.strategic_value)
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
