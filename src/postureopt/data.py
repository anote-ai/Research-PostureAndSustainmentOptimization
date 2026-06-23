"""Synthetic data generation for postureopt."""
from __future__ import annotations

import copy
import math
import random
from typing import List

from .core import (
    Asset,
    AssetType,
    Location,
    PostureState,
    ReplenishmentPolicy,
    ResourceProfile,
    SLATarget,
    SustainmentAction,
    TelemetrySnapshot,
)
from .drsp import ScenarioSet, ThreatScenario

THEATER_LOCATIONS: List[dict] = [
    {"name": "Kadena AB", "lat": 26.35, "lon": 127.77, "strategic_value": 0.95},
    {"name": "Andersen AFB", "lat": 13.58, "lon": 144.93, "strategic_value": 0.90},
    {"name": "MCAS Iwakuni", "lat": 34.14, "lon": 132.24, "strategic_value": 0.85},
    {"name": "Camp H.M. Smith", "lat": 21.41, "lon": -157.93, "strategic_value": 0.80},
    {"name": "Diego Garcia", "lat": -7.32, "lon": 72.42, "strategic_value": 0.78},
    {"name": "Misawa AB", "lat": 40.70, "lon": 141.37, "strategic_value": 0.75},
    {"name": "Osan AB", "lat": 37.09, "lon": 127.03, "strategic_value": 0.88},
    {"name": "Clark AB", "lat": 15.19, "lon": 120.56, "strategic_value": 0.72},
]

# Realistic cloud instance profiles
INSTANCE_PROFILES: List[dict] = [
    {"profile_id": "c5.xlarge", "cpu_cores": 4, "memory_gb": 8.0, "gpu_count": 0, "hourly_cost_usd": 0.17},
    {"profile_id": "c5.4xlarge", "cpu_cores": 16, "memory_gb": 32.0, "gpu_count": 0, "hourly_cost_usd": 0.68},
    {"profile_id": "p3.2xlarge", "cpu_cores": 8, "memory_gb": 61.0, "gpu_count": 1, "hourly_cost_usd": 3.06},
    {"profile_id": "p3.8xlarge", "cpu_cores": 32, "memory_gb": 244.0, "gpu_count": 4, "hourly_cost_usd": 12.24},
    {"profile_id": "g4dn.xlarge", "cpu_cores": 4, "memory_gb": 16.0, "gpu_count": 1, "hourly_cost_usd": 0.526},
    {"profile_id": "g4dn.12xlarge", "cpu_cores": 48, "memory_gb": 192.0, "gpu_count": 4, "hourly_cost_usd": 3.912},
]


def make_location(i: int = 0) -> Location:
    d = THEATER_LOCATIONS[i % len(THEATER_LOCATIONS)]
    return Location(
        location_id=f"L{i:03d}",
        name=d["name"],
        lat=d["lat"],
        lon=d["lon"],
        capacity=20,
        strategic_value=d["strategic_value"],
    )


def make_asset(
    asset_type: AssetType = AssetType.AIRCRAFT,
    location_id: str = "L001",
    readiness: float = 0.85,
    seed: int = 42,
) -> Asset:
    rng = random.Random(seed)
    return Asset(
        asset_id=f"asset_{rng.randint(1000,9999)}",
        asset_type=asset_type,
        location_id=location_id,
        quantity=rng.randint(2, 10),
        readiness_rate=min(max(readiness, 0.0), 1.0),
        maintenance_days_remaining=rng.randint(5, 60),
    )


def make_posture_state(n_assets: int = 20, seed: int = 42) -> PostureState:
    rng = random.Random(seed)
    locations = [make_location(i) for i in range(5)]
    assets = []
    asset_types = list(AssetType)
    for i in range(n_assets):
        loc = rng.choice(locations)
        atype = rng.choice(asset_types)
        assets.append(Asset(
            asset_id=f"A{i:04d}",
            asset_type=atype,
            location_id=loc.location_id,
            quantity=rng.randint(1, 10),
            readiness_rate=round(rng.uniform(0.4, 1.0), 3),
            maintenance_days_remaining=rng.randint(1, 90),
        ))
    return PostureState(assets=assets, locations=locations, time_step=0)


def simulate_degradation(
    state: PostureState, n_steps: int = 5, seed: int = 42
) -> List[PostureState]:
    """Simulate asset degradation + replenishment over time steps."""
    rng = random.Random(seed)
    policy = ReplenishmentPolicy()
    history = []
    current = copy.deepcopy(state)
    for step in range(n_steps):
        new_assets = []
        for asset in current.assets:
            degradation = rng.uniform(0.05, 0.15)
            new_readiness = max(0.0, asset.readiness_rate - degradation)
            new_days = max(0, asset.maintenance_days_remaining - 1)
            a = Asset(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                location_id=asset.location_id,
                quantity=asset.quantity,
                readiness_rate=round(new_readiness, 4),
                maintenance_days_remaining=new_days,
            )
            action = policy.decide(a)
            if action == SustainmentAction.MAINTAIN:
                a.readiness_rate = min(1.0, a.readiness_rate + 0.2)
            elif action == SustainmentAction.RESUPPLY:
                a.quantity += 2
            new_assets.append(a)
        current = PostureState(
            assets=new_assets,
            locations=current.locations,
            time_step=step + 1,
        )
        history.append(copy.deepcopy(current))
    return history


# ---------------------------------------------------------------------------
# Infrastructure telemetry generation
# ---------------------------------------------------------------------------


def make_resource_profile(idx: int = 0) -> ResourceProfile:
    """Return a ResourceProfile from the standard instance catalogue."""
    d = INSTANCE_PROFILES[idx % len(INSTANCE_PROFILES)]
    return ResourceProfile(**d)


def make_all_resource_profiles() -> List[ResourceProfile]:
    """Return all catalogued resource profiles."""
    return [ResourceProfile(**d) for d in INSTANCE_PROFILES]


def make_sla_target(
    max_latency_ms: float = 200.0,
    min_availability: float = 0.999,
    max_error_rate: float = 0.01,
    cost_per_hour_usd: float = 5.0,
) -> SLATarget:
    return SLATarget(
        sla_id="default-sla",
        max_latency_ms=max_latency_ms,
        min_availability=min_availability,
        max_error_rate=max_error_rate,
        cost_per_hour_usd=cost_per_hour_usd,
    )


def make_telemetry_snapshot(seed: int = 42, load_factor: float = 0.5) -> TelemetrySnapshot:
    """Generate a single telemetry snapshot at a given load factor."""
    rng = random.Random(seed)
    # Higher load => higher latency, error rate, and cost
    base_latency = 20.0 + load_factor * 400.0
    jitter = rng.uniform(-10.0, 10.0)
    return TelemetrySnapshot(
        snapshot_id=f"snap-{rng.randint(1000, 9999)}",
        timestamp_s=rng.uniform(0, 3600),
        p99_latency_ms=max(10.0, base_latency + jitter),
        availability=max(0.90, min(1.0, 1.0 - load_factor * 0.01 + rng.uniform(-0.001, 0.001))),
        error_rate=max(0.0, min(1.0, load_factor * 0.02 + rng.uniform(0.0, 0.005))),
        cpu_utilization=round(min(0.99, load_factor + rng.uniform(-0.05, 0.05)), 3),
        memory_utilization=round(min(0.99, load_factor * 0.8 + rng.uniform(-0.05, 0.05)), 3),
        gpu_utilization=round(min(0.99, load_factor * 0.6 + rng.uniform(-0.05, 0.05)), 3),
        cost_per_hour_usd=round(1.0 + load_factor * 8.0 + rng.uniform(-0.5, 0.5), 2),
        requests_per_second=round(load_factor * 500.0 + rng.uniform(-10.0, 10.0), 1),
    )


def make_telemetry_series(
    n_snapshots: int = 24,
    seed: int = 42,
    diurnal: bool = True,
) -> List[TelemetrySnapshot]:
    """Generate a time-series of telemetry snapshots.

    When diurnal=True, load follows a sinusoidal pattern simulating day/night cycles.
    """
    rng = random.Random(seed)
    snapshots: List[TelemetrySnapshot] = []
    for i in range(n_snapshots):
        if diurnal:
            # Peak load at step n/2
            load_factor = 0.3 + 0.5 * math.sin(math.pi * i / max(n_snapshots - 1, 1))
        else:
            load_factor = rng.uniform(0.2, 0.9)
        snap = make_telemetry_snapshot(seed=rng.randint(0, 99999), load_factor=load_factor)
        snapshots.append(snap)
    return snapshots


# ---------------------------------------------------------------------------
# Experiment 3 — scenario-set factories
# ---------------------------------------------------------------------------

def make_uniform_scenario_set(locations: List[Location], n_scenarios: int = 5) -> ScenarioSet:
    """Uniform threat: all locations equally contested across all scenarios.

    CEV and greedy should behave similarly here — used as a control condition.
    """
    loc_ids = [loc.location_id for loc in locations]
    prob = 1.0 / n_scenarios
    scenarios = [
        ThreatScenario(
            scenario_id=f"U{i}",
            threat_weights={lid: round(0.1 + 0.1 * i / max(n_scenarios - 1, 1), 3) for lid in loc_ids},
            probability=prob,
        )
        for i in range(n_scenarios)
    ]
    return ScenarioSet(scenarios=scenarios)


def make_skewed_scenario_set(locations: List[Location]) -> ScenarioSet:
    """High threat concentrated on the top-strategic-value location (80% weight).

    Forces the optimizer away from the greedy-preferred location.
    """
    sorted_locs = sorted(locations, key=lambda l: -l.strategic_value)
    top_id = sorted_locs[0].location_id
    return ScenarioSet(scenarios=[
        ThreatScenario(
            scenario_id="SK_high",
            threat_weights={top_id: 0.95},
            probability=0.8,
        ),
        ThreatScenario(
            scenario_id="SK_low",
            threat_weights={lid: 0.05 for lid in (loc.location_id for loc in locations)},
            probability=0.2,
        ),
    ])


def make_adversarial_scenario_set(locations: List[Location]) -> ScenarioSet:
    """Adversarially-shaped prior: threat is spread across top-two locations with
    high demand multipliers, forcing the defender to hedge.
    """
    sorted_locs = sorted(locations, key=lambda l: -l.strategic_value)
    top_id = sorted_locs[0].location_id
    second_id = sorted_locs[1].location_id if len(sorted_locs) > 1 else top_id
    return ScenarioSet(scenarios=[
        ThreatScenario(
            scenario_id="ADV_atk_top",
            threat_weights={top_id: 0.90},
            probability=0.4,
            demand_multiplier=1.5,
        ),
        ThreatScenario(
            scenario_id="ADV_atk_second",
            threat_weights={second_id: 0.85},
            probability=0.35,
            demand_multiplier=1.3,
        ),
        ThreatScenario(
            scenario_id="ADV_quiet",
            threat_weights={lid: 0.05 for lid in (loc.location_id for loc in locations)},
            probability=0.25,
            demand_multiplier=1.0,
        ),
    ])


def make_deceptive_scenario_set(locations: List[Location]) -> ScenarioSet:
    """Deceptive prior that lures naive CEV into a vulnerable concentration.

    The prior looks mostly safe (90% weight on a no-threat scenario), so naive CEV
    piles assets at the top-strategic-value location.  The remaining 10% is a
    high-intensity attack on exactly that location.  A rational adversary who
    observes the resulting concentration immediately shifts all weight to the attack
    scenario, collapsing naive efficiency.  Robust CEV iterates away from the lure.

    This is the distribution that most clearly demonstrates the paper's core claim:
    at high p_obs, naive CEV is exploitable; robust CEV maintains a higher floor.
    """
    sorted_locs = sorted(locations, key=lambda l: -l.strategic_value)
    top_id = sorted_locs[0].location_id
    return ScenarioSet(scenarios=[
        ThreatScenario(
            scenario_id="DEC_safe",
            threat_weights={},
            probability=0.95,
        ),
        ThreatScenario(
            scenario_id="DEC_atk_top",
            threat_weights={top_id: 0.99},
            probability=0.05,
            demand_multiplier=2.0,
        ),
    ])
